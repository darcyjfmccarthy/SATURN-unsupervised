"""Bounded bf16 training checks preserve fp32 state, losses and public arrays."""

from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jax_saturn.models.precision import matmul_dtype, resolve_precision
from jax_saturn.models.saturn import SaturnPretrainModule
from jax_saturn.training.baseline import train_baseline
from jax_saturn.training.label_agnostic import LabelConfig, train_label_free
from jax_saturn.training.pretrain import emit_pretrain_anndata, train_pretrain
from jax_saturn.verification.test_pretrain import tiny_prepared_fixture
from label_agnostic.artifacts import save_label_free_artifact


class PrecisionChecks(unittest.TestCase):
    def assert_float_state(self, state):
        for value in jax.tree_util.tree_leaves((state.params, state.opt_state)):
            if jnp.issubdtype(value.dtype, jnp.floating):
                self.assertEqual(value.dtype, jnp.float32)
                self.assertTrue(np.isfinite(value).all())

    def test_policy_and_invalid_values(self):
        self.assertEqual(resolve_precision(platform="tpu"), "bf16")
        for platform in ("cpu", "gpu", "cuda", None):
            self.assertEqual(resolve_precision(platform=platform), "fp32")
        self.assertEqual(resolve_precision("fp32", platform="tpu"), "fp32")
        self.assertEqual(matmul_dtype("bf16"), jnp.bfloat16)
        with self.assertRaises(ValueError):
            resolve_precision("fp16")

    def test_bf16_all_stages_outputs_and_resume_policy(self):
        self.check_all_stages()

    def check_all_stages(self, devices=None):
        data, scores = tiny_prepared_fixture()
        module = SaturnPretrainModule(data.species_names, data.gene_counts, 4, 7, 5, dtype=jnp.bfloat16)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pretrain, history = train_pretrain(module, data, scores, output_dir=root / "pretrain", epochs=1, batch_size=8, devices=devices)
            self.assert_float_state(pretrain)
            self.assertTrue(all(np.isfinite(value) for key, value in history[0].items()))
            metadata = json.loads((root / "pretrain/checkpoints/epoch_0001/metadata.json").read_text())
            self.assertEqual(metadata["hyperparameters"]["dtype"], "bfloat16")
            atlas = emit_pretrain_anndata(module, pretrain.params, data, root / "truth.h5ad", batch_size=8, devices=devices)
            self.assertEqual(atlas.X.dtype, np.float32)
            self.assertEqual(atlas.obsm["macrogenes"].dtype, np.float32)
            baseline, _, final = train_baseline(pretrain.params, atlas, output_dir=root / "baseline/results",
                epochs=1, batch_size=8, hidden_dim=7, model_dim=5, mixed_precision="bf16",
                source_manifest_sha256=data.source_manifest_sha256, devices=devices)
            self.assert_float_state(baseline)
            self.assertEqual(final.X.dtype, np.float32)
            baseline_checkpoint = root / "baseline/shared/metric_orbax/checkpoints/epoch_0001"
            with self.assertRaises(ValueError):
                train_baseline(pretrain.params, atlas, output_dir=root / "baseline_wrong",
                    epochs=1, batch_size=8, hidden_dim=7, model_dim=5, mixed_precision="fp32",
                    source_manifest_sha256=data.source_manifest_sha256, resume=baseline_checkpoint, devices=devices)
            artifact = root / "artifact.npz"
            save_label_free_artifact(artifact, atlas.X, atlas.obsm["macrogenes"], atlas.obs["species"], atlas.obs_names)
            config = LabelConfig(epochs=1, batch_size=8, hidden_dim=7, model_dim=5,
                                 ot_iterations=15, mixed_precision="bf16")
            for objective in ("infonce", "mmd", "ot"):
                state, history, summary = train_label_free(artifact, root / "pretrain/checkpoints/epoch_0001",
                    root / objective, objective=objective, config=config, devices=devices)
                self.assert_float_state(state)
                self.assertTrue(np.isfinite(history[0]["metric_loss"]))
                self.assertEqual(summary["configuration"]["mixed_precision"], "bf16")
                metadata = json.loads((root / objective / "checkpoints/epoch_0001/metadata.json").read_text())
                self.assertEqual(metadata["hyperparameters"]["mixed_precision"], "bf16")
                with np.load(root / objective / "final_embeddings.npz", allow_pickle=False) as arrays:
                    self.assertEqual(arrays["embeddings"].dtype, np.float32)
                    self.assertEqual(arrays["embeddings"].shape, (16, 5))
                with self.assertRaises(ValueError):
                    train_label_free(artifact, root / "pretrain/checkpoints/epoch_0001", root / "wrong_precision",
                        objective=objective, config=replace(config, mixed_precision="fp32"),
                        resume=root / objective / "checkpoints/epoch_0001", devices=devices)


class PrecisionProcessChecks(unittest.TestCase):
    def test_bf16_two_device_process(self):
        code = """
import jax
from jax_saturn.verification.test_precision import PrecisionChecks
devices = jax.local_devices()
assert len(devices) == 2
PrecisionChecks().check_all_stages(devices=devices)
print('All five bf16 training stages passed on two CPU devices')
"""
        result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[2],
            env={**os.environ, "JAX_PLATFORMS": "cpu", "XLA_FLAGS": "--xla_force_host_platform_device_count=2"},
            capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()

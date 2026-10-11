"""Two local CPU devices exercise pmap without cloud capacity."""

import os
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import anndata as ad
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

from jax_saturn.distributed.metric import (
    make_metric_pmap, replicate_state, unreplicate_state, validate_devices,
)
from jax_saturn.distributed.pretrain import make_pretrain_pmap
from jax_saturn.distributed.label_agnostic import make_label_pmap, replicate_training_state
from jax_saturn.distributed.inference import _metric_kernel, _pretrain_kernel
from jax_saturn.losses.core import normalize, triplet_margin_loss
from jax_saturn.models.saturn import SaturnMetricModule, SaturnPretrainModule
from jax_saturn.training.baseline import create_metric_state, embed_metric, make_metric_step, train_baseline
from jax_saturn.verification.test_baseline import fixture
from jax_saturn.training.pretrain import create_state, emit_pretrain_anndata, make_pretrain_step, train_pretrain
from jax_saturn.training.label_agnostic import LabelAgnosticTrainState, LabelConfig, build_graphs, make_label_step, train_label_free
from jax_saturn.verification.test_pretrain import tiny_prepared_fixture
from label_agnostic.artifacts import save_label_free_artifact


@unittest.skipUnless(os.environ.get("SATURN_PMAP_TEST_CHILD") == "1", "runs inside the two-device CPU subprocess")
class MetricPmapChecks(unittest.TestCase):
    def setUp(self):
        self.devices = jax.local_devices()
        self.assertEqual(len(self.devices), 2)

    def test_global_triplets_and_three_adam_updates_match_single_device(self):
        values, _, _ = fixture()
        module = SaturnMetricModule(7, 9, 5, dropout=0.)
        params = module.init(jax.random.key(3), values)["params"]
        optimizer = optax.adam(.001, eps=1e-8)
        single = create_metric_state(params, optimizer, 0)
        parallel = replicate_state(single, self.devices)
        single_preview, single_step = make_metric_step(module, optimizer)
        preview, step = make_metric_pmap(module, optimizer, self.devices)
        np.testing.assert_allclose(preview(parallel, values), single_preview(single, values), atol=1e-5)
        # Positives and negatives cross the shard boundary; active counts vary.
        indices = tuple(jnp.asarray(array) for array in (
            [0, 1, 4, 5, 8, 0, 0, 0], [7, 8, 10, 11, 2, 0, 0, 0], [2, 3, 6, 7, 0, 0, 0, 0]))
        for update in range(3):
            mask = jnp.array([True] * 5 + [False] * 3) if update < 2 else jnp.zeros(8, dtype=bool)
            single, expected_metrics = single_step(single, values, indices, mask)
            parallel, metrics = step(parallel, values, indices, mask)
            for key in expected_metrics:
                np.testing.assert_allclose(metrics[key], expected_metrics[key], rtol=1e-4, atol=1e-5)
            for actual, expected in zip(jax.tree_util.tree_leaves(unreplicate_state(parallel)),
                                        jax.tree_util.tree_leaves(single)):
                if jax.dtypes.issubdtype(actual.dtype, jax.dtypes.prng_key):
                    actual, expected = jax.random.key_data(actual), jax.random.key_data(expected)
                np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
            for leaf in jax.tree_util.tree_leaves(parallel.params):
                np.testing.assert_array_equal(leaf[0], leaf[1])

    def test_preview_and_update_reuse_shard_dropout(self):
        values, _, _ = fixture()
        module = SaturnMetricModule(7, 9, 5, dropout=.1)
        params = module.init(jax.random.key(5), values)["params"]
        optimizer = optax.adam(.001)
        state = replicate_state(create_metric_state(params, optimizer, 0), self.devices)
        preview, step = make_metric_pmap(module, optimizer, self.devices)
        indices = tuple(jnp.array(array) for array in ([0, 3, 7], [7, 9, 2], [2, 1, 10]))
        mask = jnp.array([True, True, False])
        expected = triplet_margin_loss(normalize(preview(state, values)), indices, valid_mask=mask)
        _, metrics = step(state, values, indices, mask)
        np.testing.assert_allclose(metrics["loss"], expected, rtol=1e-5, atol=1e-6)
        with self.assertRaises(ValueError):
            preview(state, values[:11])
        with self.assertRaises(ValueError):
            validate_devices([self.devices[0], self.devices[0]])

    def test_training_padding_checkpoint_and_resume(self):
        values, labels, species = fixture()
        macros = np.abs(values)
        obs = pd.DataFrame(index=[f"cell_{i}" for i in range(len(values))])
        for key, column in (("labels", labels), ("labels2", labels % 2),
                            ("ref_labels", labels % 2), ("species", species)):
            obs[key] = pd.Categorical(column.astype(str))
        atlas = ad.AnnData(np.zeros((12, 5), dtype=np.float32), obs=obs, obsm={"macrogenes": macros})
        module = SaturnMetricModule(7, 9, 5)
        params = module.init(jax.random.key(1), macros)["params"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = dict(epochs=2, batch_size=8, hidden_dim=9, model_dim=5,
                             source_manifest_sha256="a" * 64, devices=self.devices)
            full, history, final = train_baseline(params, atlas, output_dir=root / "full/results", **arguments)
            resumed, resumed_history, restored = train_baseline(
                params, atlas, output_dir=root / "copy/results",
                resume=root / "full/shared/metric_orbax/checkpoints/epoch_0001", **arguments)
            self.assertEqual(history, resumed_history)
            self.assertEqual(final.shape, (12, 5))
            np.testing.assert_array_equal(final.X, restored.X)
            np.testing.assert_array_equal(final.obs_names, atlas.obs_names)
            for actual, expected in zip(jax.tree_util.tree_leaves(resumed.params), jax.tree_util.tree_leaves(full.params)):
                np.testing.assert_array_equal(actual, expected)
            with self.assertRaises(ValueError):
                train_baseline(params, atlas, output_dir=root / "bad", **{**arguments, "batch_size": 7})
            with self.assertRaises(ValueError):
                train_baseline(params, atlas, output_dir=root / "wrong_topology",
                    resume=root / "full/shared/metric_orbax/checkpoints/epoch_0001",
                    **{**arguments, "devices": self.devices[:1]})


@unittest.skipUnless(os.environ.get("SATURN_PMAP_TEST_CHILD") == "1", "runs inside the two-device CPU subprocess")
class PretrainPmapChecks(unittest.TestCase):
    def setUp(self):
        self.devices = jax.local_devices()
        self.assertEqual(len(self.devices), 2)

    def test_global_valid_counts_and_once_only_regularizers(self):
        data, scores = tiny_prepared_fixture()
        module = SaturnPretrainModule(data.species_names, data.gene_counts, 4, 7, 5)
        optimizer = optax.adam(.0005, eps=1e-8)
        single = create_state(module, scores, optimizer, 0)
        parallel = replicate_state(single, self.devices)
        arguments = dict(l1_penalty=.01, pe_sim_penalty=.2)
        single_step = make_pretrain_step(module, optimizer, data.gene_embeddings, **arguments)
        parallel_step = make_pretrain_pmap(module, optimizer, data.gene_embeddings, self.devices, **arguments)
        rng = np.random.default_rng(55)
        with patch.object(nn.Dropout, "__call__", lambda self, inputs, **kwargs: inputs):
            # Unequal shard counts, an entirely absent species, and a singleton.
            for counts in ((3, 6), (5, 0), (1, 7)):
                batch = []
                for genes, valid_count in zip(data.gene_counts, counts):
                    values = rng.poisson(3, (8, genes)).astype(np.float32)
                    values[valid_count:] = 999.
                    batch.append({"values": jnp.asarray(values), "valid_mask": jnp.arange(8) < valid_count})
                single, expected_metrics = single_step(single, tuple(batch))
                parallel, metrics = parallel_step(parallel, tuple(batch))
                for key in expected_metrics:
                    np.testing.assert_allclose(metrics[key], expected_metrics[key], rtol=1e-4, atol=1e-5, err_msg=key)
                for actual, expected in zip(jax.tree_util.tree_leaves(unreplicate_state(parallel)), jax.tree_util.tree_leaves(single)):
                    if jax.dtypes.issubdtype(actual.dtype, jax.dtypes.prng_key):
                        actual, expected = jax.random.key_data(actual), jax.random.key_data(expected)
                    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
                for leaf in jax.tree_util.tree_leaves(parallel.params):
                    np.testing.assert_array_equal(leaf[0], leaf[1])

    def test_pretraining_resume_and_unpadded_outputs(self):
        data, scores = tiny_prepared_fixture()
        module = SaturnPretrainModule(data.species_names, data.gene_counts, 4, 7, 5)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = dict(epochs=2, batch_size=8, devices=self.devices)
            full, history = train_pretrain(module, data, scores, output_dir=root / "full", **arguments)
            resumed, restored_history = train_pretrain(module, data, scores, output_dir=root / "copy",
                resume=root / "full/checkpoints/epoch_0001", **arguments)
            self.assertEqual(history, restored_history)
            for actual, expected in zip(jax.tree_util.tree_leaves(resumed), jax.tree_util.tree_leaves(full)):
                if jax.dtypes.issubdtype(actual.dtype, jax.dtypes.prng_key):
                    actual, expected = jax.random.key_data(actual), jax.random.key_data(expected)
                np.testing.assert_array_equal(actual, expected)
            atlas = emit_pretrain_anndata(module, resumed.params, data, root / "pretrain.h5ad", batch_size=8)
            self.assertEqual(atlas.shape, (16, 5))
            np.testing.assert_array_equal(atlas.obs_names, data.obs.index)
            with self.assertRaises(ValueError):
                train_pretrain(module, data, scores, output_dir=root / "wrong_topology",
                    resume=root / "full/checkpoints/epoch_0001", **{**arguments, "devices": self.devices[:1]})


@unittest.skipUnless(os.environ.get("SATURN_PMAP_TEST_CHILD") == "1", "runs inside the two-device CPU subprocess")
class LabelPmapChecks(unittest.TestCase):
    def setUp(self):
        self.devices = jax.local_devices()
        self.assertEqual(len(self.devices), 2)

    def test_global_objectives_calibration_and_three_updates(self):
        rng = np.random.default_rng(123)
        artifact = {"embeddings": rng.normal(size=(18, 5)).astype(np.float32),
                    "macrogenes": np.abs(rng.normal(size=(18, 7))).astype(np.float32),
                    "species": np.repeat(["a", "b", "c"], 6),
                    "obs_ids": np.array([f"cell_{i}" for i in range(18)])}
        config = LabelConfig(epochs=2, batch_size=8, hidden_dim=9, model_dim=5,
                             candidate_k=3, positives_per_species=2, local_positive_k=2, ot_iterations=15)
        module = SaturnMetricModule(7, 9, 5, dropout=0.)
        params = module.init(jax.random.key(47), artifact["macrogenes"])["params"]
        optimizer = optax.adam(config.learning_rate)
        bank = normalize(module.apply({"params": params}, artifact["macrogenes"]))
        for objective in ("infonce", "mmd", "ot"):
            state = LabelAgnosticTrainState(params, optimizer.init(params), jax.random.key(0), jnp.int32(0),
                jnp.int32(0), params, 0, 0, 0., 0., .7, {}, {})
            parallel = replicate_training_state(state, self.devices)
            _, codes, graphs, _, bandwidth, _ = build_graphs(artifact, objective, config)
            single_step, single_losses = make_label_step(module, optimizer, objective, graphs, bandwidth, config)
            step, calibration = make_label_pmap(module, optimizer, objective, graphs, bandwidth, config, self.devices)
            # Each shard contains one species: shard-local MMD/OT would be zero.
            selections = ([0, 1, 2, 3, 6, 7, 8, 9], [6, 7, 8, 9, 12, 13, 14, 15], [0, 1, 2, 3, 12, 13, 14, 15])
            for update, (indices, weight) in enumerate(zip(selections, (.7, 1.3, .2))):
                indices = np.asarray(indices, dtype=np.int32)
                mask = np.arange(8) < 8 - update
                values = artifact["macrogenes"][indices].copy()
                values[~mask] = 999.
                batch = {"macrogenes": jnp.asarray(values), "global_indices": jnp.asarray(indices),
                         "species_codes": jnp.asarray(codes[indices]), "valid_mask": jnp.asarray(mask)}
                if update == 0:
                    _, key = jax.random.split(state.rng)
                    actual = calibration(parallel, batch, bank)
                    for position, gradients in enumerate(actual):
                        expected = jax.grad(lambda p: single_losses(p, batch, bank, key)[position])(state.params)
                        for got, want in zip(jax.tree_util.tree_leaves(gradients), jax.tree_util.tree_leaves(expected)):
                            np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)
                state = state.replace(preservation_weight=weight)
                state, expected_metrics = single_step(state, batch, bank)
                parallel, metrics = step(parallel, batch, bank, weight)
                with self.subTest(objective=objective, update=update + 1):
                    for key in expected_metrics:
                        np.testing.assert_allclose(metrics[key], expected_metrics[key], rtol=1e-4, atol=1e-5, err_msg=key)
                    actual = unreplicate_state(parallel)
                    for got, want in zip(jax.tree_util.tree_leaves((actual.params, actual.opt_state, actual.rng, actual.step)),
                                          jax.tree_util.tree_leaves((state.params, state.opt_state, state.rng, state.step))):
                        if jax.dtypes.issubdtype(got.dtype, jax.dtypes.prng_key):
                            got, want = jax.random.key_data(got), jax.random.key_data(want)
                        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)

    def test_all_objective_outputs_and_adaptive_weight_resume(self):
        data, scores = tiny_prepared_fixture()
        module = SaturnPretrainModule(data.species_names, data.gene_counts, 4, 7, 5)
        config = LabelConfig(epochs=1, batch_size=8, hidden_dim=7, model_dim=5, ot_iterations=15)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pretrain, _ = train_pretrain(module, data, scores, output_dir=root / "pretrain", epochs=1, batch_size=8)
            atlas = emit_pretrain_anndata(module, pretrain.params, data, root / "truth.h5ad", batch_size=8)
            artifact = root / "artifact.npz"
            save_label_free_artifact(artifact, atlas.X, atlas.obsm["macrogenes"], atlas.obs["species"], atlas.obs_names)
            checkpoint = root / "pretrain/checkpoints/epoch_0001"
            for objective in ("infonce", "mmd", "ot"):
                trial_config = replace(config, epochs=2) if objective == "mmd" else config
                full, history, summary = train_label_free(artifact, checkpoint, root / objective,
                    objective=objective, config=trial_config, devices=self.devices)
                self.assertEqual(summary["selected_epoch"], 0)
                with np.load(root / objective / "final_embeddings.npz", allow_pickle=False) as final:
                    self.assertEqual(final["embeddings"].shape, (16, 5))
                    np.testing.assert_array_equal(final["obs_ids"], data.obs.index)
                resumed, restored_history, restored_summary = train_label_free(artifact, checkpoint, root / (objective + "_copy"),
                    objective=objective, config=trial_config, devices=self.devices, resume=root / objective / "checkpoints/epoch_0001")
                self.assertEqual(history, restored_history)
                self.assertEqual(full.preservation_weight, resumed.preservation_weight)
                self.assertEqual(full.diagnostics, resumed.diagnostics)
                for got, want in zip(jax.tree_util.tree_leaves(full.params), jax.tree_util.tree_leaves(resumed.params)):
                    np.testing.assert_array_equal(got, want)
                self.assertEqual(summary["selected_epoch"], restored_summary["selected_epoch"])
            config = replace(config, epochs=2)
            selected_mixing = .40000000000013
            with patch("jax_saturn.training.label_agnostic.species_mixing_fraction",
                       side_effect=[.30000000000009, selected_mixing, .20000000000007]), \
                 patch("jax_saturn.training.label_agnostic.topology_recall_at_50", side_effect=[.8, .9, .6]):
                full, history, summary = train_label_free(artifact, checkpoint, root / "selected",
                    objective="mmd", config=config, devices=self.devices)
            self.assertEqual(full.best_epoch, 1)
            self.assertEqual(full.best_mixing, selected_mixing)
            with patch("jax_saturn.training.label_agnostic.species_mixing_fraction", return_value=.20000000000007), \
                 patch("jax_saturn.training.label_agnostic.topology_recall_at_50", return_value=.6):
                resumed, restored_history, _ = train_label_free(artifact, checkpoint, root / "selected_copy",
                    objective="mmd", config=config, devices=self.devices,
                    resume=root / "selected/checkpoints/epoch_0001")
            self.assertEqual(history, restored_history)
            self.assertEqual(resumed.best_epoch, 1)
            self.assertEqual(resumed.best_mixing, selected_mixing)
            for got, want in zip(jax.tree_util.tree_leaves(full.best_params), jax.tree_util.tree_leaves(resumed.best_params)):
                np.testing.assert_array_equal(got, want)
            self.assertTrue(any(not np.array_equal(a, b) for a, b in zip(
                jax.tree_util.tree_leaves(full.params), jax.tree_util.tree_leaves(full.best_params))))


@unittest.skipUnless(os.environ.get("SATURN_PMAP_TEST_CHILD") == "1", "runs inside the two-device CPU subprocess")
class InferencePmapChecks(unittest.TestCase):
    def test_order_padding_and_changing_metric_weights(self):
        devices = jax.local_devices()
        values = np.random.default_rng(76).uniform(0., 3., (13, 7)).astype(np.float32)
        module = SaturnMetricModule(7, 9, 5)
        params = module.init(jax.random.key(4), values)["params"]
        expected = embed_metric(module, params, values, batch_size=8)
        actual = embed_metric(module, params, values, batch_size=8, devices=devices)
        self.assertEqual(actual.shape, (13, 5))
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
        kernel = _metric_kernel(module, tuple(devices))
        changed = jax.tree_util.tree_map(lambda value: value + .03, params)
        updated = embed_metric(module, changed, values, batch_size=8, devices=devices)
        np.testing.assert_allclose(updated, embed_metric(module, changed, values, batch_size=8), rtol=1e-4, atol=1e-5)
        self.assertFalse(np.array_equal(actual, updated))
        self.assertIs(kernel, _metric_kernel(module, tuple(devices)))
        self.assertEqual(updated.dtype, np.float32)
        with self.assertRaises(ValueError):
            embed_metric(module, params, values, batch_size=7, devices=devices)

    def test_species_outputs_order_and_changing_pretrain_weights(self):
        devices = jax.local_devices()
        data, scores = tiny_prepared_fixture()
        module = SaturnPretrainModule(data.species_names, data.gene_counts, 4, 7, 5)
        params = create_state(module, scores, optax.adam(.0005), 0).params
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = emit_pretrain_anndata(module, params, data, root / "single.h5ad", batch_size=8)
            actual = emit_pretrain_anndata(module, params, data, root / "parallel.h5ad", batch_size=8, devices=devices)
            self.assertEqual(actual.shape, (16, 5))
            np.testing.assert_array_equal(actual.obs_names, data.obs.index)
            np.testing.assert_allclose(actual.X, expected.X, rtol=1e-4, atol=1e-5)
            np.testing.assert_allclose(actual.obsm["macrogenes"], expected.obsm["macrogenes"], rtol=1e-4, atol=1e-5)
            kernel = _pretrain_kernel(module, tuple(devices), 0)
            changed = jax.tree_util.tree_map(lambda value: value + .03, params)
            updated = emit_pretrain_anndata(module, changed, data, root / "updated.h5ad", batch_size=8, devices=devices)
            reference = emit_pretrain_anndata(module, changed, data, root / "updated_single.h5ad", batch_size=8)
            np.testing.assert_allclose(updated.X, reference.X, rtol=1e-4, atol=1e-5)
            np.testing.assert_allclose(updated.obsm["macrogenes"], reference.obsm["macrogenes"], rtol=1e-4, atol=1e-5)
            self.assertFalse(np.array_equal(updated.X, actual.X))
            self.assertIs(kernel, _pretrain_kernel(module, tuple(devices), 0))


class DistributedProcessTests(unittest.TestCase):
    def test_two_device_cpu_process(self):
        code = """
import unittest
from jax_saturn.verification.test_distributed import MetricPmapChecks, PretrainPmapChecks
suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls) for cls in (MetricPmapChecks, PretrainPmapChecks))
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not (result.wasSuccessful() and result.testsRun == 5 and not result.skipped))
"""
        result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[2],
            env={**os.environ, "JAX_PLATFORMS": "cpu", "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
                 "SATURN_PMAP_TEST_CHILD": "1"},
            capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_two_device_inference_process(self):
        code = """
import unittest
from jax_saturn.verification.test_distributed import InferencePmapChecks
result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(InferencePmapChecks))
raise SystemExit(not (result.wasSuccessful() and result.testsRun == 2 and not result.skipped))
"""
        result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[2],
            env={**os.environ, "JAX_PLATFORMS": "cpu", "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
                 "SATURN_PMAP_TEST_CHILD": "1"}, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_two_device_label_free_process(self):
        code = """
import unittest
from jax_saturn.verification.test_distributed import LabelPmapChecks
result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(LabelPmapChecks))
raise SystemExit(not (result.wasSuccessful() and result.testsRun == 2 and not result.skipped))
"""
        result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[2],
            env={**os.environ, "JAX_PLATFORMS": "cpu", "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
                 "SATURN_PMAP_TEST_CHILD": "1"}, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main(defaultTest="DistributedProcessTests")

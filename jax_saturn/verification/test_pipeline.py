"""Exercise the public benchmark shell through the unchanged evaluator."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import anndata as ad
import pandas as pd

from jax_saturn.contracts.validation import load_json, load_npz
from jax_saturn.distributed.checkpoint import latest_checkpoint
from jax_saturn.verification.test_pretrain import write_atlas_fixture


ROOT = Path(__file__).resolve().parents[2]


class PipelineTests(unittest.TestCase):
    def test_benchmark_shell_and_evaluator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_atlas_fixture(root)
            # The reference artifact miner excludes both pair labels; use three
            # cell types so every cross-label pair has a valid negative pool.
            for atlas_path in root.glob("*.h5ad"):
                atlas = ad.read_h5ad(atlas_path)
                names = ("T_one", "T_two", "T_three")
                atlas.obs["cellType"] = pd.Categorical([names[i % 3] for i in range(atlas.n_obs)])
                atlas.write_h5ad(atlas_path)
            output = root / "benchmark"
            environment = {**os.environ, "PYTHON": sys.executable, "IN_DATA": str(manifest),
                           "OUT_DIR": str(output), "DEVICE": "cpu", "JAX_PLATFORMS": "cpu",
                           "HV_GENES": "32", "NUM_MACROGENES": "4", "HIDDEN_DIM": "9",
                           "MODEL_DIM": "5", "PRETRAIN_EPOCHS": "1", "EPOCHS": "1",
                           "PRETRAIN_BATCH_SIZE": "16", "BATCH_SIZE": "16", "RESUME": "0",
                           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                           "KEEP_PYTORCH_COMPAT_CHECKPOINT": "1"}
            result = subprocess.run(["bash", "scripts/run_label_agnostic_benchmark_jax.sh"],
                                    cwd=ROOT, env=environment, timeout=360,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            comparison = pd.read_csv(output / "comparison.csv")
            self.assertEqual(len(comparison), 4)
            self.assertTrue((output / "acceptance.json").is_file())
            json.loads((output / "acceptance.json").read_text())
            for objective in ("infonce", "mmd", "ot"):
                summary = load_json(output / objective / "run_summary.json", "run_summary")
                self.assertFalse(summary["selection_uses_labels"])
                self.assertEqual(load_npz(output / objective / "final_embeddings.npz",
                                         "final_embeddings")["embeddings"].shape, (68, 5))
                self.assertTrue(latest_checkpoint(output / objective).is_dir())
            self.assertTrue((output / "shared/pretrain_orbax.pt").is_file())
            self.assertTrue((output / "baseline/metric_orbax.pt").is_file())


if __name__ == "__main__":
    unittest.main()

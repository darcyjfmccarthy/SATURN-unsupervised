"""Label-free graph parity, all objective integrations, strict inputs and resume."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import anndata as ad
import jax
import numpy as np
import pandas as pd

from jax_saturn.contracts.validation import load_json, load_npz, validate_metric_history
from jax_saturn.data import graphs
from jax_saturn.data.cache import file_sha256
from jax_saturn.distributed.checkpoint import load_pretrain_metric_params
from jax_saturn.models.saturn import SaturnPretrainModule
from jax_saturn.training.baseline import embed_metric, train_baseline
from jax_saturn.training.label_agnostic import LabelConfig, train_label_free
from jax_saturn.training.pretrain import emit_pretrain_anndata, train_pretrain
from jax_saturn.verification.test_pretrain import tiny_prepared_fixture
from label_agnostic import objectives as reference
from label_agnostic.metrics import species_mixing_fraction, topology_recall_at_50
from label_agnostic.artifacts import save_label_free_artifact


ROOT = Path(__file__).resolve().parents[2]


class LabelTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.root = Path(cls.directory.name)
        data, scores = tiny_prepared_fixture()
        cls.data = data
        cls.pretrain_module = SaturnPretrainModule(data.species_names, data.gene_counts, 4, 7, 5)
        cls.pretrain, _ = train_pretrain(cls.pretrain_module, data, scores,
                                        output_dir=cls.root / "pretrain", epochs=1, batch_size=8)
        cls.atlas = emit_pretrain_anndata(cls.pretrain_module, cls.pretrain.params, data,
                                          cls.root / "truth.h5ad", batch_size=8)
        cls.artifact = cls.root / "artifact.npz"
        save_label_free_artifact(cls.artifact, cls.atlas.X, cls.atlas.obsm["macrogenes"],
                                 cls.atlas.obs["species"], cls.atlas.obs_names)
        cls.checkpoint = cls.root / "pretrain/checkpoints/epoch_0001"
        cls.config = LabelConfig(epochs=1, batch_size=8, hidden_dim=7, model_dim=5, ot_iterations=15)

    def test_cpu_graphs_and_selection_metrics_match_reference(self):
        embeddings = np.asarray(self.atlas.X)
        macros = np.asarray(self.atlas.obsm["macrogenes"])
        species = np.asarray(self.atlas.obs["species"]).astype(str)
        actual = graphs.build_preservation_graph(embeddings, species, seed=3)
        expected = reference.build_preservation_graph(embeddings, species, seed=3)
        for got, want in zip(actual, expected):
            np.testing.assert_array_equal(got, want)
        actual_positives = graphs.build_cross_species_positives(embeddings, macros, species)
        expected_positives = reference.build_cross_species_positives(embeddings, macros, species)
        for got, want in zip(actual_positives, expected_positives):
            np.testing.assert_array_equal(got, want)
        self.assertEqual(graphs.species_mixing_fraction(embeddings, species), species_mixing_fraction(embeddings, species))
        self.assertEqual(graphs.topology_recall_at_50(embeddings, species, actual[3]), topology_recall_at_50(embeddings, species, actual[3]))

    def test_all_objectives_write_finite_strict_outputs_and_reject_labels(self):
        for objective in ("infonce", "mmd", "ot"):
            with self.subTest(objective=objective):
                state, history, summary = train_label_free(self.artifact, self.checkpoint, self.root / objective,
                                                           objective=objective, config=self.config)
                self.assertEqual(summary["artifact_keys_seen_by_trainer"], ["embeddings", "macrogenes", "obs_ids", "species"])
                self.assertIs(summary["selection_uses_labels"], False)
                self.assertEqual(summary["implementation"], "jax")
                self.assertEqual(summary, load_json(self.root / objective / "run_summary.json", "run_summary"))
                validate_metric_history(pd.DataFrame(history), epochs=1)
                values = load_npz(self.root / objective / "final_embeddings.npz", "final_embeddings")
                np.testing.assert_array_equal(values["obs_ids"], self.atlas.obs_names)
                self.assertEqual(values["embeddings"].shape, (16, 5))
                self.assertTrue(np.isfinite(values["embeddings"]).all())
        bad = self.root / "label_leak.npz"
        artifact = load_npz(self.artifact, "label_free")
        np.savez(bad, **artifact, labels=np.arange(16))
        with self.assertRaises(ValueError):
            train_label_free(bad, self.checkpoint, self.root / "bad", objective="mmd", config=self.config)

    def test_resume_preserves_best_state_and_adaptive_weight(self):
        config = replace(self.config, epochs=2)
        full, history, summary = train_label_free(self.artifact, self.checkpoint, self.root / "resume_full",
                                                  objective="mmd", config=config)
        checkpoint = self.root / "resume_full/checkpoints/epoch_0001"
        resumed, resumed_history, resumed_summary = train_label_free(self.artifact, self.checkpoint, self.root / "resume_copy",
                                                                     objective="mmd", config=config, resume=checkpoint)
        self.assertEqual(history, resumed_history)
        self.assertEqual(full.preservation_weight, resumed.preservation_weight)
        self.assertEqual(full.best_epoch, resumed.best_epoch)
        self.assertEqual(full.best_mixing, resumed.best_mixing)
        self.assertEqual(full.best_recall, resumed.best_recall)
        self.assertEqual(full.diagnostics, resumed.diagnostics)
        for a, b in zip(jax.tree_util.tree_leaves(full.params), jax.tree_util.tree_leaves(resumed.params)):
            np.testing.assert_array_equal(a, b)
        for a, b in zip(jax.tree_util.tree_leaves(full.best_params), jax.tree_util.tree_leaves(resumed.best_params)):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(summary["selected_epoch"], resumed_summary["selected_epoch"])
        with self.assertRaises(ValueError):
            train_label_free(self.artifact, self.checkpoint, self.root / "wrong_objective",
                             objective="ot", config=config, resume=checkpoint)

    def test_epoch_zero_selection_is_preserved_exactly(self):
        config = replace(self.config, preservation_target=1.)
        state, _, summary = train_label_free(self.artifact, self.checkpoint, self.root / "initial_selected",
                                             objective="mmd", config=config)
        # On 16 cells, k=15 mixing is invariant under changes to embedding geometry.
        self.assertEqual(summary["selected_epoch"], 0)
        initial_params, _ = load_pretrain_metric_params(self.checkpoint, input_dim=4, hidden_dim=7, model_dim=5)
        for got, want in zip(jax.tree_util.tree_leaves(state.best_params), jax.tree_util.tree_leaves(initial_params)):
            np.testing.assert_array_equal(got, want)

    def test_selected_epoch_and_params_survive_resume(self):
        config = replace(self.config, epochs=2)
        with patch("jax_saturn.training.label_agnostic.species_mixing_fraction", side_effect=[.3, .4, .2]), \
             patch("jax_saturn.training.label_agnostic.topology_recall_at_50", side_effect=[.8, .9, .6]):
            full, history, summary = train_label_free(self.artifact, self.checkpoint, self.root / "selected_full",
                                                      objective="mmd", config=config)
        self.assertEqual(summary["selected_epoch"], 1)
        with patch("jax_saturn.training.label_agnostic.species_mixing_fraction", return_value=.2), \
             patch("jax_saturn.training.label_agnostic.topology_recall_at_50", return_value=.6):
            resumed, resumed_history, _ = train_label_free(self.artifact, self.checkpoint, self.root / "selected_resume",
                objective="mmd", config=config, resume=self.root / "selected_full/checkpoints/epoch_0001")
        self.assertEqual(history, resumed_history)
        self.assertEqual(resumed.best_epoch, 1)
        for got, want in zip(jax.tree_util.tree_leaves(resumed.best_params), jax.tree_util.tree_leaves(full.best_params)):
            np.testing.assert_array_equal(got, want)
        self.assertTrue(any(not np.array_equal(a, b) for a, b in zip(jax.tree_util.tree_leaves(full.params), jax.tree_util.tree_leaves(full.best_params))))
        # Re-exporting a completed epoch validates the immutable selected checkpoint.
        train_label_free(self.artifact, self.checkpoint, self.root / "selected_full", objective="mmd", config=config,
                         resume=self.root / "selected_full/checkpoints/epoch_0002")
        with self.assertRaises(ValueError):
            train_label_free(self.artifact, self.checkpoint, self.root / "selected_rejected", objective="mmd", config=config,
                             resume=summary["orbax_checkpoint_path"])

    def test_label_free_cli_imports_without_torch_and_runs(self):
        script = "import sys; import jax_saturn.training.label_agnostic; assert 'torch' not in sys.modules"
        subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True, timeout=30, capture_output=True, text=True)
        result = subprocess.run([sys.executable, str(ROOT / "scripts/train_label_agnostic_jax.py"),
                                 "--objective", "mmd", "--artifact", str(self.artifact),
                                 "--pretrain-checkpoint", str(self.checkpoint), "--output-dir", str(self.root / "cli"),
                                 "--device", "cpu", "--epochs", "1", "--batch-size", "8",
                                 "--hidden-dim", "7", "--model-dim", "5"], cwd=ROOT,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        load_json(self.root / "cli/run_summary.json", "run_summary")


if __name__ == "__main__":
    unittest.main()

"""Phase 1 fixtures and contract tests; run with unittest discovery."""

import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from jax_saturn.contracts.validation import (
    load_json, load_npz, validate_anndata, validate_checkpoint_metadata,
    validate_metric_history, validate_npz, validate_run_summary,
)
from jax_saturn.data.cache import (
    convert_gene_embeddings, file_sha256, import_centroid_pickle, load_cache,
    metadata_path, save_cache, validate_centroid_cache,
)
from jax_saturn.data.manifest import load_manifest
from label_agnostic.artifacts import load_label_free_artifact, save_label_free_artifact


ROOT = Path(__file__).resolve().parents[2]
DIGEST = "a" * 64


def label_free_fixture():
    return {
        "embeddings": np.arange(12, dtype=np.float32).reshape(4, 3),
        "macrogenes": np.arange(8, dtype=np.float32).reshape(4, 2),
        "species": np.asarray(["a_species", "a_species", "z_species", "z_species"]),
        "obs_ids": np.asarray(["a0", "a1", "z0", "z1"]),
    }


def legacy_centroid_fixture():
    return {
        "scores": {"a_species_G2": np.array([0.2, 0.8]),
                   "a_species_G1": np.array([0.7, 0.3]),
                   "z_species_G3": np.array([0.4, 0.6])},
        "centroids": np.arange(6, dtype=np.float32).reshape(2, 3),
        "score_func": "default", "sorted_species_names": ["a_species", "z_species"],
        "species_to_gene_idx_hv": {"a_species": (0, 2), "z_species": (2, 3)},
        "all_gene_names": ["a_species_G1", "a_species_G2", "z_species_G3"],
    }


class TemporaryTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)


class ManifestTests(TemporaryTest):
    def write_manifest(self, text):
        path = self.root / "manifest.csv"
        path.write_text(text)
        return path

    def test_repository_manifests(self):
        rows = load_manifest(ROOT / "data/human_monkey_mouse.csv")
        self.assertEqual([r.species for r in rows], ["h_sapiens", "m_murinus", "m_musculus"])
        self.assertEqual([r.source_row_index for r in rows], [0, 2, 1])
        self.assertEqual(rows[0].path, ROOT / "data/cell_atlases/h_sapiens.h5ad")
        large = load_manifest(ROOT / "data/datatable.csv")
        self.assertEqual(len(large), 25)
        self.assertTrue(all(r.in_label_col == "cellType" for r in large))
        self.assertEqual(len(load_manifest(ROOT / "data/saturn_run_tiny.csv")), 4)

    def test_external_root_and_file_validation(self):
        path = self.write_manifest("species,atlas,gene_embeddings\nz,atlas.h5ad,genes.pt\n")
        with self.assertRaises(FileNotFoundError):
            load_manifest(path, base_dir=self.root, check_paths=True)
        (self.root / "atlas.h5ad").touch()
        (self.root / "genes.pt").touch()
        self.assertEqual(load_manifest(path, base_dir=self.root, check_paths=True)[0].path,
                         self.root / "atlas.h5ad")

    def test_invalid_manifests(self):
        cases = [
            "species,path\na,a.h5ad\n",
            "species,path,embedding_path\n",
            "species,path,embedding_path\na,a,e\na,b,f\n",
            "species,path,embedding_path\na,,e\n",
            "species,path,atlas,embedding_path\na,a,b,e\n",
            "species,path,path,embedding_path\na,a,a,e\n",
            "species,path,embedding_path\na,a,e,extra\n",
        ]
        for text in cases:
            with self.subTest(text=text), self.assertRaises(ValueError):
                load_manifest(self.write_manifest(text))


class ArtifactTests(TemporaryTest):
    def test_reference_label_free_round_trip(self):
        fixture = label_free_fixture()
        path = self.root / "artifact.npz"
        save_label_free_artifact(path, **fixture)
        actual = load_npz(path, "label_free")
        reference = load_label_free_artifact(path)
        for key in fixture:
            np.testing.assert_array_equal(actual[key], reference[key])
            self.assertEqual(actual[key].dtype, fixture[key].dtype)

    def test_reject_labels_dimensions_dtypes_and_identity(self):
        for key, value in [
            ("labels", np.arange(4)), ("embeddings", np.ones((4, 3), dtype=np.float64)),
            ("embeddings", np.full((4, 3), np.nan, dtype=np.float32)),
            ("species", np.asarray([["a"]] * 4)),
            ("obs_ids", np.asarray(["a", "a", "b", "c"])),
            ("macrogenes", np.ones((3, 2), dtype=np.float32)),
            ("species", np.asarray(["a"] * 4, dtype=object)),
        ]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_npz({**label_free_fixture(), key: value}, "label_free")

    def test_final_embeddings_and_triplets(self):
        fixture = label_free_fixture()
        del fixture["macrogenes"]
        validate_npz(fixture, "final_embeddings")
        triplets = {"anchor": np.array([0, 1], dtype=np.int64),
                    "positive": np.array([1, 0], dtype=np.int64),
                    "negative": np.array([2, 3], dtype=np.int64), "obs_ids": fixture["obs_ids"]}
        validate_npz(triplets, "evaluation_triplets")
        for invalid in (np.array([4, 3], dtype=np.int64), np.array([-1, 2], dtype=np.int64),
                        np.array([2], dtype=np.int64), np.array([2, 3], dtype=np.int32)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_npz({**triplets, "negative": invalid}, "evaluation_triplets")

    def test_anndata_disk_round_trip_and_order(self):
        import anndata as ad
        from scipy import sparse

        fixture = label_free_fixture()
        obs = pd.DataFrame(index=fixture["obs_ids"])
        for key in ("labels", "labels2", "ref_labels"):
            obs[key] = pd.Categorical(["one", "two", "one", "two"])
        obs["species"] = pd.Categorical(fixture["species"])
        adata = ad.AnnData(X=fixture["embeddings"], obs=obs,
                          obsm={"macrogenes": fixture["macrogenes"],
                                "X_umap": np.zeros((4, 2), dtype=np.float32)})
        path = self.root / "tiny.h5ad"
        adata.write_h5ad(path)
        restored = ad.read_h5ad(path)
        validate_anndata(restored, evaluated=True, expected_obs_ids=fixture["obs_ids"],
                         expected_species=fixture["species"])
        restored.X = sparse.csr_matrix(restored.X)
        validate_anndata(restored)
        with self.assertRaises(ValueError):
            validate_anndata(restored, expected_obs_ids=fixture["obs_ids"][::-1])
        restored.obs["species"] = restored.obs["species"].astype(str)
        with self.assertRaises(ValueError):
            validate_anndata(restored)

    def test_csv_history_round_trip(self):
        history = pd.DataFrame({"epoch": [1, 2], "metric_loss": [0.3, 0.2], "diagnostic": [5, 6]})
        path = self.root / "history.csv"
        history.to_csv(path, index=False)
        validate_metric_history(pd.read_csv(path), epochs=2)
        for epochs, losses in [([0, 1], [0.3, 0.2]), ([1, 3], [0.3, 0.2]), ([1, 2], [0.3, np.inf])]:
            with self.subTest(epochs=epochs), self.assertRaises(ValueError):
                validate_metric_history(pd.DataFrame({"epoch": epochs, "metric_loss": losses}), epochs=2)

    def test_checkpoint_metadata(self):
        metadata = {"schema_version": 1, "implementation": "jax", "model_kind": "pretrain",
                    "epoch": 0, "step": 1, "seed": 0, "hyperparameters": {"model_dim": 3},
                    "species_names": ["a_species", "z_species"], "input_shapes": {"a_species": [2, 3]},
                    "source_manifest_sha256": DIGEST}
        path = self.root / "metadata.json"
        path.write_text(json.dumps(metadata))
        self.assertEqual(load_json(path, "checkpoint"), metadata)
        for key, value in [("labels", []), ("step", True), ("schema_version", 2),
                           ("species_names", ["z", "a"]), ("source_manifest_sha256", "bad")]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_checkpoint_metadata({**metadata, key: value})
        path.write_text('{"schema_version": 1, "schema_version": 1}')
        with self.assertRaises(ValueError):
            load_json(path, "checkpoint")

    def test_run_summary(self):
        summary = {"objective": "infonce", "label_free": True,
                   "artifact_keys_seen_by_trainer": sorted(label_free_fixture()),
                   "selected_epoch": 1, "selected_species_mixing_fraction": 0.4,
                   "selected_teacher_top15_recall_at_50": 0.8,
                   "selection_uses_labels": False, "epochs": 2, "seed": 0}
        validate_run_summary(summary)
        reference_summary = {**summary, "selected_epoch": 0, "schema_version": 1,
                             "trainer_version": 4, "configuration": {}, "selection_rule": "mixing"}
        path = self.root / "run_summary.json"
        path.write_text(json.dumps(reference_summary))
        self.assertEqual(load_json(path, "run_summary"), reference_summary)
        for key, value in [("selection_uses_labels", True), ("selected_epoch", 3),
                           ("selected_species_mixing_fraction", np.nan), ("extra", 0)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_run_summary({**summary, key: value})


class CacheTests(TemporaryTest):
    def test_embedding_conversion_and_determinism(self):
        import torch

        source = self.root / "genes.pt"
        torch.save({"Z": torch.tensor([1., 2.]), "GeneA": torch.tensor([3., 4.]),
                    "genea": torch.tensor([5., 6.])}, source)
        first, second = self.root / "first.npz", self.root / "second.npz"
        convert_gene_embeddings(source, first, species="a_species")
        convert_gene_embeddings(source, second, species="a_species")
        self.assertEqual(file_sha256(first), file_sha256(second))
        self.assertEqual(metadata_path(first).read_bytes(), metadata_path(second).read_bytes())
        values, metadata = load_cache(first, kind="embedding")
        self.assertEqual(list(values["gene_symbols"]), ["genea", "z"])
        np.testing.assert_array_equal(values["embeddings"], [[5., 6.], [1., 2.]])
        self.assertEqual(values["embeddings"].dtype, np.float32)
        self.assertEqual(metadata["source_sha256"], file_sha256(source))
        metadata["species"] = ""
        with self.assertRaises(ValueError):
            save_cache(second, values, metadata, kind="embedding")

    def test_malformed_embedding_source(self):
        import torch

        for raw in ({}, {"a": torch.ones(2), "b": torch.ones(3)},
                    {"a": torch.tensor([np.nan])}, {"a": torch.ones(1, 2)}):
            source = self.root / "bad.pt"
            torch.save(raw, source)
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                convert_gene_embeddings(source, self.root / "bad.npz", species="a")

    def test_centroid_import_and_export(self):
        source = self.root / "centroids.pkl"
        with source.open("wb") as handle:
            pickle.dump(legacy_centroid_fixture(), handle)
        path = self.root / "centroids.npz"
        import_centroid_pickle(source, path, seed=0, hv_genes=2000,
                               embedding_cache_sha256=DIGEST, source_manifest_sha256=DIGEST)
        values, metadata = load_cache(path, kind="centroid")
        np.testing.assert_array_equal(values["scores"], np.array([[0.7, 0.3], [0.2, 0.8], [0.4, 0.6]], dtype=np.float32))
        np.testing.assert_array_equal(values["species_gene_starts"], [0, 2])
        self.assertEqual(values["species_gene_starts"].dtype, np.int64)
        exported = self.root / "exported.npz"
        save_cache(exported, values, metadata, kind="centroid")
        self.assertEqual(file_sha256(path), file_sha256(exported))
        for key, value in [("species_gene_starts", np.array([0, 1], dtype=np.int64)),
                           ("all_gene_names", np.array(["a_species_G1", "z_species_G2", "z_species_G3"])),
                           ("scores", np.zeros((3, 3), dtype=np.float32))]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_centroid_cache({**values, key: value}, metadata)


if __name__ == "__main__":
    unittest.main()

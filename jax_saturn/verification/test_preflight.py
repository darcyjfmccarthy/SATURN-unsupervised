"""Pilot preflight must report incomplete inputs without dropping species."""

import csv
from pathlib import Path
import tempfile
import unittest

import h5py

from scripts.preflight_saturn_jax import inspect_manifest


class PreflightChecks(unittest.TestCase):
    def test_dense_sparse_metadata_and_all_missing_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with h5py.File(root / "dense.h5ad", "w") as handle:
                handle.create_dataset("X", shape=(3, 7), dtype="float32")
            with h5py.File(root / "sparse.h5ad", "w") as handle:
                handle.create_group("X").attrs["shape"] = (5, 11)
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["species", "atlas", "gene_embeddings"])
                writer.writerows([
                    ["z", "sparse.h5ad", "z.pt"],
                    ["a", "dense.h5ad", "a.pt"],
                    ["m", "absent.h5ad", "m.pt"],
                ])
            report = inspect_manifest(manifest, base_dir=root, hv_genes=4,
                                      embed_dim=2, num_macrogenes=3,
                                      expected_species=25)
            self.assertFalse(report["ready_for_input_loading"])
            self.assertEqual(report["species_count"], 3)
            self.assertEqual(report["inspected_atlas_count"], 2)
            self.assertEqual(report["total_cells_in_inspected_atlases"], 8)
            self.assertEqual([entry["species"] for entry in report["species"]], ["a", "m", "z"])
            self.assertEqual(len(report["issues"]), 5)
            self.assertEqual(report["host_array_bytes"], {
                "dense_hv_expression_float32": 128,
                "one_embedding_bank_float32": 64,
                "one_macrogene_export_float32": 96,
            })

    def test_bad_atlas_is_reported_and_dimensions_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bad.h5ad").write_text("not HDF5")
            (root / "a.pt").touch()
            manifest = root / "manifest.csv"
            manifest.write_text("species,path,embedding_path\na,bad.h5ad,a.pt\n")
            report = inspect_manifest(manifest, base_dir=root)
            self.assertFalse(report["ready_for_input_loading"])
            self.assertEqual(len(report["issues"]), 1)
            self.assertIn("cannot inspect atlas", report["issues"][0])
            self.assertEqual(report["inspected_atlas_count"], 0)
            with self.assertRaisesRegex(ValueError, "positive"):
                inspect_manifest(manifest, base_dir=root, hv_genes=0)


if __name__ == "__main__":
    unittest.main()

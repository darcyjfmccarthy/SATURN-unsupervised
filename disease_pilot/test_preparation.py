"""Critical count provenance, gene mapping, identity and split leakage checks."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from disease_pilot.preparation import (
    assert_counts, count_source, donor_folds, download_dataset, map_counts,
    materialize_experiment, normalized_expression, prepare_external, resolve_dataset, SPECIES,
)


class PreparationTests(unittest.TestCase):
    def test_legacy_expression_and_new_counts_get_the_same_per_cell_scale(self):
        x = sparse.csr_matrix([[.2, .8], [20, 80]], dtype=np.float32)
        values = normalized_expression(x)
        np.testing.assert_allclose(values.toarray(), [[2000, 8000], [2000, 8000]])
        with self.assertRaises(ValueError):
            normalized_expression(sparse.csr_matrix([[0, 0]], dtype=np.float32))

    def test_raw_feature_table_is_independent_of_normalized_features(self):
        a = ad.AnnData(sparse.csr_matrix([[1, 2, 3]], dtype=np.float32),
                       var=pd.DataFrame({"feature_name": ["A", "B", "C"]}, index=["ens1", "ens2", "ens3"]))
        a.raw = a.copy()
        a = a[:, [0]].copy()
        a.X = sparse.csr_matrix([[.5]], dtype=np.float32)
        x, var, slot = count_source(a)
        self.assertEqual(slot, "raw.X")
        self.assertEqual(x.shape[1], len(var))
        np.testing.assert_array_equal(var.feature_name, ["A", "B", "C"])
        assert_counts(x)
        with self.assertRaises(ValueError):
            assert_counts(a.X)

    def test_duplicates_sum_and_unmeasured_genes_are_not_zero_filled(self):
        x = sparse.csr_matrix([[2, 3, 7, 11], [1, 4, 9, 8]], dtype=np.float32)
        var = pd.DataFrame({"feature_name": ["A", "a", "B", "other"]}, index=["e1", "e2", "e3", "e4"])
        counts, genes, audit = map_counts(x, var, ["B", "A", "unmeasured"])
        self.assertEqual(list(genes), ["B", "A"])
        np.testing.assert_array_equal(counts.toarray(), [[7, 5], [9, 5]])
        self.assertEqual(audit.at[3, "atlas_symbol"], "")
        with self.assertRaises(ValueError):
            map_counts(x, var, ["A", "a"])

    def test_group_folds_never_split_donors_and_test_each_once(self):
        obs = pd.DataFrame({"donor_group": np.repeat(["WT1", "WT2", "KO1", "KO2", "KO3"], 3),
                            "condition": np.repeat(["healthy", "healthy", "disease", "disease", "disease"], 3)},
                           index=[f"cell{i}" for i in range(15)])
        folds = donor_folds(obs)
        self.assertEqual(folds.loc[folds.split == "test"].obs_id.value_counts().to_dict(), {k: 1 for k in obs.index})
        for _, fold in folds.groupby("fold"):
            self.assertTrue(fold.groupby("donor_group")["split"].nunique().eq(1).all())
            for _, part in fold.groupby("split"):
                self.assertEqual(set(part.condition), {"healthy", "disease"})
        changed = obs.copy()
        changed.loc["cell0", "condition"] = "disease"
        with self.assertRaises(ValueError):
            donor_folds(changed)

    def test_version_resolution_rejects_revision(self):
        meta = {"dataset_id": "d", "dataset_version_id": "v", "collection_id": "c", "cell_count": 5,
                "assets": [{"filetype": "H5AD", "filesize": 2, "url": "https://example.org/v.h5ad"}]}
        with patch("disease_pilot.preparation.api_json", return_value={**meta, "dataset_version_id": "changed"}):
            with self.assertRaises(ValueError):
                resolve_dataset({"datasets": {"pilot": meta}}, "pilot")

    def test_cached_download_detects_content_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "v.h5ad"
            ad.AnnData(sparse.csr_matrix([[1, 2]])).write_h5ad(p)
            metadata = {"dataset_id": "d", "dataset_version_id": "v", "assets": [
                {"filetype": "H5AD", "filesize": p.stat().st_size, "url": "https://example.org/v.h5ad"}]}
            download_dataset(metadata, tmp)
            receipt = p.with_suffix(".download.json")
            record = json.loads(receipt.read_text()); record["sha256"] = "0" * 64
            receipt.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                download_dataset(metadata, tmp)

    def test_realistic_preparation_and_five_species_fold_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            obs = pd.DataFrame({"donor_id": ["WT1", "WT2", "KO1", "KO2"],
                "disease_ontology_term_id": ["PATO:0000461"] * 2 + ["MONDO:0006541"] * 2,
                "cell_type": ["keratinocyte"] * 4, "assay": ["10x 3' v3"] * 4,
                "suspension_type": ["cell"] * 4, "is_primary_data": [True] * 4,
                "batch": ["WT1", "WT2", "KO1", "KO2"],
                "author_covariate": [11, 12, 13, 14]}, index=["c1", "c2", "c3", "c4"])
            a = ad.AnnData(sparse.csr_matrix([[2, 3, 1]] * 4, dtype=np.float32), obs=obs,
                var=pd.DataFrame({"feature_name": ["A", "B", "mt-Nd1"]}, index=["ens1", "ens2", "ens3"]))
            a.raw = a.copy()
            a = a[:, [0]].copy()  # normalized features differ from raw features
            a.X = a.X.astype(float) * .5
            a.write_h5ad(root / "download.h5ad")
            metadata = {"dataset_id": "d", "dataset_version_id": "v", "collection_id": "c", "cell_count": 4,
                        "disease": [{"ontology_term_id": "PATO:0000461"}, {"ontology_term_id": "MONDO:0006541"}]}
            external, qc, mapping = prepare_external(root / "download.h5ad", metadata, ["A", "B", "absent"],
                organ="skin", min_genes=1, max_mito_fraction=.5)
            self.assertEqual(external.shape, (4, 2))
            self.assertEqual(external.uns["disease_preparation"]["counts_from"], "raw.X")
            np.testing.assert_array_equal(external.X.toarray(), [[2, 3]] * 4)
            rows = []
            for species in sorted(SPECIES):
                atlas = ad.AnnData(sparse.csr_matrix([[.5, 1, 1.5]], dtype=np.float32),
                    obs=pd.DataFrame({"cellType": ["keratinocyte"], "organ": ["skin"]}, index=[species + "-a"]),
                    var=pd.DataFrame(index=["A", "B", "absent"]))
                atlas.write_h5ad(root / (species + ".h5ad"))
                (root / (species + ".pt")).write_bytes(b"hash-only-fixture")
                rows.append({"species": species, "path": species + ".h5ad", "embedding_path": species + ".pt", "in_label_col": "cellType"})
            pd.DataFrame(rows).to_csv(root / "source.csv", index=False)
            destination = root / "prepared"
            combined_obs, folds = materialize_experiment(root / "source.csv", external, destination, root=root)
            manifest = pd.read_csv(destination / "fold_0/train.csv")
            self.assertEqual(set(manifest.species), SPECIES)
            self.assertEqual(len(manifest), 5)
            train_path = manifest.loc[manifest.species == "m_musculus", "path"].iloc[0]
            train, test = ad.read_h5ad(train_path), ad.read_h5ad(destination / "fold_0/m_musculus_test.h5ad")
            self.assertFalse(set(train.obs_names) & set(test.obs_names))
            self.assertFalse(set(train.obs.loc[train.obs.source == "external_study", "donor_group"]) & set(test.obs.donor_group))
            self.assertEqual(set(train.obs.species) | set(test.obs.species), {"m_musculus"})
            self.assertEqual(train.n_obs + test.n_obs, len(combined_obs))
            self.assertEqual(set(test.obs.author_covariate),
                             set(external.obs.loc[test.obs_names, 'author_covariate']))
            self.assertTrue(train.obs.loc[train.obs.source == 'atlas', 'author_covariate'].isna().all())
            np.testing.assert_allclose(np.asarray(train.X.sum(axis=1)).ravel(), 10000, rtol=1e-5)
            np.testing.assert_allclose(np.asarray(test.X.sum(axis=1)).ravel(), 10000, rtol=1e-5)
            raw = ad.read_h5ad(destination / 'external_prepared.h5ad')
            np.testing.assert_array_equal(raw.X.toarray(), [[2, 3]] * 4)
            for row in manifest.itertuples():
                self.assertNotEqual(Path(row.path), root / (row.species + '.h5ad'))
            with self.assertRaises(ValueError):
                materialize_experiment(root / "source.csv", external, destination, root=root, seed=99)


if __name__ == "__main__":
    unittest.main()

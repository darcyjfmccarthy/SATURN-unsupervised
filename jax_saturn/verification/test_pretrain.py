"""CPU preprocessing, pretraining, artifact compatibility and process resume."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import anndata as ad
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import scanpy as sc
from scipy import sparse
from flax.traverse_util import flatten_dict

from jax_saturn.contracts.validation import load_npz, validate_anndata
from jax_saturn.data.batching import equal_species_batches
from jax_saturn.data.cache import save_cache
from jax_saturn.data.preparation import PreparedData, prepare_centroids, prepare_data
from jax_saturn.distributed.checkpoint import restore_checkpoint
from jax_saturn.models.saturn import SaturnPretrainModule
from jax_saturn.training.pretrain import create_state, emit_pretrain_anndata, make_pretrain_step, train_pretrain
from label_agnostic.artifacts import save_label_free_artifact
from model.saturn_model import make_centroids


ROOT = Path(__file__).resolve().parents[2]


def write_atlas_fixture(directory):
    rng = np.random.default_rng(29)
    rows = []
    for species, size in (("a_species", 36), ("z_species", 32)):
        counts = rng.poisson(rng.uniform(.5, 12, (1, 80)), (size, 80)).astype(np.float32)
        obs = pd.DataFrame({"cellType": pd.Categorical(["T_one", "T_two"] * (size // 2))},
                           index=[f"{species}_cell_{index}" for index in range(size)])
        atlas = ad.AnnData(sparse.csr_matrix(counts), obs=obs,
                          var=pd.DataFrame(index=[f"G{index}" for index in range(80)]))
        atlas_path = directory / f"{species}.h5ad"
        atlas.write_h5ad(atlas_path)
        cache_path = directory / f"{species}.npz"
        save_cache(cache_path, {"gene_symbols": np.array([f"g{index}" for index in range(1, 80)]),
                               "embeddings": rng.normal(size=(79, 8)).astype(np.float32)},
                   {"schema_version": 1, "species": species, "source_path": "synthetic-fixture",
                    "source_sha256": "a" * 64, "created_by": "test_pretrain", "gene_symbol_case": "lower"},
                   kind="embedding")
        rows.append({"species": species, "path": str(atlas_path), "embedding_path": str(cache_path),
                     "in_label_col": "cellType"})
    manifest = directory / "manifest.csv"
    pd.DataFrame(rows[::-1]).to_csv(manifest, index=False)
    return manifest


def tiny_prepared_fixture():
    rng = np.random.default_rng(19)
    values = tuple(rng.poisson(3, (n, g)).astype(np.float32) for n, g in ((10, 4), (6, 3)))
    obs = pd.DataFrame(index=[f"cell_{index}" for index in range(16)])
    for key in ("labels", "labels2", "ref_labels"):
        obs[key] = pd.Categorical(["T1", "T2"] * 8)
    obs["species"] = pd.Categorical(["a"] * 10 + ["b"] * 6)
    data = PreparedData(("a", "b"), values, rng.normal(size=(7, 5)).astype(np.float32),
                        (("a_g1", "a_g2", "a_g3", "a_g4"), ("b_g1", "b_g2", "b_g3")),
                        obs, "a" * 64, "b" * 64)
    return data, rng.uniform(.1, 1, (7, 4)).astype(np.float32)


class PretrainTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def run_process(self, command, *, timeout):
        result = subprocess.run(command, cwd=ROOT, timeout=timeout, capture_output=True, text=True,
                                env={**os.environ, "JAX_PLATFORMS": "cpu"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_sparse_preprocessing_and_centroid_reference_parity(self):
        manifest = write_atlas_fixture(self.root)
        data = prepare_data(manifest, cache_dir=self.root / "cache", hv_genes=32)
        self.assertEqual(data.species_names, ("a_species", "z_species"))
        self.assertEqual(data.gene_counts, (32, 32))
        self.assertEqual(data.values[0].dtype, np.float32)
        self.assertEqual(data.obs["labels2"].iloc[0], "one")
        for code, species in enumerate(data.species_names):
            atlas = ad.read_h5ad(self.root / f"{species}.h5ad")[:, 1:].copy()
            raw = atlas.X.copy()
            sc.pp.normalize_total(atlas, target_sum=1e4)
            sc.pp.highly_variable_genes(atlas, flavor="seurat_v3", n_top_genes=32)
            hv = atlas.var["highly_variable"].to_numpy()
            np.testing.assert_array_equal(data.values[code], raw[:, hv].toarray())
            self.assertEqual(data.gene_names[code], tuple(f"{species}_{gene}" for gene in atlas.var_names[hv]))
        path = self.root / "centroids.npz"
        cache = prepare_centroids(data, path, seed=0, hv_genes=32, num_macrogenes=4)
        genes = [name for block in data.gene_names for name in block]
        expected_scores, expected_centroids = make_centroids(data.gene_embeddings, genes, 4, seed=0, device="cpu")
        np.testing.assert_allclose(cache["scores"], np.stack([expected_scores[gene] for gene in genes]), rtol=1e-6)
        np.testing.assert_allclose(cache["centroids"], expected_centroids, rtol=1e-6)
        again = prepare_centroids(data, path, seed=0, hv_genes=32, num_macrogenes=4)
        np.testing.assert_array_equal(again["scores"], cache["scores"])
        with self.assertRaises(ValueError):
            prepare_centroids(data, path, seed=1, hv_genes=32, num_macrogenes=4)

    def test_equal_sampling_padding_order_and_epoch_determinism(self):
        data, _ = tiny_prepared_fixture()
        batches = list(equal_species_batches(data.values, 7, seed=0, epoch=1))
        again = list(equal_species_batches(data.values, 7, seed=0, epoch=1))
        self.assertEqual(sum(record["valid_mask"].sum() for batch in batches for record in batch), 20)
        for batch, repeated in zip(batches, again):
            for code, (record, other) in enumerate(zip(batch, repeated)):
                for key in record:
                    np.testing.assert_array_equal(record[key], other[key])
                self.assertEqual(record["values"].shape, (7, data.gene_counts[code]))
                indices = record["global_index"][record["valid_mask"]] - (0 if code == 0 else 10)
                np.testing.assert_array_equal(record["values"][record["valid_mask"]], data.values[code][indices])
                np.testing.assert_array_equal(record["values"][~record["valid_mask"]], 0)
        for code in range(2):
            self.assertEqual(sum(batch[code]["valid_mask"].sum() for batch in batches), 10)

    def test_epoch_resume_matches_uninterrupted_training(self):
        data, scores = tiny_prepared_fixture()
        module = SaturnPretrainModule(data.species_names, data.gene_counts, 4, 7, 5)
        uninterrupted, expected_history = train_pretrain(module, data, scores, output_dir=self.root / "full",
                                                         epochs=2, batch_size=8)
        checkpoint = self.root / "full/checkpoints/epoch_0001"
        resumed, history = train_pretrain(module, data, scores, output_dir=self.root / "resumed",
                                          epochs=2, batch_size=8, resume=checkpoint)
        self.assertEqual(history, expected_history)
        for got, expected in zip(jax.tree_util.tree_leaves(resumed), jax.tree_util.tree_leaves(uninterrupted)):
            if jax.dtypes.issubdtype(got.dtype, jax.dtypes.prng_key):
                got, expected = jax.random.key_data(got), jax.random.key_data(expected)
            np.testing.assert_array_equal(got, expected)
        adata = emit_pretrain_anndata(module, resumed.params, data, self.root / "pretrain.h5ad", batch_size=8)
        validate_anndata(ad.read_h5ad(self.root / "pretrain.h5ad"), expected_obs_ids=data.obs.index)
        self.assertEqual(adata.shape, (16, 5))
        artifact = self.root / "artifact.npz"
        save_label_free_artifact(artifact, adata.X, adata.obsm["macrogenes"], data.obs["species"], data.obs.index)
        self.assertEqual(load_npz(artifact, "label_free")["macrogenes"].shape, (16, 4))
        with self.assertRaises(ValueError):
            train_pretrain(module, data, scores, output_dir=self.root / "bad", epochs=2, batch_size=7, resume=checkpoint)

    def test_resume_in_fresh_process_without_torch(self):
        data, scores = tiny_prepared_fixture()
        module = SaturnPretrainModule(data.species_names, data.gene_counts, 4, 7, 5)
        trained, _ = train_pretrain(module, data, scores, output_dir=self.root / "run", epochs=1, batch_size=8)
        optimizer = optax.adam(.0005, eps=1e-8)
        record = next(equal_species_batches(data.values, 8, seed=0, epoch=2))
        batch = tuple({key: jnp.asarray(item[key]) for key in ("values", "valid_mask")} for item in record)
        step = make_pretrain_step(module, optimizer, data.gene_embeddings)
        expected, metrics = step(trained, batch)
        inputs = self.root / "inputs.npz"
        np.savez(inputs, scores=scores, proteins=data.gene_embeddings,
                 values_0=record[0]["values"], valid_0=record[0]["valid_mask"],
                 values_1=record[1]["values"], valid_1=record[1]["valid_mask"])
        result = self.root / "result.npz"
        script = """
import sys
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.traverse_util import flatten_dict
from jax_saturn.models.saturn import SaturnPretrainModule
from jax_saturn.training.pretrain import create_state, make_pretrain_step
from jax_saturn.distributed.checkpoint import restore_checkpoint
with np.load(sys.argv[2], allow_pickle=False) as source:
    module = SaturnPretrainModule(('a', 'b'), (4, 3), 4, 7, 5)
    optimizer = optax.adam(.0005, eps=1e-8)
    template = create_state(module, source['scores'], optimizer, 0)
    state, _, _ = restore_checkpoint(sys.argv[1], template)
    batch = tuple({'values': jnp.asarray(source[f'values_{i}']), 'valid_mask': jnp.asarray(source[f'valid_{i}'])} for i in range(2))
    step = make_pretrain_step(module, optimizer, source['proteins'])
    state, metrics = step(state, batch)
    output = {'/'.join(path): np.asarray(value) for path, value in flatten_dict(state.params).items()}
    output['loss'] = np.asarray(metrics['loss'])
    output['rng'] = np.asarray(jax.random.key_data(state.rng))
    np.savez(sys.argv[3], **output)
assert 'torch' not in sys.modules
"""
        self.run_process([sys.executable, "-c", script, str(self.root / "run/checkpoints/epoch_0001"),
                          str(inputs), str(result)], timeout=90)
        with np.load(result, allow_pickle=False) as actual:
            for path, value in flatten_dict(expected.params).items():
                np.testing.assert_array_equal(actual["/".join(path)], value)
            np.testing.assert_array_equal(actual["loss"], metrics["loss"])
            np.testing.assert_array_equal(actual["rng"], jax.random.key_data(expected.rng))

    def test_cli_pretrain_and_existing_artifact_builder(self):
        manifest = write_atlas_fixture(self.root)
        output = self.root / "cli"
        command = [sys.executable, str(ROOT / "scripts/train_saturn_jax.py"), "--in_data", str(manifest),
                   "--work_dir", str(output), "--device", "cpu", "--epochs", "0", "--pretrain_epochs", "1",
                   "--pretrain_batch_size", "16", "--hv_genes", "32", "--num_macrogenes", "4",
                   "--model_dim", "5", "--hidden_dim", "9"]
        self.run_process(command, timeout=120)
        atlas_path = output / "saturn_results/adata_pretrain.h5ad"
        atlas = ad.read_h5ad(atlas_path)
        validate_anndata(atlas)
        self.assertEqual(atlas.shape, (68, 5))
        self.run_process(command + ["--pretrain", "false", "--resume",
            str(output / "shared/pretrain_orbax/checkpoints/epoch_0001")], timeout=120)
        np.testing.assert_array_equal(ad.read_h5ad(atlas_path).X, atlas.X)
        self.assertEqual(pd.read_csv(output / "saturn_results/pretrain_losses.csv")["epoch"].tolist(), [1])
        artifact = self.root / "builder_artifact.npz"
        self.run_process([sys.executable, str(ROOT / "scripts/prepare_label_agnostic_artifacts.py"),
                        "--pretrain-adata", str(atlas_path), "--artifact", str(artifact),
                        "--triplets", str(self.root / "triplets.npz"), "--metadata", str(self.root / "builder_metadata.json"),
                        "--batch-size", "68"], timeout=90)
        self.assertEqual(load_npz(artifact, "label_free")["embeddings"].shape, (68, 5))
        self.assertGreater(len(load_npz(self.root / "triplets.npz", "evaluation_triplets")["anchor"]), 0)


if __name__ == "__main__":
    unittest.main()

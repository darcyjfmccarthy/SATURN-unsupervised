"""Reference candidate sampling, padded updates and baseline resume contracts."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import anndata as ad
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import torch

from distances.cosine_similarity import CosineSimilarity
from losses.triplet_margin_loss import TripletMarginLoss
from miners.triplet_margin_miner import TripletMarginMiner
from model.saturn_model import SATURNMetricModel
from utils.loss_and_miner_utils import get_species_triplet_indices
from jax_saturn.contracts.validation import validate_anndata, validate_metric_history
from jax_saturn.data.batching import metric_batches
from jax_saturn.models.conversion import torch_to_flax
from jax_saturn.models.saturn import SaturnMetricModule
from jax_saturn.training.baseline import create_metric_state, make_metric_step, train_baseline
from jax_saturn.training.mining import cross_species_candidates, filter_triplets, normalize_numpy, pad_triplets


def fixture():
    rng = np.random.default_rng(38)
    species = np.repeat(np.arange(3), 4)
    labels = np.repeat(np.arange(6), 2)
    centers = rng.normal(size=(2, 7)).astype(np.float32)
    values = centers[np.tile([0, 0, 1, 1], 3)] + rng.normal(0, .15, (12, 7)).astype(np.float32)
    return values, labels, species


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, previous)

    def test_candidate_sampling_matches_reference_with_identical_draws(self):
        values, labels, species = fixture()
        for mnn in (False, True):
            numpy_calls, torch_calls = [], []
            def choose(low, high, shape):
                numpy_calls.append((low, high, shape))
                return (np.arange(np.prod(shape)).reshape(shape) % high).astype(np.int64)
            def torch_choose(low, high, shape):
                torch_calls.append((low, high, shape))
                return torch.from_numpy((np.arange(np.prod(shape)).reshape(shape) % high).astype(np.int64))
            with patch("torch.randint", side_effect=torch_choose):
                expected = get_species_triplet_indices(torch.tensor(labels), torch.tensor(species),
                                                       torch.tensor(values), CosineSimilarity(), mnn=mnn)
            actual = cross_species_candidates(values, labels, species, mnn=mnn, randint=choose)
            self.assertEqual(numpy_calls, torch_calls)
            self.assertGreater(len(actual[0]), 0)
            for got, want in zip(actual, expected):
                np.testing.assert_array_equal(got, want.numpy())
            a, p, n = actual
            self.assertTrue(np.all(species[a] != species[p]))
            self.assertTrue(np.all((species[n] == species[a]) | (species[n] == species[p])))
            self.assertTrue(np.all((labels[n] != labels[a]) & (labels[n] != labels[p])))

    def test_miner_semihard_filter_and_padding_match_reference(self):
        values, labels, species = fixture()
        def choose(low, high, shape):
            return (np.arange(np.prod(shape)).reshape(shape) % high).astype(np.int64)
        candidates = cross_species_candidates(values, labels, species, randint=choose)
        miner = TripletMarginMiner(margin=.2, type_of_triplets="semihard", distance=CosineSimilarity())
        with patch("utils.loss_and_miner_utils.get_species_triplet_indices",
                   return_value=tuple(torch.tensor(item) for item in candidates)):
            expected = miner.mine(torch.tensor(values), torch.tensor(labels), torch.tensor(values),
                                  torch.tensor(labels), torch.tensor(species), True)
        actual = filter_triplets(values, candidates)
        for got, want in zip(actual, expected):
            np.testing.assert_array_equal(got, want.numpy())
        padded, mask = pad_triplets(actual)
        for got, want in zip(padded, actual):
            np.testing.assert_array_equal(got[mask], want)
        self.assertTrue(np.all(np.asarray(padded)[:, ~mask] == 0))
        for degenerate_species in (np.zeros(12), species):
            result = cross_species_candidates(values, np.zeros(12), degenerate_species, rng=np.random.default_rng(0))
            self.assertEqual(len(result[0]), 0)

    def test_metric_step_loss_gradient_and_adam_update_parity(self):
        torch.manual_seed(5)
        reference = SATURNMetricModel(input_dim=7, hidden_dim=9, embed_dim=5, dropout=0.)
        module = SaturnMetricModule(7, 9, 5, dropout=0.)
        values, _, _ = fixture()
        params = torch_to_flax(reference.state_dict(), module.init(jax.random.key(0), values)["params"], model_kind="metric")
        optimizer = optax.adam(.001, eps=1e-8)
        state = create_metric_state(params, optimizer, 0)
        preview, step = make_metric_step(module, optimizer)
        tuples = (np.array([0, 1, 4, 5]), np.array([4, 5, 8, 9]), np.array([2, 3, 6, 7]))
        padded, mask = pad_triplets(tuples)
        criterion = TripletMarginLoss(margin=.2, distance=CosineSimilarity())
        torch_optimizer = torch.optim.Adam(reference.parameters(), lr=.001)
        tensor = torch.tensor(values)
        np.testing.assert_allclose(preview(state, values), reference(tensor).detach(), atol=3e-5)
        for update in range(3):
            torch_optimizer.zero_grad(set_to_none=True)
            loss = criterion(torch.nn.functional.normalize(reference(tensor)), torch.arange(len(values)),
                             tuple(torch.tensor(item) for item in tuples))
            loss.backward()
            torch_optimizer.step()
            state, metrics = step(state, values, tuple(jnp.asarray(item) for item in padded), mask)
            with self.subTest(update=update + 1):
                np.testing.assert_allclose(metrics["loss"], loss.detach(), rtol=1e-4, atol=1e-5)
                mapped = torch_to_flax(reference.state_dict(), params, model_kind="metric")
                for path, expected in flatten_dict(mapped).items():
                    np.testing.assert_allclose(flatten_dict(state.params)[path], expected, rtol=1e-4, atol=1e-5)

    def test_metric_batch_contract(self):
        values, labels, species = fixture()
        batches = list(metric_batches(values, species, 5, seed=0, epoch=1, labels=labels, ref_labels=labels))
        selected = np.concatenate([batch["global_indices"][batch["valid_mask"]] for batch in batches])
        np.testing.assert_array_equal(np.sort(selected), np.arange(12))
        for batch in batches:
            self.assertEqual(batch["macrogenes"].shape, (5, 7))
            np.testing.assert_array_equal(batch["macrogenes"][batch["valid_mask"]], values[batch["global_indices"][batch["valid_mask"]]])
        free = next(metric_batches(values, species, 5, seed=0, epoch=1))
        self.assertEqual(set(free), {"macrogenes", "species_codes", "global_indices", "valid_mask"})

    def test_baseline_history_order_and_epoch_resume(self):
        values, labels, species = fixture()
        macros = np.abs(values)
        obs = pd.DataFrame(index=[f"cell_{i}" for i in range(len(values))])
        for name, column in (("labels", labels), ("labels2", labels % 2), ("ref_labels", labels % 2), ("species", species)):
            obs[name] = pd.Categorical(column.astype(str))
        atlas = ad.AnnData(np.zeros((12, 5), dtype=np.float32), obs=obs, obsm={"macrogenes": macros})
        module = SaturnMetricModule(7, 9, 5)
        params = module.init(jax.random.key(1), macros)["params"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full, history, result = train_baseline(params, atlas, output_dir=root / "full/results", epochs=2,
                                                  batch_size=12, hidden_dim=9, model_dim=5, source_manifest_sha256="a" * 64)
            checkpoint = root / "full/shared/metric_orbax/checkpoints/epoch_0001"
            resumed, resumed_history, _ = train_baseline(params, atlas, output_dir=root / "resumed/results", epochs=2,
                                                         batch_size=12, hidden_dim=9, model_dim=5,
                                                         source_manifest_sha256="a" * 64, resume=checkpoint)
            self.assertEqual(history, resumed_history)
            validate_metric_history(pd.read_csv(root / "full/results/metric_history.csv"), epochs=2)
            validate_anndata(result, expected_obs_ids=atlas.obs_names, expected_species=atlas.obs["species"])
            for got, want in zip(jax.tree_util.tree_leaves(resumed.params), jax.tree_util.tree_leaves(full.params)):
                np.testing.assert_array_equal(got, want)
            np.testing.assert_array_equal(result.obsm["macrogenes"], macros)


if __name__ == "__main__":
    unittest.main()

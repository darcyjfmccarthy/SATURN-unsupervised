"""Deterministic value/gradient parity with scvi and repository PyTorch losses."""

import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F
from scvi.distributions import ZeroInflatedNegativeBinomial

from distances.cosine_similarity import CosineSimilarity
from label_agnostic import objectives as reference
from losses.triplet_margin_loss import TripletMarginLoss
from miners.triplet_margin_miner import TripletMarginMiner
from jax_saturn.losses import core, objectives


class LossParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, threads)
        rng = np.random.default_rng(123)
        cls.teacher = rng.normal(size=(18, 5)).astype(np.float32)
        cls.macrogenes = np.abs(rng.normal(size=(18, 7))).astype(np.float32)
        cls.species = np.repeat(["a", "b", "c"], 6)
        cls.codes = np.repeat(np.arange(3), 6).astype(np.int32)
        cls.indices = np.array([0, 3, 6, 9, 12, 15], dtype=np.int64)
        cls.student = (cls.teacher[cls.indices] + rng.normal(0, .2, (6, 5))).astype(np.float32)
        cls.candidates, cls.probabilities, cls.similarities, cls.neighbors = reference.build_preservation_graph(
            cls.teacher, cls.species, neighbor_k=2, negative_k=2)
        cls.positives, _ = reference.build_cross_species_positives(
            cls.teacher, cls.macrogenes, cls.species, candidate_k=3, positives_per_species=2)
        cls.targets = [np.flatnonzero(cls.codes == code) for code in range(3)]
        cls.maps = []
        for target in cls.targets:
            mapping = np.full(18, -1, dtype=np.int64)
            mapping[target] = np.arange(len(target))
            cls.maps.append(mapping)

    def assert_close(self, got, expected, *, rtol=1e-4, atol=1e-5):
        if hasattr(expected, "detach"):
            expected = expected.detach().numpy()
        np.testing.assert_allclose(np.asarray(got), expected, rtol=rtol, atol=atol)

    def parity(self, function, torch_function, values):
        tensor = torch.tensor(values, requires_grad=True)
        expected = torch_function(tensor)
        expected.backward()
        actual, gradient = jax.jit(jax.value_and_grad(function))(jnp.asarray(values))
        self.assert_close(actual, expected)
        self.assert_close(gradient, tensor.grad)
        self.assertTrue(np.isfinite(gradient).all())

    def graph_arguments(self, array):
        return [array(x) for x in (self.indices, self.teacher, self.candidates,
                                   self.probabilities, self.similarities)]

    def infonce_arguments(self, array):
        return [array(self.indices), array(self.teacher), array(self.positives),
                [array(x) for x in self.targets], [array(x) for x in self.maps]]

    def local_arguments(self, array):
        return [array(self.indices), array(self.codes[self.indices]), array(self.teacher),
                array(self.neighbors), [array(x) for x in self.targets], [array(x) for x in self.maps]]

    def test_zinb_values_and_parameter_gradients(self):
        rng = np.random.default_rng(5)
        counts = rng.poisson(3, (4, 5)).astype(np.float32)
        counts[0] = 0
        mu = rng.uniform(0.1, 8, counts.shape).astype(np.float32)
        theta = rng.uniform(0.1, 3, 5).astype(np.float32)
        logits = rng.normal(size=counts.shape).astype(np.float32)
        expected = ZeroInflatedNegativeBinomial(mu=torch.tensor(mu), theta=torch.tensor(theta),
                                                zi_logits=torch.tensor(logits)).log_prob(torch.tensor(counts))
        self.assert_close(jax.jit(core.zinb_log_prob)(counts, mu, theta, logits), expected)
        t_mu, t_theta, t_logits = [torch.tensor(x, requires_grad=True) for x in (mu, theta, logits)]
        loss = -ZeroInflatedNegativeBinomial(mu=t_mu, theta=t_theta, zi_logits=t_logits).log_prob(torch.tensor(counts)).sum()
        loss.backward()
        value, gradients = jax.jit(jax.value_and_grad(core.zinb_reconstruction_loss, argnums=(1, 2, 3)))(counts, mu, theta, logits)
        self.assert_close(value, loss)
        for actual, parameter in zip(gradients, (t_mu, t_theta, t_logits)):
            self.assert_close(actual, parameter.grad)

    def test_zinb_saturated_dropout_gradients(self):
        # Near saturation, subtracting softplus gradients exposes errors hidden
        # by the usual 1e-5 gradient tolerance; Adam amplifies these tiny errors.
        logits = np.tile(np.array([-20., -16., -14., -10., 10., 14., 16., 20.], np.float32), (2, 1))
        counts = np.stack((np.zeros(8), np.full(8, 3))).astype(np.float32)
        mu = np.full_like(counts, .7)
        theta = np.full(8, 3., np.float32)
        tensor = torch.tensor(logits, requires_grad=True)
        loss = -ZeroInflatedNegativeBinomial(mu=torch.tensor(mu), theta=torch.tensor(theta),
            zi_logits=tensor).log_prob(torch.tensor(counts)).sum() / 170
        loss.backward()
        actual = jax.jit(jax.grad(lambda values: core.zinb_reconstruction_loss(
            counts, mu, theta, values) / 170))(jnp.asarray(logits))
        # One float32 ULP of the upstream cotangent allows rounding in the
        # subtraction while detecting premature rounding to a unit derivative.
        self.assert_close(actual, tensor.grad, rtol=1e-6, atol=np.spacing(np.float32(1 / 170)))

    def test_zinb_zero_means_extreme_logits_and_mask(self):
        counts = np.array([[0., 1., 20.], [2., 0., 5.]], dtype=np.float32)
        mu = np.array([[0., 1e-7, 20.], [1., 0., 1e4]], dtype=np.float32)
        theta = np.array([1e-4, 1., 100.], dtype=np.float32)
        logits = np.array([[-50., 0., 50.], [50., -50., 0.]], dtype=np.float32)
        expected = ZeroInflatedNegativeBinomial(mu=torch.tensor(mu), theta=torch.tensor(theta),
                                                zi_logits=torch.tensor(logits), validate_args=False).log_prob(torch.tensor(counts))
        self.assert_close(core.zinb_log_prob(counts, mu, theta, logits), expected, atol=3e-4)
        loss = core.zinb_reconstruction_loss(counts, mu, theta, logits,
                                             valid_mask=jnp.array([True, False]), weights=jnp.array([2., 500.]))
        self.assert_close(loss, -2 * expected[0].sum(), atol=3e-4)

    def test_l1_and_ranking(self):
        values = np.log(np.arange(1, 13).reshape(3, 4)).astype(np.float32)
        self.parity(core.l1_loss, lambda tensor: tensor.exp().abs().sum(), values)
        rng = np.random.default_rng(8)
        learned = rng.normal(size=(12, 8)).astype(np.float32)
        protein = rng.normal(size=(12, 9)).astype(np.float32)
        indices = rng.integers(0, 12, size=12)
        self.parity(lambda tensor: core.gene_weight_ranking_loss(tensor, jnp.asarray(protein), indices),
                    lambda tensor: F.mse_loss(F.cosine_similarity(tensor, tensor[indices]),
                                              F.cosine_similarity(torch.tensor(protein), torch.tensor(protein)[indices]), reduction="sum"), learned)
        first = core.gene_weight_ranking_loss(learned, protein, key=jax.random.key(0))
        again = core.gene_weight_ranking_loss(learned, protein, key=jax.random.key(0))
        self.assert_close(first, again)

    def test_triplet_loss_and_filtering(self):
        rng = np.random.default_rng(11)
        values = rng.normal(size=(7, 5)).astype(np.float32)
        indices = (np.array([0, 1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 5, 6]), np.array([3, 4, 5, 6, 0, 1]))
        torch_indices = tuple(torch.tensor(x) for x in indices)
        labels = torch.arange(7)
        criterion = TripletMarginLoss(margin=.2, distance=CosineSimilarity())
        self.parity(lambda tensor: core.triplet_margin_loss(tensor, indices),
                    lambda tensor: criterion(tensor, labels, torch_indices), values)
        for kind in ("all", "semihard", "hard", "easy", "unfiltered"):
            miner = TripletMarginMiner(margin=.2, type_of_triplets=kind, distance=CosineSimilarity())
            with patch("utils.loss_and_miner_utils.get_species_triplet_indices", return_value=torch_indices):
                selected = miner.mine(torch.tensor(values), labels, torch.tensor(values), labels,
                                      torch.zeros(7, dtype=torch.long), False)
            mask = core.triplet_filter_mask(values, indices, kind=kind)
            for actual, expected in zip(indices, selected):
                np.testing.assert_array_equal(actual[np.asarray(mask)], expected)
        empty = tuple(jnp.array([], dtype=jnp.int32) for _ in range(3))
        value, gradient = jax.value_and_grad(lambda x: core.triplet_margin_loss(x, empty))(values)
        self.assertEqual(float(value), 0)
        np.testing.assert_array_equal(gradient, np.zeros_like(values))

    def test_preservation_value_and_gradient(self):
        self.parity(lambda tensor: objectives.preservation_distillation_loss(tensor, *self.graph_arguments(jnp.asarray)),
                    lambda tensor: reference.preservation_distillation_loss(tensor, *self.graph_arguments(torch.tensor)), self.student)

    def test_cross_species_infonce_value_gradient_and_coverage(self):
        self.parity(lambda tensor: objectives.multi_positive_infonce_loss(tensor, *self.infonce_arguments(jnp.asarray))[0],
                    lambda tensor: reference.multi_positive_infonce_loss(tensor, *self.infonce_arguments(torch.tensor))[0], self.student)
        actual = objectives.multi_positive_infonce_loss(self.student, *self.infonce_arguments(jnp.asarray))
        expected = reference.multi_positive_infonce_loss(torch.tensor(self.student), *self.infonce_arguments(torch.tensor))
        self.assertEqual(int(actual[1]), expected[1])

    def test_local_infonce_value_gradient_and_coverage(self):
        self.parity(lambda tensor: objectives.within_species_graph_infonce_loss(tensor, *self.local_arguments(jnp.asarray), positive_k=2)[0],
                    lambda tensor: reference.within_species_graph_infonce_loss(tensor, *self.local_arguments(torch.tensor), positive_k=2)[0], self.student)
        actual = objectives.within_species_graph_infonce_loss(self.student, *self.local_arguments(jnp.asarray), positive_k=2)
        expected = reference.within_species_graph_infonce_loss(torch.tensor(self.student), *self.local_arguments(torch.tensor), positive_k=2)
        self.assertEqual(int(actual[1]), expected[1])

    def test_mmd_value_and_gradient(self):
        codes = self.codes[self.indices]
        self.parity(lambda tensor: objectives.multi_species_mmd(tensor, codes, .7, num_species=3),
                    lambda tensor: reference.multi_species_mmd(tensor, torch.tensor(codes), .7), self.student)

    def test_sinkhorn_plan_parity_and_marginals(self):
        cost = np.random.default_rng(0).uniform(0, 2, (4, 6)).astype(np.float32)
        for mass in (.5, .8, 1.):
            actual = jax.jit(lambda x: objectives.partial_sinkhorn(x, transported_mass=mass, iterations=200))(cost)
            expected = reference.partial_sinkhorn(torch.tensor(cost), transported_mass=mass, iterations=200)
            self.assert_close(actual, expected)
            self.assert_close(actual.sum(), mass)
            self.assertTrue(np.all(np.asarray(actual.sum(axis=1)) <= 1 / 4 + .005))
            self.assertTrue(np.all(np.asarray(actual.sum(axis=0)) <= 1 / 6 + .005))

    def test_ot_value_gradient_and_masses(self):
        teacher = self.teacher[self.indices]
        macro = self.macrogenes[self.indices]
        codes = self.codes[self.indices]
        self.parity(lambda tensor: objectives.partial_ot_alignment_loss(tensor, jnp.asarray(teacher), jnp.asarray(macro), codes,
                                                                        num_species=3, iterations=30)[0],
                    lambda tensor: reference.partial_ot_alignment_loss(tensor, torch.tensor(teacher), torch.tensor(macro),
                                                                        torch.tensor(codes), iterations=30)[0], self.student)
        actual = objectives.partial_ot_alignment_loss(self.student, jnp.asarray(teacher), jnp.asarray(macro), codes,
                                                      num_species=3, iterations=30)
        expected = reference.partial_ot_alignment_loss(torch.tensor(self.student), torch.tensor(teacher), torch.tensor(macro),
                                                        torch.tensor(codes), iterations=30)
        self.assert_close(actual[1], expected[1])
        teacher_gradient = jax.grad(lambda t: objectives.partial_ot_alignment_loss(self.student, t, jnp.asarray(macro), codes,
                                                                                   num_species=3, iterations=30)[0])(teacher)
        np.testing.assert_array_equal(teacher_gradient, np.zeros_like(teacher))

    def test_padding_invariance_across_objectives(self):
        # Duplicate a real row with extreme values; valid_mask must fully exclude it.
        padded = np.concatenate((self.student, np.ones((1, 5), dtype=np.float32) * 999))
        indices = jnp.concatenate((jnp.asarray(self.indices), jnp.array([0])))
        mask = jnp.array([True] * 6 + [False])
        codes = jnp.concatenate((jnp.asarray(self.codes[self.indices]), jnp.array([0])))
        teacher = jnp.concatenate((jnp.asarray(self.teacher[self.indices]), jnp.ones((1, 5))))
        macro = jnp.concatenate((jnp.asarray(self.macrogenes[self.indices]), jnp.ones((1, 7))))
        pairs = [
            (lambda x, m: objectives.preservation_distillation_loss(x, indices, jnp.asarray(self.teacher),
                jnp.asarray(self.candidates), jnp.asarray(self.probabilities), jnp.asarray(self.similarities), valid_mask=m),
             lambda x: objectives.preservation_distillation_loss(x, *self.graph_arguments(jnp.asarray))),
            (lambda x, m: objectives.multi_positive_infonce_loss(x, indices, jnp.asarray(self.teacher), jnp.asarray(self.positives),
                [jnp.asarray(t) for t in self.targets], [jnp.asarray(t) for t in self.maps], valid_mask=m)[0],
             lambda x: objectives.multi_positive_infonce_loss(x, *self.infonce_arguments(jnp.asarray))[0]),
            (lambda x, m: objectives.within_species_graph_infonce_loss(x, indices, codes, jnp.asarray(self.teacher),
                jnp.asarray(self.neighbors), [jnp.asarray(t) for t in self.targets], [jnp.asarray(t) for t in self.maps], positive_k=2, valid_mask=m)[0],
             lambda x: objectives.within_species_graph_infonce_loss(x, *self.local_arguments(jnp.asarray), positive_k=2)[0]),
            (lambda x, m: objectives.multi_species_mmd(x, codes, .7, num_species=3, valid_mask=m),
             lambda x: objectives.multi_species_mmd(x, self.codes[self.indices], .7, num_species=3)),
            (lambda x, m: objectives.partial_ot_alignment_loss(x, teacher, macro, codes, num_species=3, iterations=30, valid_mask=m)[0],
             lambda x: objectives.partial_ot_alignment_loss(x, jnp.asarray(self.teacher[self.indices]), jnp.asarray(self.macrogenes[self.indices]),
                self.codes[self.indices], num_species=3, iterations=30)[0]),
        ]
        for index, (with_padding, without_padding) in enumerate(pairs):
            with self.subTest(objective=index):
                expected, expected_gradient = jax.value_and_grad(without_padding)(jnp.asarray(self.student))
                actual, gradient = jax.jit(jax.value_and_grad(lambda x: with_padding(x, mask)))(jnp.asarray(padded))
                self.assert_close(actual, expected)
                self.assert_close(gradient[:-1], expected_gradient)
                np.testing.assert_array_equal(gradient[-1], np.zeros(5))

    def test_empty_candidates_and_species_are_finite_with_zero_gradients(self):
        codes = jnp.zeros(6, dtype=jnp.int32)
        missing = jnp.full(self.candidates.shape, -1)
        positive = jnp.full(self.positives.shape, -1)
        functions = [
            lambda x: objectives.preservation_distillation_loss(x, jnp.asarray(self.indices), jnp.asarray(self.teacher), missing,
                                                                jnp.asarray(self.probabilities), jnp.asarray(self.similarities)),
            lambda x: objectives.multi_positive_infonce_loss(x, jnp.asarray(self.indices), jnp.asarray(self.teacher), positive,
                [jnp.asarray(t) for t in self.targets], [jnp.asarray(t) for t in self.maps])[0],
            lambda x: objectives.multi_species_mmd(x, codes, .7, num_species=3),
            lambda x: objectives.partial_ot_alignment_loss(x, jnp.asarray(self.teacher[self.indices]),
                jnp.asarray(self.macrogenes[self.indices]), codes, num_species=3, iterations=10)[0],
        ]
        for index, function in enumerate(functions):
            with self.subTest(objective=index):
                value, gradient = jax.jit(jax.value_and_grad(function))(jnp.asarray(self.student))
                self.assertEqual(float(value), 0)
                np.testing.assert_array_equal(gradient, np.zeros_like(self.student))
        zero_gradient = jax.grad(lambda x: core.normalize(x).sum())(jnp.zeros((2, 5)))
        self.assertTrue(np.isfinite(zero_gradient).all())


if __name__ == "__main__":
    unittest.main()

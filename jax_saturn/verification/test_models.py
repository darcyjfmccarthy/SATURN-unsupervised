"""CPU fp32 parity against the actual PyTorch model, including gradients and JIT."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax.core import unfreeze
from flax.traverse_util import flatten_dict

from jax_saturn.models.conversion import export_pytorch_checkpoint, torch_to_flax
from jax_saturn.models.saturn import (
    FullBlock, SaturnMetricModule, SaturnPretrainModule,
    init_pretrain_params, metric_apply, pretrain_apply,
)
from model.saturn_model import SATURNMetricModel, SATURNPretrainModel, full_block


RTOL = 3e-5
ATOL = 3e-5


class ModelParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, threads)
        torch.manual_seed(42)
        cls.rng = np.random.default_rng(42)
        cls.names = ("h_sapiens", "m_murinus", "m_musculus")
        cls.counts = (4, 3, 5)
        cls.offsets = {"h_sapiens": (0, 4), "m_murinus": (4, 7), "m_musculus": (7, 12)}
        cls.scores = cls.rng.uniform(0.05, 1, (12, 6)).astype(np.float32)
        cls.reference = SATURNPretrainModel(
            torch.tensor(cls.scores), hidden_dim=9, embed_dim=5, dropout=0.1,
            species_to_gene_idx=cls.offsets).eval()
        with torch.no_grad():
            for name, value in cls.reference.named_parameters():
                if "norm" in name:
                    value.uniform_(0.6, 1.4) if name.endswith("weight") else value.uniform_(-0.3, 0.3)
        cls.module = SaturnPretrainModule(cls.names, cls.counts, 6, 9, 5)
        cls.template = init_pretrain_params(cls.module, jax.random.key(0), cls.scores)
        cls.params = torch_to_flax(cls.reference.state_dict(), cls.template, model_kind="pretrain")
        cls.metric_reference = SATURNMetricModel(input_dim=6, hidden_dim=9, embed_dim=5,
                                                dropout=0.1, species_to_gene_idx=cls.offsets).eval()
        cls.metric_reference.encoder.load_state_dict(cls.reference.encoder.state_dict())
        cls.metric_reference.cl_layer_norm.load_state_dict(cls.reference.cl_layer_norm.state_dict())
        cls.metric = SaturnMetricModule(6, 9, 5)
        cls.metric_template = cls.metric.init(jax.random.key(1), jnp.ones((2, 6)))["params"]
        cls.metric_params = torch_to_flax(cls.reference.state_dict(), cls.metric_template, model_kind="metric")

    def assert_close(self, actual, expected):
        if hasattr(expected, "detach"):
            expected = expected.detach().numpy()
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=RTOL, atol=ATOL)

    def test_full_block_and_layernorm_epsilon(self):
        reference = full_block(6, 5).eval()
        module = FullBlock(5)
        values = np.random.default_rng(1).normal(size=(3, 6)).astype(np.float32)
        params = {"dense": {"kernel": reference[0].weight.detach().numpy().T,
                            "bias": reference[0].bias.detach().numpy()},
                  "layer_norm": {"scale": reference[1].weight.detach().numpy(),
                                 "bias": reference[1].bias.detach().numpy()}}
        self.assert_close(module.apply({"params": params}, values), reference(torch.tensor(values)))
        # Near-constant activations expose an epsilon mismatch hidden by typical inputs.
        params["dense"]["kernel"] = np.zeros((6, 5), dtype=np.float32)
        params["dense"]["bias"] = np.linspace(0, 1e-4, 5, dtype=np.float32)
        with torch.no_grad():
            reference[0].weight.zero_()
            reference[0].bias.copy_(torch.tensor(params["dense"]["bias"]))
        self.assert_close(module.apply({"params": params}, values), reference(torch.tensor(values)))

    def test_pretrain_forward_every_species_and_singletons(self):
        rng = np.random.default_rng(8)
        apply_jit = jax.jit(lambda values, code: pretrain_apply(self.module, self.params, values, code),
                            static_argnums=(1,))
        for code, name in enumerate(self.names):
            for batch_size in (1, 2, 5):
                with self.subTest(species=name, batch_size=batch_size):
                    values = rng.poisson(3, (batch_size, self.counts[code])).astype(np.float32)
                    expected = self.reference(torch.tensor(values), name)
                    actual = apply_jit(values, code)
                    for index, (got, want) in enumerate(zip(actual, expected)):
                        if want is None:
                            self.assertIsNone(got)
                        else:
                            want = want.detach().numpy()
                            if index == 0:
                                want = np.atleast_2d(want)
                            self.assert_close(got, want)
                    self.assertEqual(actual.macrogenes.shape, (batch_size, 6))
                    self.assertEqual(actual.px_rate.shape, values.shape)
                    self.assert_close(actual.px_rate.sum(axis=-1), values.sum(axis=-1))

    def test_metric_forward_and_pretrain_encoder_transfer(self):
        for batch_size in (1, 4):
            values = np.random.default_rng(batch_size).normal(size=(batch_size, 6)).astype(np.float32)
            expected = self.metric_reference(torch.tensor(values))
            actual = jax.jit(lambda x: metric_apply(self.metric, self.metric_params, x))(values)
            self.assert_close(actual, expected)
        values = np.ones((2, 4), dtype=np.float32)
        pretrain = pretrain_apply(self.module, self.params, values, 0)
        self.assert_close(metric_apply(self.metric, self.metric_params, pretrain.macrogenes), pretrain.embedding)
        # The metric checkpoint retains cl_layer_norm but must never apply it.
        altered = unfreeze(self.metric_params)
        altered["cl_layer_norm"]["scale"] = jnp.zeros(6)
        altered["cl_layer_norm"]["bias"] = jnp.ones(6) * 100
        self.assert_close(metric_apply(self.metric, altered, pretrain.macrogenes), pretrain.embedding)

    def test_ranking_embedding_parity(self):
        expected = self.reference.p_weights_embeddings(self.reference.p_weights.exp().T)
        actual = self.module.apply({"params": self.params}, method=self.module.gene_weight_embeddings)
        self.assert_close(actual, expected)

    def test_conversion_snapshots_mutable_source_arrays(self):
        source = {name: value.detach().numpy().copy()
                  for name, value in self.reference.state_dict().items()}
        converted = torch_to_flax(source, self.template, model_kind="pretrain")
        before = {path: np.asarray(value).copy() for path, value in flatten_dict(converted).items()}
        for value in source.values():
            value[...] = 99.
        for path, value in flatten_dict(converted).items():
            np.testing.assert_array_equal(value, before[path])

    def test_pretrain_gradient_parity(self):
        values = np.random.default_rng(10).poisson(4, (3, 3)).astype(np.float32)
        reference = self.reference
        reference.zero_grad(set_to_none=True)
        output = reference(torch.tensor(values), "m_murinus")
        ranking = reference.p_weights_embeddings(reference.p_weights.exp().T)
        loss = sum(x.square().mean() for x in (output[0], output[1], output[4], output[5], output[6]))
        loss = loss + 0.01 * ranking.square().mean()
        loss.backward()

        def objective(params):
            out = pretrain_apply(self.module, params, values, 1)
            rank = self.module.apply({"params": params}, method=self.module.gene_weight_embeddings)
            return sum(jnp.mean(x ** 2) for x in (out.macrogenes, out.embedding, out.px_rate, out.px_r, out.px_drop)) + 0.01 * jnp.mean(rank ** 2)

        loss_jax, gradients = jax.jit(jax.value_and_grad(objective))(self.params)
        self.assert_close(loss_jax, loss)
        reference_gradients = {name: torch.zeros_like(value) if value.grad is None else value.grad
                               for name, value in reference.named_parameters()}
        expected = flatten_dict(torch_to_flax(reference_gradients, self.template, model_kind="pretrain"))
        actual = flatten_dict(gradients)
        for path in expected:
            with self.subTest(parameter="/".join(path)):
                self.assert_close(actual[path], expected[path])

    def test_metric_gradient_parity(self):
        values = np.random.default_rng(2).normal(size=(4, 6)).astype(np.float32)
        self.metric_reference.zero_grad(set_to_none=True)
        loss = self.metric_reference(torch.tensor(values)).square().mean()
        loss.backward()
        gradients = jax.grad(lambda params: jnp.mean(metric_apply(self.metric, params, values) ** 2))(self.metric_params)
        reference_gradients = {name: torch.zeros_like(value) if value.grad is None else value.grad
                               for name, value in self.metric_reference.named_parameters()}
        expected = flatten_dict(torch_to_flax(reference_gradients, self.metric_template, model_kind="metric"))
        for path, actual in flatten_dict(gradients).items():
            with self.subTest(parameter="/".join(path)):
                self.assert_close(actual, expected[path])

    def test_dropout_rng_and_eval_determinism(self):
        values = jnp.ones((12, 4))
        key = {"dropout": jax.random.key(9)}
        first = pretrain_apply(self.module, self.params, values, 0, train=True, rngs=key)
        again = pretrain_apply(self.module, self.params, values, 0, train=True, rngs=key)
        changed = pretrain_apply(self.module, self.params, values, 0, train=True,
                                 rngs={"dropout": jax.random.key(10)})
        np.testing.assert_array_equal(first.embedding, again.embedding)
        self.assertFalse(np.array_equal(first.embedding, changed.embedding))
        evaluate = pretrain_apply(self.module, self.params, values, 0)
        evaluate_again = pretrain_apply(self.module, self.params, values, 0, rngs=key)
        np.testing.assert_array_equal(evaluate.embedding, evaluate_again.embedding)
        np.testing.assert_array_equal(evaluate.px_rate, evaluate_again.px_rate)

    def test_all_heads_initialized_and_bfloat16_policy(self):
        mixed = SaturnPretrainModule(self.names, self.counts, 6, 9, 5, dtype=jnp.bfloat16)
        params = init_pretrain_params(mixed, jax.random.key(6), self.scores)
        self.assertTrue(all(value.dtype == jnp.float32 for value in jax.tree_util.tree_leaves(params)))
        for code in range(3):
            result = mixed.apply({"params": params}, jnp.ones((2, self.counts[code])), code)
            self.assertTrue(all(value.dtype == jnp.float32 for value in result if value is not None))
            self.assertTrue(all(np.isfinite(value).all() for value in result if value is not None))

    def test_zero_counts_and_one_hot_scores(self):
        values = jnp.zeros((2, 4))
        actual = pretrain_apply(self.module, self.params, values, 0)
        self.assert_close(actual.px_rate, np.zeros((2, 4)))
        scores = np.eye(6, dtype=np.float32)[np.arange(12) % 6]
        params = init_pretrain_params(self.module, jax.random.key(7), scores)
        np.testing.assert_array_equal(jnp.exp(params["macrogene"]["log_gene_to_macrogene"]), scores.T)
        result = pretrain_apply(self.module, params, jnp.ones((2, 4)), 0)
        self.assertTrue(np.isfinite(result.px_rate).all())

    def test_conversion_rejects_missing_extra_and_wrong_shapes(self):
        state = dict(self.reference.state_dict())
        wrong = dict(state)
        del wrong["encoder.0.0.bias"]
        with self.assertRaises(ValueError):
            torch_to_flax(wrong, self.template, model_kind="pretrain")
        for key, value in [("fc_mu.weight", torch.ones(5, 9)),
                           ("encoder.0.0.weight", torch.ones(7, 6)),
                           ("px_rs.h_sapiens", torch.full((4,), float("nan")))]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                torch_to_flax({**state, key: value}, self.template, model_kind="pretrain")

    def test_optional_pytorch_export_round_trip(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pretrain.pt"
            export_pytorch_checkpoint(path, self.params)
            state = torch.load(path, weights_only=True, map_location="cpu")
            for name, expected in self.reference.state_dict().items():
                np.testing.assert_array_equal(state[name], expected.detach().cpu().numpy())
            export_pytorch_checkpoint(path, self.metric_params)
            state = torch.load(path, weights_only=True, map_location="cpu")
            self.metric_reference.load_state_dict(state, strict=True)

    def test_invalid_model_configuration_and_shapes(self):
        for code, values in [(3, jnp.ones((2, 4))), (0, jnp.ones((2, 3))), (0, jnp.ones(4))]:
            with self.subTest(code=code), self.assertRaises(ValueError):
                pretrain_apply(self.module, self.params, values, code)
        with self.assertRaises(ValueError):
            init_pretrain_params(self.module, jax.random.key(0), np.ones((3, 6)))
        with self.assertRaises(ValueError):
            invalid = SaturnPretrainModule(("z", "a"), (3, 4), 6, 9, 5)
            invalid.init(jax.random.key(0), jnp.ones((2, 3)), 0)


if __name__ == "__main__":
    unittest.main()

"""Bounded end-to-end optimizer parity against the actual PyTorch model."""

import unittest
from unittest.mock import patch

from flax import linen as nn
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import torch.nn.functional as F

from label_agnostic import objectives as reference_losses
from jax_saturn.models.conversion import torch_to_flax
from jax_saturn.models.saturn import SaturnMetricModule, SaturnPretrainModule
from jax_saturn.training.label_agnostic import (
    LabelAgnosticTrainState, LabelConfig, build_graphs, make_label_step,
)
from jax_saturn.training.pretrain import create_state, make_pretrain_step
from model.saturn_model import SATURNMetricModel, SATURNPretrainModel


class UpdateParityTests(unittest.TestCase):
    def test_three_label_free_adam_updates_for_each_objective(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        rng = np.random.default_rng(123)
        artifact = {"embeddings": rng.normal(size=(18, 5)).astype(np.float32),
                    "macrogenes": np.abs(rng.normal(size=(18, 7))).astype(np.float32),
                    "species": np.repeat(["a", "b", "c"], 6),
                    "obs_ids": np.array([f"cell_{i}" for i in range(18)])}
        config = LabelConfig(epochs=2, batch_size=8, hidden_dim=9, model_dim=5,
                             candidate_k=3, positives_per_species=2,
                             local_positive_k=2, ot_iterations=30)
        for objective in ("infonce", "mmd", "ot"):
            torch.manual_seed(47)
            reference = SATURNMetricModel(input_dim=7, hidden_dim=9, embed_dim=5, dropout=0.)
            module = SaturnMetricModule(7, 9, 5, dropout=0.)
            template = module.init(jax.random.key(0), artifact["macrogenes"])["params"]
            params = torch_to_flax(reference.state_dict(), template, model_kind="metric")
            optimizer = optax.adam(config.learning_rate, eps=1e-8)
            state = LabelAgnosticTrainState(
                params=params, opt_state=optimizer.init(params), rng=jax.random.key(0),
                step=jnp.int32(0), epoch=jnp.int32(0), best_params=params,
                best_epoch=0, best_step=0, best_mixing=0., best_recall=0.,
                preservation_weight=.7, config={}, diagnostics={})
            _, codes, graphs, _, bandwidth, _ = build_graphs(artifact, objective, config)
            # Use fully populated small reference graphs. Its KL implementation
            # evaluates 0 * -inf for absent candidates in larger padded graphs.
            preservation_graph = reference_losses.build_preservation_graph(
                artifact["embeddings"], artifact["species"], neighbor_k=2, negative_k=2,
                temperature=config.preservation_temperature, seed=config.seed)
            for key, value in zip(("candidates", "probabilities", "similarities", "neighbors"), preservation_graph):
                graphs[key] = jnp.asarray(value)
            step, _ = make_label_step(module, optimizer, objective, graphs, bandwidth, config)

            def to_torch(value):
                array = np.asarray(value)
                return torch.tensor(array, dtype=torch.long if array.dtype.kind in "iu" else torch.float32)

            torch_graphs = {key: tuple(to_torch(item) for item in value) if isinstance(value, tuple)
                            else None if value is None else to_torch(value)
                            for key, value in graphs.items()}
            # Freeze the identical inference bank for this bounded trajectory.
            with torch.no_grad():
                bank = F.normalize(reference(torch.tensor(artifact["macrogenes"])), dim=1)
            torch_optimizer = torch.optim.Adam(reference.parameters(), lr=config.learning_rate, eps=1e-8)
            for update, weight in enumerate((.7, 1.3, .2)):
                indices = np.array([0, 3, 6, 9, 12, 15]) + update
                selected_codes = codes[indices]
                values = artifact["macrogenes"][indices]
                batch = {"macrogenes": jnp.asarray(np.concatenate((values, np.full((2, 7), 999., dtype=np.float32)))),
                         "global_indices": jnp.asarray(np.r_[indices, 0, 0], dtype=jnp.int32),
                         "species_codes": jnp.asarray(np.r_[selected_codes, 0, 0]),
                         "valid_mask": jnp.array([True] * 6 + [False] * 2)}
                ti, tc = to_torch(indices), to_torch(selected_codes)
                torch_optimizer.zero_grad(set_to_none=True)
                output = F.normalize(reference(torch.tensor(values)), dim=1)
                if objective == "infonce":
                    alignment, coverage = reference_losses.multi_positive_infonce_loss(
                        output, ti, bank, torch_graphs["positives"], torch_graphs["targets"],
                        torch_graphs["maps"], temperature=config.infonce_temperature)
                elif objective == "mmd":
                    alignment = reference_losses.multi_species_mmd(output, tc, bandwidth)
                    coverage = len(indices)
                else:
                    alignment, masses = reference_losses.partial_ot_alignment_loss(
                        output, torch_graphs["teacher"][ti], torch_graphs["macrogenes"][ti], tc,
                        epsilon=config.ot_epsilon, transported_mass=config.ot_mass,
                        iterations=config.ot_iterations)
                    coverage = sum(masses) / max(sum(mass > 0 for mass in masses), 1)
                preservation = reference_losses.preservation_distillation_loss(
                    output, ti, bank, torch_graphs["candidates"], torch_graphs["probabilities"],
                    torch_graphs["similarities"], temperature=config.preservation_temperature)
                local, local_coverage = reference_losses.within_species_graph_infonce_loss(
                    output, ti, tc, bank, torch_graphs["neighbors"], torch_graphs["targets"],
                    torch_graphs["maps"], positive_k=config.local_positive_k,
                    temperature=config.local_graph_temperature)
                loss = alignment + weight * preservation + config.local_graph_weight * local
                loss.backward()
                gradient_norm = torch.sqrt(sum(parameter.grad.square().sum()
                    for parameter in reference.parameters() if parameter.grad is not None))
                torch_optimizer.step()
                state = state.replace(preservation_weight=weight)
                state, metrics = step(state, batch, jnp.asarray(bank.numpy()))
                with self.subTest(objective=objective, update=update + 1):
                    for key, expected in (("metric_loss", loss), ("alignment_loss", alignment),
                                          ("preservation_loss", preservation), ("local_graph_loss", local),
                                          ("gradient_norm", gradient_norm), ("coverage", coverage),
                                          ("local_coverage", local_coverage)):
                        expected = expected.detach().numpy() if torch.is_tensor(expected) else expected
                        np.testing.assert_allclose(metrics[key], expected, rtol=1e-4, atol=1e-5, err_msg=key)
                    expected = flatten_dict(torch_to_flax(reference.state_dict(), params, model_kind="metric"))
                    for path, actual in flatten_dict(state.params).items():
                        np.testing.assert_allclose(actual, expected[path], rtol=1e-4, atol=1e-5,
                                                   err_msg="/".join(path))
                    self.assertEqual(int(state.step), update + 1)

    def test_three_mixed_species_pretrain_adam_updates(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.manual_seed(73)
        rng = np.random.default_rng(73)
        names, counts = ("human", "lemur", "mouse"), (4, 3, 5)
        offsets = {"human": (0, 4), "lemur": (4, 7), "mouse": (7, 12)}
        scores = rng.uniform(.1, 1, (12, 6)).astype(np.float32)
        proteins = rng.normal(size=(12, 8)).astype(np.float32)
        reference = SATURNPretrainModel(torch.tensor(scores), hidden_dim=9, embed_dim=5,
                                       dropout=0., species_to_gene_idx=offsets).train()
        # Match deterministic dropout inputs rather than unrelated framework RNGs.
        # Include the reference's fixed 0.1 scale-decoder dropout in this control.
        for layer in reference.modules():
            if isinstance(layer, torch.nn.Dropout):
                layer.p = 0.
        module = SaturnPretrainModule(names, counts, 6, 9, 5, dropout=0.)
        optimizer = optax.adam(.0005, eps=1e-8)
        state = create_state(module, scores, optimizer, 0)
        params = torch_to_flax(reference.state_dict(), state.params, model_kind="pretrain")
        state = state.replace(params=params, opt_state=optimizer.init(params))
        step = make_pretrain_step(module, optimizer, proteins, l1_penalty=.01, pe_sim_penalty=.2)
        torch_optimizer = torch.optim.Adam(reference.parameters(), lr=.0005, eps=1e-8)

        # The production update compiles while both frameworks receive identity
        # dropout. Separate tests cover dropout/RNG reproducibility and resume.
        with patch.object(nn.Dropout, "__call__", lambda self, inputs, **kwargs: inputs):
            for update in range(3):
                batch = []
                torch_optimizer.zero_grad(set_to_none=True)
                species_losses = []
                for code, (name, genes) in enumerate(zip(names, counts)):
                    valid_count = 3 - ((code + update) % 3)
                    values = rng.poisson(3, (4, genes)).astype(np.float32)
                    values[valid_count:] = 0
                    valid = np.arange(4) < valid_count
                    batch.append({"values": jnp.asarray(values), "valid_mask": jnp.asarray(valid)})
                    tensor = torch.tensor(values[valid])
                    output = reference(tensor, name)
                    species_losses.append(reference.get_reconstruction_loss(tensor, *output[4:]).mean())

                _, _, ranking_key = jax.random.split(state.rng, 3)
                pairs = np.asarray(jax.random.randint(ranking_key, (12,), 0, 12))
                lasso = .01 * reference.lasso_loss(reference.p_weights.exp())
                with patch("torch.randint", return_value=torch.tensor(pairs, dtype=torch.long)):
                    ranking = .2 * reference.gene_weight_ranking_loss(reference.p_weights.exp(), torch.tensor(proteins))
                loss = sum(species_losses) + lasso + ranking
                loss.backward()
                gradient_norm = torch.sqrt(sum(parameter.grad.square().sum()
                    for parameter in reference.parameters() if parameter.grad is not None))
                torch_optimizer.step()
                state, metrics = step(state, tuple(batch))

                with self.subTest(update=update + 1):
                    for key, expected in (("loss", loss), ("l1_loss", lasso),
                                          ("ranking_loss", ranking), ("gradient_norm", gradient_norm)):
                        np.testing.assert_allclose(metrics[key], expected.detach().numpy(), rtol=1e-4, atol=1e-5,
                                                   err_msg=key)
                    np.testing.assert_allclose(metrics["species_losses"],
                        [item.detach().numpy() for item in species_losses], rtol=1e-4, atol=1e-5)
                    expected = flatten_dict(torch_to_flax(reference.state_dict(), params, model_kind="pretrain"))
                    for path, actual in flatten_dict(state.params).items():
                        np.testing.assert_allclose(actual, expected[path], rtol=1e-4, atol=1e-5,
                                                   err_msg="/".join(path))
                    self.assertEqual(int(state.step), update + 1)


if __name__ == "__main__":
    unittest.main()

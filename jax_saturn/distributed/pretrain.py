"""Replicated pretraining with global valid-count reconstruction reductions."""

from functools import partial

import jax
import jax.numpy as jnp
import optax

from jax_saturn.losses.core import gene_weight_ranking_loss, l1_loss, zinb_reconstruction_loss
from .metric import unreplicate_state, validate_devices


def make_pretrain_pmap(module, optimizer, gene_embeddings, devices, *, l1_penalty=0., pe_sim_penalty=.2):
    devices = validate_devices(devices)
    count = len(devices)
    proteins = jnp.asarray(gene_embeddings, dtype=jnp.float32)

    @partial(jax.pmap, devices=devices)
    def split_keys(rng):
        return tuple(jax.random.split(rng, 3))

    @partial(jax.pmap, axis_name="data", devices=devices,
             static_broadcasted_argnums=(4,), in_axes=(0, 0, 0, 0, None))
    def reconstruction_gradient(params, values, valid_mask, dropout_rng, code):
        key = jax.random.fold_in(jax.random.fold_in(dropout_rng, code), jax.lax.axis_index("data"))
        denominator = jnp.maximum(jax.lax.psum(jnp.sum(valid_mask), "data"), 1)
        def objective(params):
            output = module.apply({"params": params}, values, code, train=True, rngs={"dropout": key})
            reconstruction = zinb_reconstruction_loss(values, output.px_rate, output.px_r,
                                                       output.px_drop, valid_mask=valid_mask)
            # pmean below divides by count: compensate to recover the sum of
            # all shard gradients divided by the GLOBAL valid-cell count.
            return reconstruction * count / denominator
        loss, gradients = jax.value_and_grad(objective)(params)
        return jax.lax.pmean(loss, "data"), jax.lax.pmean(gradients, "data")

    @partial(jax.pmap, axis_name="data", devices=devices, in_axes=(0, None, 0, 0))
    def regularizer_gradient(params, proteins, dropout_rng, ranking_rng):
        # Shared regularizers use identical RNGs on every replica. Their pmean
        # retains one contribution rather than multiplying it by device count.
        key = jax.random.fold_in(dropout_rng, len(module.species_names))
        def objective(params):
            log_weights = params["macrogene"]["log_gene_to_macrogene"]
            lasso = l1_penalty * l1_loss(log_weights)
            learned = module.apply({"params": params}, train=True, method=module.gene_weight_embeddings,
                                   rngs={"dropout": key})
            ranking = pe_sim_penalty * gene_weight_ranking_loss(learned, proteins, key=ranking_rng)
            return lasso + ranking, {"l1_loss": lasso, "ranking_loss": ranking}
        (loss, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(params)
        return (loss, metrics), jax.lax.pmean(gradients, "data")

    @partial(jax.pmap, devices=devices)
    def add_gradients(left, right):
        return jax.tree_util.tree_map(jnp.add, left, right)

    @partial(jax.pmap, devices=devices)
    def apply_gradients(state, gradients, rng):
        updates, opt_state = optimizer.update(gradients, state.opt_state, state.params)
        return state.replace(params=optax.apply_updates(state.params, updates), opt_state=opt_state,
                             rng=rng, step=state.step + 1), optax.global_norm(gradients)

    def step(state, batch):
        if len(batch) != len(module.species_names):
            raise ValueError("Pretraining batch must contain every species")
        next_rng, dropout_rng, ranking_rng = split_keys(state.rng)
        gradients, species_losses = None, []
        for code, record in enumerate(batch):
            values, mask = record["values"], record["valid_mask"]
            size = values.shape[0]
            if size < 1 or size % count or mask.shape != (size,):
                raise ValueError("Per-species padded batch size must divide local device count")
            loss, species_gradients = reconstruction_gradient(state.params,
                values.reshape((count, size // count, values.shape[1])),
                mask.reshape((count, size // count)), dropout_rng, code)
            gradients = species_gradients if gradients is None else add_gradients(gradients, species_gradients)
            species_losses.append(loss)
        (regularization, metrics), regularizer_gradients = regularizer_gradient(
            state.params, proteins, dropout_rng, ranking_rng)
        gradients = add_gradients(gradients, regularizer_gradients)
        state, norm = apply_gradients(state, gradients, next_rng)
        species_losses = jnp.stack(species_losses, axis=-1)
        metrics = {**metrics, "species_losses": species_losses,
                   "loss": jnp.sum(species_losses, axis=-1) + regularization, "gradient_norm": norm}
        return state, unreplicate_state(metrics)

    return step

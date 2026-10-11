"""Global-batch label-free objectives with sharded encoders and replicated banks."""

from functools import partial

from flax import struct
import jax
import jax.numpy as jnp
import optax

from jax_saturn.losses.core import normalize
from .metric import replicate_state, unreplicate_state, validate_devices


@struct.dataclass
class LabelReplicaState:
    params: object
    opt_state: object
    rng: jax.Array
    step: jax.Array


def replicate_training_state(state, devices):
    # Selection statistics and adaptive weights retain host double precision.
    core = LabelReplicaState(state.params, state.opt_state, state.rng, state.step)
    return replicate_state(core, devices)


def make_label_pmap(module, optimizer, objective, graphs, bandwidth, config, devices):
    from jax_saturn.training.label_agnostic import label_losses_from_embeddings

    devices = validate_devices(devices)
    count = len(devices)

    def shard(batch):
        size = batch["macrogenes"].shape[0]
        if size < 1 or size % count or any(value.shape[0] != size for value in batch.values()):
            raise ValueError("Label-free global batch must divide local device count")
        return {key: value.reshape((count, size // count) + value.shape[1:]) for key, value in batch.items()}

    def losses(params, rng, batch, bank, graph_arrays):
        _, key = jax.random.split(rng)
        key = jax.random.fold_in(key, jax.lax.axis_index("data"))
        local = normalize(module.apply({"params": params}, batch["macrogenes"], train=True,
                                       rngs={"dropout": key}))
        output = jax.lax.all_gather(local, "data", axis=0, tiled=True)
        # In particular MMD and OT must compare species across the entire batch.
        global_batch = {name: jax.lax.all_gather(batch[name], "data", axis=0, tiled=True)
                        for name in ("global_indices", "species_codes", "valid_mask")}
        return label_losses_from_embeddings(output, global_batch, bank, graph_arrays, objective, bandwidth, config)

    @partial(jax.pmap, axis_name="data", devices=devices, in_axes=(0, 0, None, None, None))
    def update(state, batch, bank, weight, graph_arrays):
        def total(params):
            alignment, preservation, local, coverage, local_coverage = losses(params, state.rng, batch, bank, graph_arrays)
            total = alignment + weight * preservation + config.local_graph_weight * local
            return total, {"alignment_loss": alignment, "preservation_loss": preservation,
                           "local_graph_loss": local, "coverage": coverage, "local_coverage": local_coverage}
        (loss, metrics), gradients = jax.value_and_grad(total, has_aux=True)(state.params)
        gradients = jax.lax.pmean(gradients, "data")
        updates, opt_state = optimizer.update(gradients, state.opt_state, state.params)
        rng, _ = jax.random.split(state.rng)
        return state.replace(params=optax.apply_updates(state.params, updates), opt_state=opt_state,
                             rng=rng, step=state.step + 1), {**metrics, "metric_loss": loss,
                                                           "gradient_norm": optax.global_norm(gradients)}

    @partial(jax.pmap, axis_name="data", devices=devices,
             in_axes=(0, 0, None, None, None), static_broadcasted_argnums=(4,))
    def calibration(state, batch, bank, graph_arrays, position):
        gradients = jax.grad(lambda params: losses(params, state.rng, batch, bank, graph_arrays)[position])(state.params)
        return jax.lax.pmean(gradients, "data")

    def step(state, batch, bank, weight):
        state, metrics = update(state, shard(batch), bank, jnp.float32(weight), graphs)
        return state, unreplicate_state(metrics)

    def calibrate(state, batch, bank):
        return [unreplicate_state(calibration(state, shard(batch), bank, graphs, position)) for position in (0, 1)]

    return step, calibrate

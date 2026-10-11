"""Single-host data-parallel baseline updates with global triplet semantics."""

from functools import partial

import jax
import jax.numpy as jnp
import optax

from jax_saturn.losses.core import normalize, triplet_margin_loss


def validate_devices(devices, *, batch_size=None):
    devices = tuple(devices)
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("Distributed training needs distinct local devices")
    if jax.process_count() != 1 or any(device.process_index != jax.process_index() for device in devices):
        raise ValueError("Only single-host distributed training is supported")
    if len({device.platform for device in devices}) != 1:
        raise ValueError("Distributed devices must use the same platform")
    if batch_size is not None and (batch_size < 1 or batch_size % len(devices)):
        raise ValueError("Global batch size must be positive and divisible by local device count")
    return devices


def replicate_state(state, devices):
    return jax.device_put_replicated(state, validate_devices(devices))


def unreplicate_state(state):
    return jax.tree_util.tree_map(lambda value: value[0], state)


def make_metric_pmap(module, optimizer, devices, *, margin=.2):
    """Return preview/update functions consuming replicated train state.

    The host mines triplets against the global preview. Each device encodes a
    shard, then gathers embeddings so positives/negatives may cross shards.
    Differentiating that gather and averaging replica gradients yields the
    same global triplet objective, including its active-only denominator.
    """
    devices = validate_devices(devices)
    count = len(devices)

    def shard(values):
        if values.shape[0] < 1 or values.shape[0] % count:
            raise ValueError("Global batch size must be divisible by local device count")
        return values.reshape((count, values.shape[0] // count) + values.shape[1:])

    def encode(params, rng, values):
        _, dropout_key = jax.random.split(rng)
        dropout_key = jax.random.fold_in(dropout_key, jax.lax.axis_index("data"))
        return module.apply({"params": params}, values, train=True, rngs={"dropout": dropout_key})

    @partial(jax.pmap, axis_name="data", devices=devices)
    def preview_shards(state, values):
        return encode(state.params, state.rng, values)

    @partial(jax.pmap, axis_name="data", devices=devices, in_axes=(0, 0, None, None))
    def update_shards(state, values, indices, valid_mask):
        def objective(params):
            local = encode(params, state.rng, values)
            output = jax.lax.all_gather(local, "data", axis=0, tiled=True)
            return triplet_margin_loss(normalize(output), indices, margin=margin, valid_mask=valid_mask)
        loss, gradients = jax.value_and_grad(objective)(state.params)
        gradients = jax.lax.pmean(gradients, "data")
        updates, opt_state = optimizer.update(gradients, state.opt_state, state.params)
        rng, _ = jax.random.split(state.rng)
        state = state.replace(params=optax.apply_updates(state.params, updates), opt_state=opt_state,
                              rng=rng, step=state.step + 1)
        return state, {"loss": loss, "gradient_norm": optax.global_norm(gradients)}

    def preview(state, values):
        output = preview_shards(state, shard(values))
        return output.reshape((-1, output.shape[-1]))

    def step(state, values, indices, valid_mask):
        state, metrics = update_shards(state, shard(values), indices, valid_mask)
        return state, unreplicate_state(metrics)

    return preview, step

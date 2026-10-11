"""Shape-checked reference state-dict mapping, isolated from model execution."""

import jax.numpy as jnp
import numpy as np
from flax.core import freeze
from flax.traverse_util import flatten_dict, unflatten_dict


def _block_key(path, source):
    index = int(path[0].removeprefix("block_"))
    layer = {"dense": 0, "layer_norm": 1}[path[1]]
    field = "bias" if path[2] == "bias" else "weight"
    return f"{source}.{index}.{layer}.{field}"


def _source_key(path):
    if path == ("macrogene", "log_gene_to_macrogene"):
        return "p_weights"
    if path[0] == "cl_layer_norm":
        return f"cl_layer_norm.{'weight' if path[1] == 'scale' else 'bias'}"
    if path[0] in {"encoder", "p_weights_embeddings"}:
        return _block_key(path[1:], path[0])
    if path[:2] == ("decoder", "px_decoder"):
        return _block_key(path[2:], "px_decoder")
    if path[:2] == ("decoder", "cl_scale_decoder"):
        return _block_key(path[2:], "cl_scale_decoder").replace("cl_scale_decoder.0.", "cl_scale_decoder.", 1)
    if path[:2] == ("decoder", "px_dropout_decoders"):
        field = "weight" if path[-1] == "kernel" else "bias"
        return f"px_dropout_decoders.{path[2]}.0.{field}"
    if path[:2] == ("decoder", "px_rs"):
        return f"px_rs.{path[2]}"
    raise ValueError(f"Unknown Flax parameter path: {path}")


def torch_to_flax(state_dict, template_params, *, model_kind):
    """Map a non-VAE reference state dict onto an initialized Flax parameter tree.

    Metric conversion can consume a pretrain state dict, selecting only encoder
    and cl_layer_norm just as the reference label-free trainer does. Torch is
    not imported: tensor conversion is duck-typed and NumPy inputs also work.
    """
    if model_kind not in {"pretrain", "metric"}:
        raise ValueError("model_kind must be pretrain or metric")
    template = flatten_dict(template_params)
    mapped = {}
    used = set()
    for path, expected in template.items():
        source = _source_key(path)
        if source not in state_dict:
            raise ValueError(f"Reference checkpoint is missing {source}")
        value = state_dict[source]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        value = np.asarray(value, dtype=np.float32)
        if path[-1] == "kernel":
            value = value.T
        if value.shape != expected.shape:
            raise ValueError(f"{source}: expected shape {expected.shape} after mapping, received {value.shape}")
        if np.isnan(value).any() or np.isposinf(value).any() or (source != "p_weights" and np.isneginf(value).any()):
            raise ValueError(f"Reference parameter {source} has invalid nonfinite values")
        # CPU device_put may share a contiguous NumPy/Torch backing buffer.
        # Conversion must take a snapshot, independent of later reference updates.
        mapped[path] = jnp.asarray(value.copy())
        used.add(source)
    if model_kind == "pretrain":
        extra = set(state_dict) - used - {"expr_filler"}
    else:
        extra = {key for key in state_dict if key.startswith(("encoder.", "cl_layer_norm.", "fc_mu.", "fc_var."))} - used
    if extra:
        raise ValueError(f"Unexpected reference parameters: {sorted(extra)}")
    return freeze(unflatten_dict(mapped))


def export_pytorch_checkpoint(path, params):
    """Optional reference-shaped export; only this compatibility operation imports Torch."""
    from pathlib import Path
    import torch

    state = {}
    for parameter, value in flatten_dict(params).items():
        source = _source_key(parameter)
        value = np.asarray(value, dtype=np.float32)
        if parameter[-1] == "kernel":
            value = value.T
        state[source] = torch.from_numpy(value.copy())
    if "p_weights" in state:
        state["expr_filler"] = torch.zeros(state["p_weights"].shape[1])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)

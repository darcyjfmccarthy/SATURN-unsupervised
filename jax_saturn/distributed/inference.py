"""Cached local-device inference kernels with dynamic weights and padded rows."""

from functools import lru_cache

import jax
import numpy as np

from .metric import validate_devices


@lru_cache(maxsize=32)
def _metric_kernel(module, devices):
    return jax.pmap(lambda params, values: module.apply({"params": params}, values), devices=devices)


@lru_cache(maxsize=128)
def _pretrain_kernel(module, devices, code):
    def infer(params, values):
        output = module.apply({"params": params}, values, code)
        return output.embedding, output.macrogenes
    return jax.pmap(infer, devices=devices)


def _batches(kernel, params, values, batch_size, devices):
    for start in range(0, len(values), batch_size):
        selected = np.asarray(values[start:start + batch_size], dtype=np.float32)
        count = len(selected)
        padded = np.zeros((batch_size, selected.shape[1]), dtype=np.float32)
        padded[:count] = selected
        outputs = kernel(params, padded.reshape(len(devices), batch_size // len(devices), selected.shape[1]))
        yield jax.tree_util.tree_map(lambda output: np.asarray(output, dtype=np.float32).reshape(
            (batch_size,) + output.shape[2:])[:count], outputs)


def metric_embeddings(module, params, values, *, batch_size, devices):
    devices = validate_devices(devices, batch_size=batch_size)
    if np.ndim(values) != 2 or len(values) < 1:
        raise ValueError("Inference requires nonempty two-dimensional inputs")
    replicated = jax.device_put_replicated(params, devices)
    return np.concatenate(list(_batches(_metric_kernel(module, devices), replicated, values, batch_size, devices)))


def pretrain_embeddings(module, params, species_values, *, batch_size, devices):
    devices = validate_devices(devices, batch_size=batch_size)
    if len(species_values) != len(module.species_names) or any(np.ndim(values) != 2 or len(values) < 1 for values in species_values):
        raise ValueError("Inference requires nonempty inputs for every species")
    replicated = jax.device_put_replicated(params, devices)
    embeddings, macrogenes = [], []
    for code, values in enumerate(species_values):
        kernel = _pretrain_kernel(module, devices, code)
        for embedding, macro in _batches(kernel, replicated, values, batch_size, devices):
            embeddings.append(embedding)
            macrogenes.append(macro)
    return np.concatenate(embeddings), np.concatenate(macrogenes)

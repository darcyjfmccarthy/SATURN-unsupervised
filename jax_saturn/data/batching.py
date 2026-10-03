"""Deterministic fixed-shape host batches with reference equal-species sampling."""

import numpy as np


def equal_species_batches(values, batch_size, *, seed, epoch):
    if batch_size <= 0 or not values or any(len(matrix) == 0 for matrix in values):
        raise ValueError("Batches require positive size and nonempty species arrays")
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    num_species = len(values)
    max_cells = max(len(matrix) for matrix in values)
    order = rng.permutation(max_cells * num_species)
    offsets = np.cumsum([0] + [len(matrix) for matrix in values[:-1]])
    for start in range(0, len(order), batch_size):
        virtual = order[start:start + batch_size]
        batch = []
        for code, matrix in enumerate(values):
            indices = virtual[virtual % num_species == code] // num_species
            overflow = indices >= len(matrix)
            indices[overflow] = rng.integers(0, len(matrix), size=overflow.sum())
            count = len(indices)
            padded_values = np.zeros((batch_size, matrix.shape[1]), dtype=np.float32)
            padded_values[:count] = matrix[indices]
            global_index = np.zeros(batch_size, dtype=np.int64)
            global_index[:count] = indices + offsets[code]
            batch.append({"values": padded_values, "valid_mask": np.arange(batch_size) < count,
                          "global_index": global_index, "species_code": np.full(batch_size, code, dtype=np.int32)})
        yield tuple(batch)


def metric_batches(macrogenes, species_codes, batch_size, *, seed, epoch, labels=None, ref_labels=None):
    """One shuffled pass in global observation order; labels are baseline-only."""
    values = np.asarray(macrogenes, dtype=np.float32)
    species_codes = np.asarray(species_codes, dtype=np.int32)
    if values.ndim != 2 or len(values) == 0 or species_codes.shape != (len(values),) or batch_size < 1:
        raise ValueError("Invalid metric batch inputs")
    if (labels is None) != (ref_labels is None):
        raise ValueError("Labeled batches require both labels and ref_labels")
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    order = rng.permutation(len(values))
    for start in range(0, len(order), batch_size):
        selected = order[start:start + batch_size]
        count = len(selected)
        indices = np.zeros(batch_size, dtype=np.int64)
        indices[:count] = selected
        padded = np.zeros((batch_size, values.shape[1]), dtype=np.float32)
        padded[:count] = values[selected]
        batch = {"macrogenes": padded, "species_codes": species_codes[indices],
                 "global_indices": indices, "valid_mask": np.arange(batch_size) < count}
        if labels is not None:
            for key, array in (("labels", labels), ("ref_labels", ref_labels)):
                array = np.asarray(array, dtype=np.int32)
                if array.shape != (len(values),):
                    raise ValueError(f"Invalid {key} rows")
                batch[key] = array[indices]
        yield batch

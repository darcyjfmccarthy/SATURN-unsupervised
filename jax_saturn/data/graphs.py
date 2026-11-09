"""Label-free graphs and metrics, with an optional exhaustive GPU backend."""

from itertools import combinations

import numpy as np
from .neighbors import indices as nearest_indices


def normalize_numpy(values):
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)

def fused_teacher_view(embeddings, macrogenes):
    embeddings = normalize_numpy(embeddings)
    macrogenes = normalize_numpy(macrogenes)
    return np.concatenate((embeddings, macrogenes), axis=1) / np.sqrt(2.0)

def build_preservation_graph(
    embeddings,
    species,
    neighbor_k=15,
    negative_k=15,
    temperature=0.1,
    seed=0,
    *, search=None,
):
    """Build fixed same-species topology targets without cell labels."""
    embeddings = normalize_numpy(embeddings)
    species = np.asarray(species).astype(str)
    rng = np.random.default_rng(seed)
    n_cells = len(embeddings)
    candidates = np.full(
        (n_cells, neighbor_k + negative_k), -1, dtype=np.int64
    )
    teacher_probabilities = np.zeros(candidates.shape, dtype=np.float32)
    teacher_similarities = np.zeros(candidates.shape, dtype=np.float32)
    neighbors = np.full((n_cells, neighbor_k), -1, dtype=np.int64)

    for species_name in np.unique(species):
        global_indices = np.flatnonzero(species == species_name)
        if len(global_indices) < 2:
            continue
        local_k = min(neighbor_k, len(global_indices) - 1)
        local = embeddings[global_indices]
        local_neighbors = nearest_indices(local, local, local_k, search=search, same=True)
        global_neighbors = global_indices[local_neighbors]
        neighbors[global_indices, :local_k] = global_neighbors

        for local_anchor, global_anchor in enumerate(global_indices):
            near = global_neighbors[local_anchor]
            # Draw ranks from the same ordered pool as the reference. Mapping
            # around its small exclusion set preserves rng.choice's seeded
            # draws without rebuilding an O(species_size) pool for every cell.
            excluded = np.unique(np.concatenate(([local_anchor], local_neighbors[local_anchor])))
            count = len(global_indices) - len(excluded)
            if count:
                ranks = rng.choice(count, size=min(negative_k, count), replace=False)
                local_negative = ranks + np.searchsorted(excluded - np.arange(len(excluded)), ranks, side="right")
                negatives = global_indices[local_negative]
            else:
                negatives = np.empty(0, dtype=np.int64)
            row = np.concatenate((near, negatives))
            candidates[global_anchor, : len(row)] = row
            similarities = embeddings[row] @ embeddings[global_anchor]
            logits = similarities / temperature
            logits -= logits.max()
            probabilities = np.exp(logits)
            teacher_probabilities[global_anchor, : len(row)] = (
                probabilities / probabilities.sum()
            )
            teacher_similarities[global_anchor, : len(row)] = similarities

    return (
        candidates,
        teacher_probabilities,
        teacher_similarities,
        neighbors,
    )

def build_cross_species_positives(
    embeddings,
    macrogenes,
    species,
    candidate_k=20,
    positives_per_species=3,
    *, search=None,
):
    """Find reciprocal cross-species neighbours in a fused teacher view."""
    view = fused_teacher_view(embeddings, macrogenes)
    species = np.asarray(species).astype(str)
    species_names = np.unique(species)
    species_to_code = {
        species_name: code for code, species_name in enumerate(species_names)
    }
    positives = np.full(
        (len(view), len(species_names), positives_per_species),
        -1,
        dtype=np.int64,
    )

    for species_a, species_b in combinations(species_names, 2):
        idx_a = np.flatnonzero(species == species_a)
        idx_b = np.flatnonzero(species == species_b)
        k_ab = min(candidate_k, len(idx_b))
        k_ba = min(candidate_k, len(idx_a))
        ab = nearest_indices(view[idx_a], view[idx_b], k_ab, search=search)
        ba = nearest_indices(view[idx_b], view[idx_a], k_ba, search=search)
        ba_sets = [set(row.tolist()) for row in ba]
        reciprocal_a = [[] for _ in idx_a]
        reciprocal_b = [[] for _ in idx_b]
        for local_a, row in enumerate(ab):
            for local_b in row:
                if local_a in ba_sets[local_b]:
                    reciprocal_a[local_a].append(int(idx_b[local_b]))
                    reciprocal_b[local_b].append(int(idx_a[local_a]))

        code_a = species_to_code[species_a]
        code_b = species_to_code[species_b]
        for local_a, values in enumerate(reciprocal_a):
            values = values[:positives_per_species]
            positives[idx_a[local_a], code_b, : len(values)] = values
        for local_b, values in enumerate(reciprocal_b):
            values = values[:positives_per_species]
            positives[idx_b[local_b], code_a, : len(values)] = values

    return positives, species_names

def estimate_mmd_bandwidth(embeddings, seed=0, max_cells=2048):
    embeddings = normalize_numpy(embeddings)
    rng = np.random.default_rng(seed)
    if len(embeddings) > max_cells:
        embeddings = embeddings[
            rng.choice(len(embeddings), max_cells, replace=False)
        ]
    squared_distances = np.maximum(
        2.0 - 2.0 * embeddings @ embeddings.T, 0.0
    )
    upper = squared_distances[np.triu_indices(len(embeddings), k=1)]
    positive = upper[upper > 0]
    return float(np.median(positive)) if len(positive) else 1.0

def joint_neighbors(embeddings, k=15, *, search=None):
    embeddings = normalize_numpy(embeddings)
    local_k = min(k, max(len(embeddings) - 1, 0))
    if local_k == 0:
        return np.empty((len(embeddings), 0), np.int64)
    return nearest_indices(embeddings, embeddings, local_k, search=search, same=True)

def species_mixing_fraction(embeddings, species, k=15, *, search=None):
    species = np.asarray(species).astype(str)
    neighbors = joint_neighbors(embeddings, k, search=search)
    return float(np.mean(species[neighbors] != species[:, None]))

def topology_recall_at_50(
    embeddings,
    species,
    teacher_neighbors,
    student_k=50,
    *, search=None,
):
    embeddings = normalize_numpy(embeddings)
    species = np.asarray(species).astype(str)
    recalls = []
    for species_name in np.unique(species):
        global_indices = np.flatnonzero(species == species_name)
        if len(global_indices) < 2:
            continue
        local_k = min(student_k, len(global_indices) - 1)
        local = embeddings[global_indices]
        student_local = nearest_indices(local, local, local_k, search=search, same=True)
        student_global = global_indices[student_local]
        for start in range(0, len(global_indices), 4096):
            teacher = np.sort(teacher_neighbors[global_indices[start:start + 4096]], axis=1)
            student = student_global[start:start + 4096]
            # Set semantics also apply to externally supplied duplicate IDs.
            valid = teacher >= 0
            valid[:, 1:] &= teacher[:, 1:] != teacher[:, :-1]
            common = np.any(teacher[:, :, None] == student[:, None, :], axis=2) & valid
            recalls.extend((common.sum(axis=1) / np.maximum(valid.sum(axis=1), 1)).tolist())
    return float(np.mean(recalls)) if recalls else float("nan")

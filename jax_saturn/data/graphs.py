"""CPU-only label-free graph construction, copied from the PyTorch reference helpers.

Source: label_agnostic/objectives.py and label_agnostic/metrics.py.
These functions deliberately retain reference KNN and sampling behavior.
"""

from itertools import combinations

import numpy as np
from sklearn.neighbors import NearestNeighbors


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
        local_neighbors = NearestNeighbors(
            n_neighbors=local_k + 1, metric="cosine"
        ).fit(embeddings[global_indices]).kneighbors(
            embeddings[global_indices], return_distance=False
        )[:, 1:]
        global_neighbors = global_indices[local_neighbors]
        neighbors[global_indices, :local_k] = global_neighbors

        for local_anchor, global_anchor in enumerate(global_indices):
            near = global_neighbors[local_anchor]
            excluded = np.concatenate(([global_anchor], near))
            negative_pool = global_indices[
                ~np.isin(global_indices, excluded, assume_unique=False)
            ]
            if len(negative_pool):
                negatives = rng.choice(
                    negative_pool,
                    size=min(negative_k, len(negative_pool)),
                    replace=False,
                )
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
        ab = NearestNeighbors(
            n_neighbors=k_ab, metric="cosine"
        ).fit(view[idx_b]).kneighbors(view[idx_a], return_distance=False)
        ba = NearestNeighbors(
            n_neighbors=k_ba, metric="cosine"
        ).fit(view[idx_a]).kneighbors(view[idx_b], return_distance=False)
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

def joint_neighbors(embeddings, k=15):
    embeddings = normalize_numpy(embeddings)
    local_k = min(k, len(embeddings) - 1)
    return NearestNeighbors(
        n_neighbors=local_k + 1, metric="cosine"
    ).fit(embeddings).kneighbors(
        embeddings, return_distance=False
    )[:, 1:]

def species_mixing_fraction(embeddings, species, k=15):
    species = np.asarray(species).astype(str)
    neighbors = joint_neighbors(embeddings, k)
    return float(np.mean(species[neighbors] != species[:, None]))

def topology_recall_at_50(
    embeddings,
    species,
    teacher_neighbors,
    student_k=50,
):
    embeddings = normalize_numpy(embeddings)
    species = np.asarray(species).astype(str)
    recalls = []
    for species_name in np.unique(species):
        global_indices = np.flatnonzero(species == species_name)
        if len(global_indices) < 2:
            continue
        local_k = min(student_k, len(global_indices) - 1)
        student_local = NearestNeighbors(
            n_neighbors=local_k + 1, metric="cosine"
        ).fit(embeddings[global_indices]).kneighbors(
            embeddings[global_indices], return_distance=False
        )[:, 1:]
        student_global = global_indices[student_local]
        for local_row, global_row in enumerate(global_indices):
            teacher = set(
                teacher_neighbors[global_row][
                    teacher_neighbors[global_row] >= 0
                ].tolist()
            )
            student = set(student_global[local_row].tolist())
            recalls.append(len(teacher & student) / max(len(teacher), 1))
    return float(np.mean(recalls)) if recalls else float("nan")

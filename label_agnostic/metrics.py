"""Deterministic metrics shared by all benchmark trials."""

import numpy as np
import torch
import torch.nn.functional as F

from .objectives import normalize_numpy


# Lazy GPU imports live in NeighborSearch; default metrics remain CPU-only.
from jax_saturn.data.graphs import joint_neighbors, species_mixing_fraction, topology_recall_at_50
from jax_saturn.data.neighbors import indices as nearest_indices


def embedding_benchmark_metrics(embeddings, species, labels, k=15, *, search=None, neighbors=None):
    embeddings = normalize_numpy(embeddings)
    species = np.asarray(species).astype(str)
    labels = np.asarray(labels).astype(str)
    if neighbors is None:
        neighbors = joint_neighbors(embeddings, k, search=search)
    neighbor_species = species[neighbors]
    neighbor_labels = labels[neighbors]
    mixing_per_cell = np.mean(
        neighbor_species != species[:, None], axis=1
    )
    label_per_cell = np.mean(
        neighbor_labels == labels[:, None], axis=1
    )
    species_values = np.unique(species)

    within_scores = np.full(len(embeddings), np.nan, dtype=np.float32)
    for species_name in species_values:
        global_indices = np.flatnonzero(species == species_name)
        if len(global_indices) < 2:
            continue
        local_k = min(k, len(global_indices) - 1)
        local = embeddings[global_indices]
        local_neighbors = nearest_indices(local, local, local_k, search=search, same=True)
        global_neighbors = global_indices[local_neighbors]
        within_scores[global_indices] = np.mean(
            labels[global_neighbors] == labels[global_indices, None],
            axis=1,
        )

    # Float64 proportions retain the original iLISI definition and reduction.
    inverse_simpson = np.empty(len(embeddings), dtype=np.float64)
    for start in range(0, len(embeddings), 4096):
        rows = neighbor_species[start:start + 4096]
        probabilities = np.mean(rows[:, :, None] == species_values[None, None, :], axis=1)
        inverse_simpson[start:start + len(rows)] = 1.0 / np.sum(probabilities ** 2, axis=1)
    max_lisi = min(len(species_values), neighbors.shape[1])
    normalized_lisi = (
        (np.asarray(inverse_simpson) - 1.0) / (max_lisi - 1.0)
        if max_lisi > 1
        else np.zeros(len(embeddings))
    )

    return {
        "species_mixing_fraction": float(mixing_per_cell.mean()),
        "label_same_neighbor_fraction": float(label_per_cell.mean()),
        "species_macro_mixing_fraction": float(
            np.mean(
                [
                    mixing_per_cell[species == value].mean()
                    for value in species_values
                ]
            )
        ),
        "species_macro_label_same_neighbor_fraction": float(
            np.mean(
                [
                    label_per_cell[species == value].mean()
                    for value in species_values
                ]
            )
        ),
        "within_species_label_same_neighbor_fraction": float(
            np.nanmean(within_scores)
        ),
        "normalized_species_ilisi": float(np.mean(normalized_lisi)),
    }


def fixed_triplet_margin_loss(embeddings, triplets, margin=0.2):
    embeddings = F.normalize(
        torch.as_tensor(np.asarray(embeddings), dtype=torch.float32),
        dim=1,
    )
    anchor, positive, negative = [
        torch.as_tensor(values, dtype=torch.long) for values in triplets
    ]
    positive_similarity = torch.sum(
        embeddings[anchor] * embeddings[positive], dim=1
    )
    negative_similarity = torch.sum(
        embeddings[anchor] * embeddings[negative], dim=1
    )
    losses = F.relu(
        negative_similarity - positive_similarity + margin
    )
    return {
        "fixed_triplet_margin_loss": float(losses.mean()),
        "fixed_triplet_active_fraction": float(
            (losses > 0).float().mean()
        ),
        "fixed_triplet_ap_similarity": float(
            positive_similarity.mean()
        ),
        "fixed_triplet_an_similarity": float(
            negative_similarity.mean()
        ),
        "fixed_triplet_count": int(len(losses)),
    }


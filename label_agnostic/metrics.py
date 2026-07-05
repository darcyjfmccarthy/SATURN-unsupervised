"""Deterministic metrics shared by all benchmark trials."""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

from .objectives import normalize_numpy


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


def embedding_benchmark_metrics(embeddings, species, labels, k=15):
    embeddings = normalize_numpy(embeddings)
    species = np.asarray(species).astype(str)
    labels = np.asarray(labels).astype(str)
    neighbors = joint_neighbors(embeddings, k)
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
        local_neighbors = NearestNeighbors(
            n_neighbors=local_k + 1, metric="cosine"
        ).fit(embeddings[global_indices]).kneighbors(
            embeddings[global_indices], return_distance=False
        )[:, 1:]
        global_neighbors = global_indices[local_neighbors]
        within_scores[global_indices] = np.mean(
            labels[global_neighbors] == labels[global_indices, None],
            axis=1,
        )

    inverse_simpson = []
    for row in neighbor_species:
        probabilities = np.asarray(
            [(row == value).mean() for value in species_values],
            dtype=np.float64,
        )
        inverse_simpson.append(1.0 / np.sum(probabilities**2))
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


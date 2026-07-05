"""Deterministic evaluation metrics for the label-agnostic benchmark."""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

import distances
import miners


def _joint_neighbors(embeddings, k):
    local_k = min(k, len(embeddings) - 1)
    return NearestNeighbors(n_neighbors=local_k + 1, metric="cosine").fit(
        embeddings
    ).kneighbors(embeddings, return_distance=False)[:, 1:]


def embedding_benchmark_metrics(
    embeddings,
    species,
    labels,
    k=15,
    frozen_neighbors=None,
):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    species = np.asarray(species).astype(str)
    labels = np.asarray(labels).astype(str)
    neighbors = _joint_neighbors(embeddings, k)
    neighbor_species = species[neighbors]
    neighbor_labels = labels[neighbors]
    mixing_per_cell = np.mean(neighbor_species != species[:, None], axis=1)
    label_per_cell = np.mean(neighbor_labels == labels[:, None], axis=1)

    species_values = np.unique(species)
    mixing_macro = np.mean(
        [mixing_per_cell[species == value].mean() for value in species_values]
    )
    label_macro = np.mean(
        [label_per_cell[species == value].mean() for value in species_values]
    )

    within_scores = np.zeros(len(embeddings), dtype=np.float32)
    within_neighbors = np.full((len(embeddings), k), -1, dtype=np.int64)
    for value in species_values:
        global_indices = np.flatnonzero(species == value)
        if len(global_indices) < 2:
            within_scores[global_indices] = np.nan
            continue
        local_k = min(k, len(global_indices) - 1)
        local_neighbors = NearestNeighbors(
            n_neighbors=local_k + 1, metric="cosine"
        ).fit(embeddings[global_indices]).kneighbors(
            embeddings[global_indices], return_distance=False
        )[:, 1:]
        global_neighbors = global_indices[local_neighbors]
        within_neighbors[global_indices, :local_k] = global_neighbors
        within_scores[global_indices] = np.mean(
            labels[global_neighbors] == labels[global_indices, None], axis=1
        )

    inverse_simpson = []
    for row in neighbor_species:
        probabilities = np.asarray(
            [(row == value).mean() for value in species_values], dtype=np.float64
        )
        inverse_simpson.append(1.0 / np.sum(probabilities**2))
    max_lisi = min(len(species_values), neighbors.shape[1])
    normalized_lisi = (
        (np.asarray(inverse_simpson) - 1.0) / (max_lisi - 1.0)
        if max_lisi > 1
        else np.zeros(len(embeddings))
    )

    result = {
        "species_mixing_fraction": float(mixing_per_cell.mean()),
        "label_same_neighbor_fraction": float(label_per_cell.mean()),
        "species_macro_mixing_fraction": float(mixing_macro),
        "species_macro_label_same_neighbor_fraction": float(label_macro),
        "within_species_label_same_neighbor_fraction": float(np.nanmean(within_scores)),
        "normalized_species_ilisi": float(np.mean(normalized_lisi)),
    }

    if frozen_neighbors is not None:
        overlaps = []
        frozen_neighbors = np.asarray(frozen_neighbors)
        for row in range(len(embeddings)):
            current = set(within_neighbors[row][within_neighbors[row] >= 0].tolist())
            frozen = set(frozen_neighbors[row][frozen_neighbors[row] >= 0].tolist())
            denominator = max(len(frozen), 1)
            overlaps.append(len(current & frozen) / denominator)
        result["frozen_neighbor_overlap"] = float(np.mean(overlaps))
    else:
        result["frozen_neighbor_overlap"] = np.nan
    return result


def build_fixed_triplet_manifest(
    embeddings,
    labels,
    species,
    seed=0,
    batch_size=512,
    margin=0.2,
):
    """Freeze triplets emitted by SATURN's original label-aware miner."""
    embeddings = torch.as_tensor(np.asarray(embeddings), dtype=torch.float32)
    label_codes = torch.as_tensor(pd.factorize(np.asarray(labels).astype(str))[0])
    species_codes = torch.as_tensor(pd.factorize(np.asarray(species).astype(str))[0])
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(embeddings), generator=generator)
    torch.manual_seed(seed)
    miner = miners.TripletMarginMiner(
        margin=margin,
        distance=distances.CosineSimilarity(),
        type_of_triplets="semihard",
        miner_type="cross_species",
    )
    output = [[], [], []]
    for start in range(0, len(permutation), batch_size):
        batch_indices = permutation[start : start + batch_size]
        if len(torch.unique(species_codes[batch_indices])) < 2:
            continue
        local = miner(
            F.normalize(embeddings[batch_indices], dim=1),
            label_codes[batch_indices],
            species_codes[batch_indices],
            mnn=True,
        )
        for position in range(3):
            if len(local[position]):
                output[position].append(batch_indices[local[position]])
    if not output[0]:
        raise RuntimeError("The label-aware miner did not produce evaluation triplets")
    return tuple(torch.cat(values).numpy().astype(np.int64) for values in output)


def fixed_triplet_margin_loss(embeddings, triplets, margin=0.2):
    embeddings = F.normalize(
        torch.as_tensor(np.asarray(embeddings), dtype=torch.float32), dim=1
    )
    anchor, positive, negative = [torch.as_tensor(values) for values in triplets]
    positive_similarity = torch.sum(embeddings[anchor] * embeddings[positive], dim=1)
    negative_similarity = torch.sum(embeddings[anchor] * embeddings[negative], dim=1)
    return float(F.relu(negative_similarity - positive_similarity + margin).mean())

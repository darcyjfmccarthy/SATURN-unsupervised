"""Label-free fine-tuning objectives and graph construction for SATURN."""

from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors


def _normalize_numpy(values):
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def build_within_species_knn(embeddings, species, k=15):
    """Return global indices of each cell's nearest same-species neighbors."""
    embeddings = _normalize_numpy(embeddings)
    species = np.asarray(species)
    result = np.full((len(embeddings), k), -1, dtype=np.int64)
    for species_name in np.unique(species):
        global_indices = np.flatnonzero(species == species_name)
        if len(global_indices) < 2:
            continue
        local_k = min(k, len(global_indices) - 1)
        neighbors = NearestNeighbors(
            n_neighbors=local_k + 1, metric="cosine"
        ).fit(embeddings[global_indices])
        local_indices = neighbors.kneighbors(
            embeddings[global_indices], return_distance=False
        )[:, 1:]
        result[global_indices, :local_k] = global_indices[local_indices]
    return result


def teacher_neighbor_probabilities(embeddings, neighbor_indices, temperature=0.1):
    """Soft neighborhood targets over a fixed global neighbor graph."""
    embeddings = _normalize_numpy(embeddings)
    probabilities = np.zeros(neighbor_indices.shape, dtype=np.float32)
    for row in range(len(embeddings)):
        valid = neighbor_indices[row] >= 0
        if not np.any(valid):
            continue
        similarities = embeddings[neighbor_indices[row, valid]] @ embeddings[row]
        logits = similarities / temperature
        logits -= logits.max()
        weights = np.exp(logits)
        probabilities[row, valid] = weights / weights.sum()
    return probabilities


def build_frozen_cross_species_neighbors(
    teacher_embeddings,
    macrogenes,
    species,
    candidate_k=10,
    positives_per_anchor=3,
):
    """Build reciprocal cross-species neighbors and re-rank them in two views."""
    teacher_embeddings = _normalize_numpy(teacher_embeddings)
    macrogenes = _normalize_numpy(macrogenes)
    species = np.asarray(species)
    candidates = [[] for _ in range(len(species))]

    for species_a, species_b in combinations(np.unique(species), 2):
        idx_a = np.flatnonzero(species == species_a)
        idx_b = np.flatnonzero(species == species_b)
        k_ab = min(candidate_k, len(idx_b))
        k_ba = min(candidate_k, len(idx_a))
        if k_ab == 0 or k_ba == 0:
            continue

        ab = NearestNeighbors(n_neighbors=k_ab, metric="cosine").fit(
            teacher_embeddings[idx_b]
        ).kneighbors(teacher_embeddings[idx_a], return_distance=False)
        ba = NearestNeighbors(n_neighbors=k_ba, metric="cosine").fit(
            teacher_embeddings[idx_a]
        ).kneighbors(teacher_embeddings[idx_b], return_distance=False)
        ba_sets = [set(row.tolist()) for row in ba]

        for local_a, neighbors_b in enumerate(ab):
            for local_b in neighbors_b:
                if local_a not in ba_sets[local_b]:
                    continue
                global_a = idx_a[local_a]
                global_b = idx_b[local_b]
                score = 0.5 * float(
                    teacher_embeddings[global_a] @ teacher_embeddings[global_b]
                ) + 0.5 * float(macrogenes[global_a] @ macrogenes[global_b])
                candidates[global_a].append((global_b, score))
                candidates[global_b].append((global_a, score))

    positive_indices = np.full(
        (len(species), positives_per_anchor), -1, dtype=np.int64
    )
    positive_weights = np.zeros_like(positive_indices, dtype=np.float32)
    for anchor, anchor_candidates in enumerate(candidates):
        best_by_index = {}
        for positive, score in anchor_candidates:
            best_by_index[positive] = max(score, best_by_index.get(positive, -np.inf))
        ranked = sorted(best_by_index.items(), key=lambda item: item[1], reverse=True)
        ranked = ranked[:positives_per_anchor]
        if not ranked:
            continue
        scores = np.asarray([score for _, score in ranked], dtype=np.float32)
        weights = np.exp((scores - scores.max()) / 0.1)
        weights /= weights.sum()
        positive_indices[anchor, : len(ranked)] = [idx for idx, _ in ranked]
        positive_weights[anchor, : len(ranked)] = weights
    return positive_indices, positive_weights


def estimate_mmd_bandwidth(embeddings, seed=0, max_cells=2048):
    embeddings = _normalize_numpy(embeddings)
    rng = np.random.default_rng(seed)
    if len(embeddings) > max_cells:
        embeddings = embeddings[rng.choice(len(embeddings), max_cells, replace=False)]
    squared_distances = np.maximum(
        2.0 - 2.0 * embeddings @ embeddings.T, 0.0
    )
    upper = squared_distances[np.triu_indices(len(embeddings), k=1)]
    positive = upper[upper > 0]
    return float(np.median(positive)) if len(positive) else 1.0


@torch.no_grad()
def embed_label_free_dataset(model, dataset, device, batch_size=1024):
    """Embed the full label-free dataset in stable global-index order."""
    was_training = model.training
    model.eval()
    output = [None] * len(dataset)
    with torch.no_grad():
        for species_name in dataset.species:
            values = dataset.xs[species_name]
            offset = dataset.offsets[species_name]
            for start in range(0, len(values), batch_size):
                batch = values[start : start + batch_size].to(device)
                embedded = F.normalize(model(batch, species_name), dim=1).cpu()
                for local_idx, row in enumerate(embedded):
                    output[offset + start + local_idx] = row
    if was_training:
        model.train()
    return torch.stack(output)


def frozen_topology_loss(
    embeddings,
    global_indices,
    embedding_bank,
    neighbor_indices,
    teacher_probabilities,
    temperature=0.1,
):
    """KL distillation over fixed same-species neighbors using a detached bank."""
    row_indices = global_indices.detach().cpu()
    neighbors = neighbor_indices[row_indices].to(embeddings.device)
    targets = teacher_probabilities[row_indices].to(embeddings.device)
    valid_rows = torch.all(neighbors >= 0, dim=1)
    if not torch.any(valid_rows):
        return embeddings.sum() * 0
    anchors = F.normalize(embeddings[valid_rows], dim=1)
    neighbors = neighbors[valid_rows]
    targets = targets[valid_rows]
    bank_values = F.normalize(embedding_bank.to(embeddings.device)[neighbors], dim=2)
    logits = torch.einsum("bd,bkd->bk", anchors, bank_values) / temperature
    return F.kl_div(F.log_softmax(logits, dim=1), targets, reduction="batchmean")


def online_cross_species_mnn_loss(embeddings, species_codes):
    """Attract reciprocal nearest cross-species pairs without explicit negatives."""
    embeddings = F.normalize(embeddings, dim=1)
    similarities = embeddings @ embeddings.T
    cross_species = species_codes[:, None] != species_codes[None, :]
    masked = similarities.masked_fill(~cross_species, -torch.inf)
    nearest = masked.argmax(dim=1)
    anchors = torch.arange(len(embeddings), device=embeddings.device)
    mutual = nearest[nearest] == anchors
    if not torch.any(mutual):
        return embeddings.sum() * 0, 0
    unique_pairs = mutual & (anchors < nearest)
    if not torch.any(unique_pairs):
        unique_pairs = mutual
    pair_anchors = anchors[unique_pairs]
    pair_positives = nearest[unique_pairs]
    loss = 1.0 - torch.sum(
        embeddings[pair_anchors] * embeddings[pair_positives], dim=1
    )
    return loss.mean(), int(mutual.sum().item())


def frozen_neighbor_attraction_loss(
    embeddings,
    global_indices,
    embedding_bank,
    positive_indices,
    positive_weights,
):
    row_indices = global_indices.detach().cpu()
    positives = positive_indices[row_indices].to(embeddings.device)
    weights = positive_weights[row_indices].to(embeddings.device)
    valid = positives >= 0
    if not torch.any(valid):
        return embeddings.sum() * 0, 0
    safe_positives = positives.clamp_min(0)
    anchor_values = F.normalize(embeddings, dim=1)
    positive_values = F.normalize(
        embedding_bank.to(embeddings.device)[safe_positives], dim=2
    )
    similarities = torch.einsum("bd,bkd->bk", anchor_values, positive_values)
    weighted = ((1.0 - similarities) * weights * valid).sum()
    denominator = (weights * valid).sum().clamp_min(1e-12)
    return weighted / denominator, int(torch.sum(torch.any(valid, dim=1)).item())


def multi_species_mmd(embeddings, species_codes, base_bandwidth):
    """Biased multi-kernel RBF MMD averaged across species pairs."""
    embeddings = F.normalize(embeddings, dim=1)
    losses = []
    bandwidths = [0.5 * base_bandwidth, base_bandwidth, 2.0 * base_bandwidth]

    def kernel(x, y):
        distances = torch.cdist(x, y).pow(2)
        return sum(torch.exp(-distances / max(bandwidth, 1e-6)) for bandwidth in bandwidths) / 3.0

    for species_a, species_b in combinations(torch.unique(species_codes).tolist(), 2):
        values_a = embeddings[species_codes == species_a]
        values_b = embeddings[species_codes == species_b]
        if len(values_a) < 2 or len(values_b) < 2:
            continue
        losses.append(
            kernel(values_a, values_a).mean()
            + kernel(values_b, values_b).mean()
            - 2.0 * kernel(values_a, values_b).mean()
        )
    return torch.stack(losses).mean() if losses else embeddings.sum() * 0


def partial_sinkhorn(cost, epsilon=0.05, transported_mass=0.8, iterations=50):
    """Entropy-regularized partial transport with capped uniform marginals."""
    if cost.ndim != 2 or cost.numel() == 0:
        raise ValueError("cost must be a non-empty matrix")
    n_rows, n_cols = cost.shape
    row_cap = torch.full((n_rows,), 1.0 / n_rows, device=cost.device, dtype=cost.dtype)
    col_cap = torch.full((n_cols,), 1.0 / n_cols, device=cost.device, dtype=cost.dtype)
    shifted = cost - cost.min()
    plan = torch.exp(-shifted / epsilon).clamp_min(1e-30)

    for _ in range(iterations):
        row_sums = plan.sum(dim=1).clamp_min(1e-30)
        plan = plan * torch.minimum(torch.ones_like(row_sums), row_cap / row_sums)[:, None]
        col_sums = plan.sum(dim=0).clamp_min(1e-30)
        plan = plan * torch.minimum(torch.ones_like(col_sums), col_cap / col_sums)[None, :]
        current_mass = plan.sum().clamp_min(1e-30)
        if current_mass >= transported_mass:
            plan = plan * (transported_mass / current_mass)

    # The kernel starts above the requested mass in normal use. This fallback is
    # only for extreme numerical underflow and retains a well-defined coupling.
    if plan.sum() < transported_mass * 0.999:
        uniform = torch.full_like(plan, transported_mass / plan.numel())
        plan = uniform
    return plan * (transported_mass / plan.sum().clamp_min(1e-30))


def partial_ot_alignment_loss(
    embeddings,
    teacher_embeddings,
    macrogenes,
    species_codes,
    epsilon=0.05,
    transported_mass=0.8,
    iterations=50,
):
    embeddings = F.normalize(embeddings, dim=1)
    teacher_embeddings = F.normalize(teacher_embeddings, dim=1)
    macrogenes = F.normalize(macrogenes, dim=1)
    losses = []
    for species_a, species_b in combinations(torch.unique(species_codes).tolist(), 2):
        mask_a = species_codes == species_a
        mask_b = species_codes == species_b
        if not torch.any(mask_a) or not torch.any(mask_b):
            continue
        with torch.no_grad():
            teacher_cost = 0.5 * (1.0 - teacher_embeddings[mask_a] @ teacher_embeddings[mask_b].T)
            macrogene_cost = 0.5 * (1.0 - macrogenes[mask_a] @ macrogenes[mask_b].T)
            plan = partial_sinkhorn(
                teacher_cost + macrogene_cost,
                epsilon=epsilon,
                transported_mass=transported_mass,
                iterations=iterations,
            )
        student_cost = 1.0 - embeddings[mask_a] @ embeddings[mask_b].T
        losses.append(torch.sum(plan * student_cost) / transported_mass)
    return torch.stack(losses).mean() if losses else embeddings.sum() * 0


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, coefficient):
        ctx.coefficient = coefficient
        return values.view_as(values)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.coefficient * gradient, None


def gradient_reverse(values, coefficient):
    return _GradientReversal.apply(values, coefficient)


class SpeciesDiscriminator(nn.Module):
    def __init__(self, embedding_dim, num_species):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_species),
        )

    def forward(self, embeddings, coefficient):
        return self.network(gradient_reverse(embeddings, coefficient))


def balanced_species_cross_entropy(logits, species_codes):
    losses = F.cross_entropy(logits, species_codes, reduction="none")
    species_losses = [losses[species_codes == code].mean() for code in torch.unique(species_codes)]
    return torch.stack(species_losses).mean()

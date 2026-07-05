"""Label-free alignment and topology-preservation objectives."""

from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F
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


def preservation_distillation_loss(
    embeddings,
    global_indices,
    embedding_bank,
    candidate_indices,
    teacher_probabilities,
    teacher_similarities,
    temperature=0.1,
):
    rows = global_indices.detach().cpu()
    candidates = candidate_indices[rows].to(embeddings.device)
    targets = teacher_probabilities[rows].to(embeddings.device)
    similarity_targets = teacher_similarities[rows].to(
        embeddings.device
    )
    valid = candidates >= 0
    valid_rows = torch.any(valid, dim=1)
    if not torch.any(valid_rows):
        return embeddings.sum() * 0
    candidates = candidates[valid_rows]
    targets = targets[valid_rows]
    similarity_targets = similarity_targets[valid_rows]
    valid = valid[valid_rows]
    anchors = F.normalize(embeddings[valid_rows], dim=1)
    safe_candidates = candidates.clamp_min(0)
    bank_values = F.normalize(embedding_bank[safe_candidates], dim=2)
    similarities = torch.einsum("bd,bkd->bk", anchors, bank_values)
    logits = similarities / temperature
    logits = logits.masked_fill(~valid, -torch.inf)
    log_probabilities = F.log_softmax(logits, dim=1)
    targets = targets.masked_fill(~valid, 0)
    distribution_loss = F.kl_div(
        log_probabilities, targets, reduction="batchmean"
    )
    distortion_loss = F.smooth_l1_loss(
        similarities[valid],
        similarity_targets[valid],
    )
    return distribution_loss + distortion_loss


def multi_positive_infonce_loss(
    embeddings,
    global_indices,
    embedding_bank,
    positive_indices,
    target_indices,
    global_to_local,
    temperature=0.1,
):
    """Multi-positive InfoNCE against complete target-species memory banks."""
    rows = global_indices.detach().cpu()
    positives = positive_indices[rows].to(embeddings.device)
    losses = []
    covered_rows = torch.zeros(
        len(embeddings), dtype=torch.bool, device=embeddings.device
    )
    anchors = F.normalize(embeddings, dim=1)

    for target_code, target_global in enumerate(target_indices):
        target_global = target_global.to(embeddings.device)
        target_bank = F.normalize(embedding_bank[target_global], dim=1)
        target_positives = positives[:, target_code]
        valid = target_positives >= 0
        active = torch.any(valid, dim=1)
        if not torch.any(active):
            continue
        active_logits = anchors[active] @ target_bank.T / temperature
        active_valid = valid[active]
        positive_local = global_to_local[target_code][
            target_positives[active].clamp_min(0)
        ].to(embeddings.device)
        positive_local = positive_local.masked_fill(
            ~active_valid, 0
        )
        if torch.any(positive_local[active_valid] < 0):
            raise RuntimeError(
                "A positive index does not belong to its target species"
            )
        positive_logits = active_logits.gather(1, positive_local)
        positive_logits = positive_logits.masked_fill(
            ~active_valid, -torch.inf
        )
        numerator = torch.logsumexp(positive_logits, dim=1)
        denominator = torch.logsumexp(active_logits, dim=1)
        losses.append(denominator - numerator)
        covered_rows[active] = True

    if not losses:
        return embeddings.sum() * 0, 0
    return torch.cat(losses).mean(), int(covered_rows.sum().item())


def within_species_graph_infonce_loss(
    embeddings,
    global_indices,
    species_codes,
    embedding_bank,
    teacher_neighbors,
    target_indices,
    global_to_local,
    positive_k=5,
    temperature=0.1,
):
    """Sharpen a frozen within-species neighbour graph without labels."""
    rows = global_indices.detach().cpu()
    positives = teacher_neighbors[rows, :positive_k].to(
        embeddings.device
    )
    anchors = F.normalize(embeddings, dim=1)
    losses = []
    covered = 0
    for species_code, species_global in enumerate(target_indices):
        active = species_codes == species_code
        if not torch.any(active):
            continue
        valid = positives[active] >= 0
        active_rows = torch.any(valid, dim=1)
        if not torch.any(active_rows):
            continue
        selected_anchors = anchors[active][active_rows]
        selected_valid = valid[active_rows]
        selected_positives = positives[active][active_rows]
        species_global = species_global.to(embeddings.device)
        species_bank = F.normalize(
            embedding_bank[species_global], dim=1
        )
        logits = selected_anchors @ species_bank.T / temperature
        selected_global = global_indices[active][active_rows]
        anchor_local = global_to_local[species_code][
            selected_global
        ].to(embeddings.device)
        logits[
            torch.arange(len(logits), device=embeddings.device),
            anchor_local,
        ] = -torch.inf
        positive_local = global_to_local[species_code][
            selected_positives.clamp_min(0)
        ].to(embeddings.device)
        positive_local = positive_local.masked_fill(
            ~selected_valid, 0
        )
        if torch.any(positive_local[selected_valid] < 0):
            raise RuntimeError(
                "A local positive does not belong to its species"
            )
        positive_logits = logits.gather(1, positive_local)
        positive_logits = positive_logits.masked_fill(
            ~selected_valid, -torch.inf
        )
        losses.append(
            torch.logsumexp(logits, dim=1)
            - torch.logsumexp(positive_logits, dim=1)
        )
        covered += int(active_rows.sum().item())
    if not losses:
        return embeddings.sum() * 0, 0
    return torch.cat(losses).mean(), covered


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


def multi_species_mmd(embeddings, species_codes, base_bandwidth):
    embeddings = F.normalize(embeddings, dim=1)
    bandwidths = (
        0.5 * base_bandwidth,
        base_bandwidth,
        2.0 * base_bandwidth,
    )

    def kernel(left, right):
        distances = torch.cdist(left, right).pow(2)
        return sum(
            torch.exp(-distances / max(bandwidth, 1e-6))
            for bandwidth in bandwidths
        ) / len(bandwidths)

    losses = []
    for code_a, code_b in combinations(
        torch.unique(species_codes).tolist(), 2
    ):
        values_a = embeddings[species_codes == code_a]
        values_b = embeddings[species_codes == code_b]
        if len(values_a) < 2 or len(values_b) < 2:
            continue
        losses.append(
            kernel(values_a, values_a).mean()
            + kernel(values_b, values_b).mean()
            - 2.0 * kernel(values_a, values_b).mean()
        )
    return torch.stack(losses).mean() if losses else embeddings.sum() * 0


def partial_sinkhorn(
    cost,
    epsilon=0.05,
    transported_mass=0.8,
    iterations=100,
):
    """KL projections onto partial-transport mass and marginal constraints."""
    if cost.ndim != 2 or cost.numel() == 0:
        raise ValueError("cost must be a non-empty matrix")
    if not 0 < transported_mass <= 1:
        raise ValueError("transported_mass must be in (0, 1]")
    n_rows, n_cols = cost.shape
    row_cap = torch.full(
        (n_rows,), 1.0 / n_rows, dtype=cost.dtype, device=cost.device
    )
    col_cap = torch.full(
        (n_cols,), 1.0 / n_cols, dtype=cost.dtype, device=cost.device
    )
    shifted = cost - cost.min()
    plan = torch.exp(-shifted / epsilon).clamp_min(1e-30)
    q_row = torch.ones_like(plan)
    q_col = torch.ones_like(plan)
    q_mass = torch.ones_like(plan)

    for _ in range(iterations):
        candidate = (plan * q_row).clamp_min(1e-30)
        factors = torch.minimum(
            torch.ones_like(row_cap),
            row_cap / candidate.sum(dim=1).clamp_min(1e-30),
        )
        projected = candidate * factors[:, None]
        q_row = (candidate / projected.clamp_min(1e-30)).clamp_max(1e30)
        plan = projected

        candidate = (plan * q_col).clamp_min(1e-30)
        factors = torch.minimum(
            torch.ones_like(col_cap),
            col_cap / candidate.sum(dim=0).clamp_min(1e-30),
        )
        projected = candidate * factors[None, :]
        q_col = (candidate / projected.clamp_min(1e-30)).clamp_max(1e30)
        plan = projected

        candidate = (plan * q_mass).clamp_min(1e-30)
        projected = candidate * (
            transported_mass / candidate.sum().clamp_min(1e-30)
        )
        q_mass = (candidate / projected.clamp_min(1e-30)).clamp_max(1e30)
        plan = projected

    return plan


def partial_ot_alignment_loss(
    embeddings,
    teacher_embeddings,
    macrogenes,
    species_codes,
    epsilon=0.05,
    transported_mass=0.8,
    iterations=100,
):
    embeddings = F.normalize(embeddings, dim=1)
    teacher_embeddings = F.normalize(teacher_embeddings, dim=1)
    macrogenes = F.normalize(macrogenes, dim=1)
    losses = []
    transported = []
    for code_a, code_b in combinations(
        torch.unique(species_codes).tolist(), 2
    ):
        mask_a = species_codes == code_a
        mask_b = species_codes == code_b
        if not torch.any(mask_a) or not torch.any(mask_b):
            continue
        with torch.no_grad():
            teacher_cost = 0.5 * (
                1.0
                - teacher_embeddings[mask_a] @ teacher_embeddings[mask_b].T
            )
            macrogene_cost = 0.5 * (
                1.0 - macrogenes[mask_a] @ macrogenes[mask_b].T
            )
            plan = partial_sinkhorn(
                teacher_cost + macrogene_cost,
                epsilon=epsilon,
                transported_mass=transported_mass,
                iterations=iterations,
            )
        student_cost = (
            1.0 - embeddings[mask_a] @ embeddings[mask_b].T
        )
        realized_mass = plan.sum().clamp_min(1e-12)
        losses.append(torch.sum(plan * student_cost) / realized_mass)
        transported.append(float(realized_mass.detach().cpu()))
    if not losses:
        return embeddings.sum() * 0, []
    return torch.stack(losses).mean(), transported

"""Phase-2 preservation, alignment, and diagnostic utilities."""

from collections import defaultdict
from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

from label_free_utils import _normalize_numpy, partial_sinkhorn


PHASE2_OBJECTIVES = {
    "bridge_frozen",
    "bridge_mmd",
    "low_mass_ot",
    "fused_gw",
    "anchor_curriculum",
}


def build_multiscale_preservation_artifacts(embeddings, species, seed=0):
    embeddings = _normalize_numpy(embeddings)
    species = np.asarray(species).astype(str)
    rng = np.random.default_rng(seed)
    n_cells = len(embeddings)
    neighbors_5 = np.full((n_cells, 5), -1, dtype=np.int64)
    neighbors_15 = np.full((n_cells, 15), -1, dtype=np.int64)
    neighbors_50 = np.full((n_cells, 50), -1, dtype=np.int64)
    preservation_pairs = np.full((n_cells, 6), -1, dtype=np.int64)

    for species_name in np.unique(species):
        global_indices = np.flatnonzero(species == species_name)
        if len(global_indices) < 2:
            continue
        max_rank = min(100, len(global_indices) - 1)
        local_neighbors = NearestNeighbors(
            n_neighbors=max_rank + 1, metric="cosine"
        ).fit(embeddings[global_indices]).kneighbors(
            embeddings[global_indices], return_distance=False
        )[:, 1:]
        global_neighbors = global_indices[local_neighbors]
        for k, output in ((5, neighbors_5), (15, neighbors_15), (50, neighbors_50)):
            available = min(k, global_neighbors.shape[1])
            output[global_indices, :available] = global_neighbors[:, :available]

        for local_anchor, global_anchor in enumerate(global_indices):
            ranked = global_neighbors[local_anchor]
            buckets = (ranked[:5], ranked[5:15], ranked[15:50])
            positives = [
                int(bucket[rng.integers(len(bucket))]) if len(bucket) else int(ranked[-1])
                for bucket in buckets
            ]
            far_pool = global_indices[
                ~np.isin(global_indices, np.concatenate(([global_anchor], ranked[:100])))
            ]
            if len(far_pool) == 0:
                far_pool = ranked[-min(3, len(ranked)) :]
            negatives = rng.choice(far_pool, size=3, replace=len(far_pool) < 3).astype(int)
            preservation_pairs[global_anchor] = positives + negatives.tolist()

    safe_pairs = np.maximum(preservation_pairs, 0)
    teacher_similarities = np.sum(
        embeddings[:, None, :] * embeddings[safe_pairs], axis=2
    ).astype(np.float32)
    teacher_similarities[preservation_pairs < 0] = 0
    return {
        "neighbors_5": neighbors_5,
        "neighbors_15": neighbors_15,
        "neighbors_50": neighbors_50,
        "preservation_pairs": preservation_pairs,
        "teacher_pair_similarities": teacher_similarities,
    }


def _pairwise_mutual_edges(view, idx_a, idx_b, k):
    k_ab = min(k, len(idx_b))
    k_ba = min(k, len(idx_a))
    ab = NearestNeighbors(n_neighbors=k_ab, metric="cosine").fit(
        view[idx_b]
    ).kneighbors(view[idx_a], return_distance=False)
    ba = NearestNeighbors(n_neighbors=k_ba, metric="cosine").fit(
        view[idx_a]
    ).kneighbors(view[idx_b], return_distance=False)
    ba_sets = [set(row.tolist()) for row in ba]
    return {
        (int(idx_a[local_a]), int(idx_b[local_b]))
        for local_a, row in enumerate(ab)
        for local_b in row
        if local_a in ba_sets[local_b]
    }


def _pairwise_supported_edges(view, idx_a, idx_b, k):
    """Edges supported by a top-k relation in either direction."""
    ab = NearestNeighbors(n_neighbors=min(k, len(idx_b)), metric="cosine").fit(
        view[idx_b]
    ).kneighbors(view[idx_a], return_distance=False)
    ba = NearestNeighbors(n_neighbors=min(k, len(idx_a)), metric="cosine").fit(
        view[idx_a]
    ).kneighbors(view[idx_b], return_distance=False)
    edges = {
        (int(idx_a[local_a]), int(idx_b[local_b]))
        for local_a, row in enumerate(ab)
        for local_b in row
    }
    edges.update(
        (int(idx_a[local_a]), int(idx_b[local_b]))
        for local_b, row in enumerate(ba)
        for local_a in row
    )
    return edges


def build_cycle_consistent_anchors(
    teacher_embeddings,
    macrogenes,
    species,
    z_k=10,
    macrogene_k=10,
    minimum_cell_coverage=0.10,
):
    teacher_embeddings = _normalize_numpy(teacher_embeddings)
    macrogenes = _normalize_numpy(macrogenes)
    species = np.asarray(species).astype(str)
    species_names = sorted(np.unique(species))
    local_support = [set() for _ in range(len(species))]
    for species_name in species_names:
        indices = np.flatnonzero(species == species_name)
        local = NearestNeighbors(
            n_neighbors=min(51, len(indices)), metric="cosine"
        ).fit(teacher_embeddings[indices]).kneighbors(
            teacher_embeddings[indices], return_distance=False
        )[:, 1:]
        for local_index, row in enumerate(local):
            local_support[indices[local_index]] = set(indices[row].tolist())

    def build(macro_k, require_macro_mutual=True):
        edges = set()
        neighbors = [defaultdict(set) for _ in range(len(species))]
        for species_a, species_b in combinations(species_names, 2):
            idx_a = np.flatnonzero(species == species_a)
            idx_b = np.flatnonzero(species == species_b)
            z_edges = _pairwise_mutual_edges(teacher_embeddings, idx_a, idx_b, z_k)
            macro_edges = (
                _pairwise_mutual_edges(macrogenes, idx_a, idx_b, macro_k)
                if require_macro_mutual
                else _pairwise_supported_edges(macrogenes, idx_a, idx_b, macro_k)
            )
            for edge in z_edges & macro_edges:
                edges.add(edge)
                a, b = edge
                neighbors[a][species_b].add(b)
                neighbors[b][species_a].add(a)

        supported = []
        for a, b in edges:
            third_species = [name for name in species_names if name not in {species[a], species[b]}]
            support_count = 0
            for name in third_species:
                candidates_a = neighbors[a][name]
                candidates_b = neighbors[b][name]
                exact_support = bool(candidates_a & candidates_b)
                local_support_match = any(
                    candidate_b in local_support[candidate_a]
                    for candidate_a in candidates_a
                    for candidate_b in candidates_b
                )
                support_count += int(exact_support or local_support_match)
            if support_count == 0:
                continue
            support_fraction = support_count / max(len(third_species), 1)
            z_similarity = float(teacher_embeddings[a] @ teacher_embeddings[b])
            macro_similarity = float(macrogenes[a] @ macrogenes[b])
            confidence = 0.4 * z_similarity + 0.4 * macro_similarity + 0.2 * support_fraction
            supported.append((a, b, confidence, support_fraction))
        return edges, supported

    strict_edges, supported_edges = build(macrogene_k)
    raw_edges = strict_edges
    covered = {idx for edge in supported_edges for idx in edge[:2]}
    fallback_used = len(covered) / len(species) < minimum_cell_coverage
    if fallback_used:
        raw_edges, supported_edges = build(25, require_macro_mutual=False)
        covered = {idx for edge in supported_edges for idx in edge[:2]}

    candidates = [[] for _ in range(len(species))]
    for a, b, confidence, support in supported_edges:
        candidates[a].append((b, confidence, support))
        candidates[b].append((a, confidence, support))
    positive_indices = np.full((len(species), 3), -1, dtype=np.int64)
    confidence_values = np.full((len(species), 3), -np.inf, dtype=np.float32)
    cycle_support = np.zeros((len(species), 3), dtype=np.float32)
    for anchor, values in enumerate(candidates):
        values = sorted(values, key=lambda item: item[1], reverse=True)[:3]
        for position, (positive, confidence, support) in enumerate(values):
            positive_indices[anchor, position] = positive
            confidence_values[anchor, position] = confidence
            cycle_support[anchor, position] = support

    valid_confidence = confidence_values[np.isfinite(confidence_values)]
    quantiles = (
        np.quantile(valid_confidence, [0.75, 0.50, 0.25, 0.0]).astype(np.float32)
        if len(valid_confidence)
        else np.asarray([np.inf] * 4, dtype=np.float32)
    )
    pair_diagnostics = []
    for species_a, species_b in combinations(species_names, 2):
        pair_raw = {
            edge for edge in raw_edges
            if {species[edge[0]], species[edge[1]]} == {species_a, species_b}
        }
        pair_strict = {
            edge for edge in strict_edges
            if {species[edge[0]], species[edge[1]]} == {species_a, species_b}
        }
        pair_supported = {
            (edge[0], edge[1]) for edge in supported_edges
            if {species[edge[0]], species[edge[1]]} == {species_a, species_b}
        }
        pair_diagnostics.append({
            "species_a": species_a,
            "species_b": species_b,
            "candidate_edges": len(pair_raw),
            "rejected_candidates": len(pair_raw) - len(pair_supported),
            "reciprocal_rate": len(pair_raw & pair_strict) / max(len(pair_raw), 1),
            "cycle_support_rate": len(pair_supported) / max(len(pair_raw), 1),
        })
    diagnostics = {
        "candidate_edges": len(raw_edges),
        "cycle_supported_edges": len(supported_edges),
        "covered_cells": len(covered),
        "cell_coverage": len(covered) / len(species),
        "fallback_used": fallback_used,
        "macrogene_k_used": 25 if fallback_used else macrogene_k,
        "confidence_quantiles": quantiles.tolist(),
        "reciprocal_rate": len(raw_edges & strict_edges) / max(len(raw_edges), 1),
        "cycle_support_rate": len(supported_edges) / max(len(raw_edges), 1),
        "rejected_candidates": len(raw_edges) - len(supported_edges),
        "species_pair_diagnostics": pair_diagnostics,
    }
    return positive_indices, confidence_values, cycle_support, quantiles, diagnostics


def multiscale_preservation_loss(
    embeddings,
    global_indices,
    embedding_bank,
    teacher_embeddings,
    preservation_pairs,
    teacher_pair_similarities,
    temperature=0.1,
):
    rows = global_indices.detach().cpu()
    pairs = preservation_pairs[rows].to(embeddings.device)
    teacher_targets = teacher_pair_similarities[rows].to(embeddings.device)
    valid_rows = torch.all(pairs >= 0, dim=1)
    if not torch.any(valid_rows):
        zero = embeddings.sum() * 0
        return zero, zero, zero
    anchors = F.normalize(embeddings[valid_rows], dim=1)
    pairs = pairs[valid_rows]
    teacher_targets = teacher_targets[valid_rows]
    bank_pairs = F.normalize(embedding_bank[pairs], dim=2)
    similarities = torch.einsum("bd,bkd->bk", anchors, bank_pairs)
    logits = similarities / temperature
    info_nce = -(
        torch.logsumexp(logits[:, :3], dim=1) - torch.logsumexp(logits, dim=1)
    ).mean()
    distortion = F.smooth_l1_loss(similarities, teacher_targets)
    total = 0.5 * info_nce + 0.5 * distortion
    return total, info_nce, distortion


def graph_preservation_metrics(embeddings, species, teacher_neighbors):
    embeddings = _normalize_numpy(embeddings)
    species = np.asarray(species).astype(str)
    student_neighbors = {}
    for k in (5, 15, 50):
        output = np.full((len(embeddings), k), -1, dtype=np.int64)
        for species_name in np.unique(species):
            global_indices = np.flatnonzero(species == species_name)
            local_k = min(k, len(global_indices) - 1)
            local = NearestNeighbors(n_neighbors=local_k + 1, metric="cosine").fit(
                embeddings[global_indices]
            ).kneighbors(embeddings[global_indices], return_distance=False)[:, 1:]
            output[global_indices, :local_k] = global_indices[local]
        student_neighbors[k] = output

    result = {}
    for k in (5, 15, 50):
        overlaps = []
        for row in range(len(embeddings)):
            teacher = set(teacher_neighbors[k][row][teacher_neighbors[k][row] >= 0])
            student = set(student_neighbors[k][row][student_neighbors[k][row] >= 0])
            overlaps.append(len(teacher & student) / max(len(teacher), 1))
        result[f"top{k}_overlap"] = float(np.mean(overlaps))
    recalls = []
    for row in range(len(embeddings)):
        teacher = set(teacher_neighbors[15][row][teacher_neighbors[15][row] >= 0])
        student = set(student_neighbors[50][row][student_neighbors[50][row] >= 0])
        recalls.append(len(teacher & student) / max(len(teacher), 1))
    result["teacher_top15_recall_at_50"] = float(np.mean(recalls))
    return result


def embedding_shape_metrics(embeddings):
    embeddings = F.normalize(torch.as_tensor(embeddings, dtype=torch.float32), dim=1)
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    variance = centered.var(dim=0, unbiased=False)
    singular_values = torch.linalg.svdvals(centered)
    probabilities = singular_values / singular_values.sum().clamp_min(1e-12)
    effective_rank = torch.exp(
        -torch.sum(probabilities * torch.log(probabilities.clamp_min(1e-12)))
    )
    return {
        "embedding_variance_mean": float(variance.mean()),
        "embedding_variance_min": float(variance.min()),
        "embedding_effective_rank": float(effective_rank),
    }


def gradient_diagnostics(alignment_loss, preservation_loss, parameters, multiplier):
    parameters = [parameter for parameter in parameters if parameter.requires_grad]
    alignment_gradients = torch.autograd.grad(
        alignment_loss, parameters, retain_graph=True, allow_unused=True
    )
    preservation_gradients = torch.autograd.grad(
        preservation_loss, parameters, retain_graph=True, allow_unused=True
    )
    alignment_parts = []
    preservation_parts = []
    for parameter, alignment_gradient, preservation_gradient in zip(
        parameters, alignment_gradients, preservation_gradients
    ):
        alignment_parts.append(
            torch.zeros_like(parameter).reshape(-1)
            if alignment_gradient is None
            else alignment_gradient.reshape(-1)
        )
        preservation_parts.append(
            torch.zeros_like(parameter).reshape(-1)
            if preservation_gradient is None
            else preservation_gradient.reshape(-1)
        )
    alignment_vector = torch.cat(alignment_parts)
    preservation_vector = torch.cat(preservation_parts)
    alignment_norm = torch.linalg.vector_norm(alignment_vector)
    preservation_norm = torch.linalg.vector_norm(preservation_vector)
    cosine = F.cosine_similarity(alignment_vector, preservation_vector, dim=0)
    weighted_ratio = multiplier * preservation_norm / alignment_norm.clamp_min(1e-12)
    return {
        "alignment_gradient_norm": float(alignment_norm.detach().cpu()),
        "preservation_gradient_norm": float(preservation_norm.detach().cpu()),
        "gradient_cosine_similarity": float(cosine.detach().cpu()),
        "weighted_gradient_ratio": float(weighted_ratio.detach().cpu()),
    }


def update_preservation_multiplier(
    multiplier, recall_at_50, target=0.70, update_rate=5.0, cap=50.0
):
    violation = max(0.0, float(target) - float(recall_at_50))
    updated = min(float(cap), float(multiplier) + float(update_rate) * violation)
    return updated, violation


def curriculum_fraction(epoch):
    return (0.25, 0.50, 0.75, 1.0)[min((epoch - 1) // 5, 3)]


def curriculum_anchor_loss(
    embeddings,
    global_indices,
    embedding_bank,
    positive_indices,
    confidence_values,
    threshold,
):
    rows = global_indices.detach().cpu()
    positives = positive_indices[rows].to(embeddings.device)
    confidence = confidence_values[rows].to(embeddings.device)
    active = (positives >= 0) & torch.isfinite(confidence) & (confidence >= threshold)
    if not torch.any(active):
        return embeddings.sum() * 0, {"coverage": 0, "active_edges": 0, "mean_confidence": np.nan}
    safe = positives.clamp_min(0)
    anchors = F.normalize(embeddings, dim=1)
    targets = F.normalize(embedding_bank[safe], dim=2)
    similarities = torch.einsum("bd,bkd->bk", anchors, targets)
    weights = torch.softmax(confidence.masked_fill(~active, -torch.inf), dim=1)
    weights = torch.nan_to_num(weights, nan=0.0)
    loss = ((1 - similarities) * weights * active).sum() / (weights * active).sum().clamp_min(1e-12)
    diagnostics = {
        "coverage": int(torch.any(active, dim=1).sum()),
        "active_edges": int(active.sum()),
        "mean_confidence": float(confidence[active].mean().detach().cpu()),
    }
    return loss, diagnostics


def partial_ot_loss_with_diagnostics(
    embeddings,
    teacher_embeddings,
    macrogenes,
    species_codes,
    epsilon=0.05,
    transported_mass=0.3,
    iterations=50,
):
    embeddings = F.normalize(embeddings, dim=1)
    teacher_embeddings = F.normalize(teacher_embeddings, dim=1)
    macrogenes = F.normalize(macrogenes, dim=1)
    losses, diagnostics = [], []
    for species_a, species_b in combinations(torch.unique(species_codes).tolist(), 2):
        mask_a, mask_b = species_codes == species_a, species_codes == species_b
        with torch.no_grad():
            cost = 0.5 * (1 - teacher_embeddings[mask_a] @ teacher_embeddings[mask_b].T)
            cost += 0.5 * (1 - macrogenes[mask_a] @ macrogenes[mask_b].T)
            plan = partial_sinkhorn(cost, epsilon, transported_mass, iterations)
        student_cost = 1 - embeddings[mask_a] @ embeddings[mask_b].T
        losses.append(torch.sum(plan * student_cost) / transported_mass)
        positive_plan = plan[plan > 0]
        active_costs = cost[plan > 1e-12]
        entropy = -torch.sum(positive_plan * torch.log(positive_plan.clamp_min(1e-30)))
        diagnostics.append({
            "species_a": species_a,
            "species_b": species_b,
            "requested_mass": transported_mass,
            "realized_mass": float(plan.sum().cpu()),
            "row_marginal_violation": float(torch.clamp(plan.sum(1) - 1 / len(plan), min=0).max().cpu()),
            "column_marginal_violation": float(torch.clamp(plan.sum(0) - 1 / plan.shape[1], min=0).max().cpu()),
            "plan_entropy": float(entropy.cpu()),
            "active_edges": int((plan > 1e-12).sum().cpu()),
            "coverage": float(
                0.5 * ((plan.sum(1) > 1e-12).float().mean() + (plan.sum(0) > 1e-12).float().mean())
            ),
            "transport_cost_mean": float((plan * cost).sum().cpu() / transported_mass),
            "transport_cost_p05": float(torch.quantile(active_costs, 0.05).cpu()),
            "transport_cost_p50": float(torch.quantile(active_costs, 0.50).cpu()),
            "transport_cost_p95": float(torch.quantile(active_costs, 0.95).cpu()),
        })
    return torch.stack(losses).mean(), diagnostics


def fused_gw_loss_with_diagnostics(
    embeddings,
    teacher_embeddings,
    macrogenes,
    species_codes,
    alpha=0.5,
    epsilon=0.05,
    transported_mass=0.3,
    outer_iterations=10,
    sinkhorn_iterations=50,
):
    embeddings = F.normalize(embeddings, dim=1)
    teacher_embeddings = F.normalize(teacher_embeddings, dim=1)
    macrogenes = F.normalize(macrogenes, dim=1)
    losses, diagnostics = [], []
    for species_a, species_b in combinations(torch.unique(species_codes).tolist(), 2):
        mask_a, mask_b = species_codes == species_a, species_codes == species_b
        with torch.no_grad():
            feature_cost = 0.5 * (1 - teacher_embeddings[mask_a] @ teacher_embeddings[mask_b].T)
            feature_cost += 0.5 * (1 - macrogenes[mask_a] @ macrogenes[mask_b].T)
            geometry_a = 1 - teacher_embeddings[mask_a] @ teacher_embeddings[mask_a].T
            geometry_b = 1 - teacher_embeddings[mask_b] @ teacher_embeddings[mask_b].T
            plan = partial_sinkhorn(feature_cost, epsilon, transported_mass, sinkhorn_iterations)
            structural_cost = torch.zeros_like(feature_cost)
            for _ in range(outer_iterations):
                row_mass, col_mass = plan.sum(1), plan.sum(0)
                structural_cost = (
                    (geometry_a.pow(2) @ row_mass)[:, None]
                    + (geometry_b.pow(2) @ col_mass)[None, :]
                )
                structural_cost -= 2 * geometry_a @ plan @ geometry_b.T
                structural_cost -= structural_cost.min()
                structural_cost /= structural_cost.max().clamp_min(1e-12)
                fused_cost = (1 - alpha) * feature_cost + alpha * structural_cost
                plan = partial_sinkhorn(fused_cost, epsilon, transported_mass, sinkhorn_iterations)
        student_cost = 1 - embeddings[mask_a] @ embeddings[mask_b].T
        losses.append(torch.sum(plan * student_cost) / transported_mass)
        positive_plan = plan[plan > 0]
        active_costs = feature_cost[plan > 1e-12]
        entropy = -torch.sum(positive_plan * torch.log(positive_plan.clamp_min(1e-30)))
        diagnostics.append({
            "species_a": species_a,
            "species_b": species_b,
            "requested_mass": transported_mass,
            "realized_mass": float(plan.sum().cpu()),
            "row_marginal_violation": float(torch.clamp(plan.sum(1) - 1 / len(plan), min=0).max().cpu()),
            "column_marginal_violation": float(torch.clamp(plan.sum(0) - 1 / plan.shape[1], min=0).max().cpu()),
            "plan_entropy": float(entropy.cpu()),
            "active_edges": int((plan > 1e-12).sum().cpu()),
            "coverage": float(
                0.5 * ((plan.sum(1) > 1e-12).float().mean() + (plan.sum(0) > 1e-12).float().mean())
            ),
            "transport_cost_mean": float((plan * feature_cost).sum().cpu() / transported_mass),
            "transport_cost_p05": float(torch.quantile(active_costs, 0.05).cpu()),
            "transport_cost_p50": float(torch.quantile(active_costs, 0.50).cpu()),
            "transport_cost_p95": float(torch.quantile(active_costs, 0.95).cpu()),
            "structural_cost_mean": float((plan * structural_cost).sum().cpu() / transported_mass),
        })
    return torch.stack(losses).mean(), diagnostics

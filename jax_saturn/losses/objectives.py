"""Label-free alignment losses using fixed arrays and explicit padding masks."""

from itertools import combinations

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp, xlogy

from .core import masked_mean, normalize


def _valid_rows(embeddings, valid_mask):
    return jnp.ones(embeddings.shape[0], dtype=bool) if valid_mask is None else jnp.asarray(valid_mask, dtype=bool)


def _masked_logsumexp(values, mask):
    # All-invalid rows receive harmless logits, avoiding NaN gradients at -inf.
    mask = jnp.broadcast_to(mask, values.shape)
    active = jnp.any(mask, axis=-1, keepdims=True)
    logits = jnp.where(active, jnp.where(mask, values, -jnp.inf), 0)
    return logsumexp(logits, axis=-1)


def preservation_distillation_loss(embeddings, global_indices, embedding_bank,
                                  candidate_indices, teacher_probabilities,
                                  teacher_similarities, temperature=0.1, *, valid_mask=None):
    candidates = candidate_indices[global_indices]
    targets = teacher_probabilities[global_indices].astype(jnp.float32)
    similarity_targets = teacher_similarities[global_indices].astype(jnp.float32)
    valid = (candidates >= 0) & _valid_rows(embeddings, valid_mask)[:, None]
    rows = jnp.any(valid, axis=-1)
    anchors = normalize(embeddings)
    bank = normalize(jax.lax.stop_gradient(embedding_bank)[jnp.maximum(candidates, 0)])
    similarities = jnp.einsum("bd,bkd->bk", anchors, bank, precision=jax.lax.Precision.HIGHEST)
    logits = similarities / temperature
    log_probs = logits - _masked_logsumexp(logits, valid)[:, None]
    targets = jnp.where(valid, targets, 0)
    # Never multiply target zero by a masked -inf log probability.
    kl = xlogy(targets, targets) - targets * log_probs
    distribution = masked_mean(jnp.sum(jnp.where(valid, kl, 0), axis=-1), rows)
    error = jnp.abs(similarities - similarity_targets)
    smooth_l1 = jnp.where(error < 1, 0.5 * error ** 2, error - 0.5)
    return distribution + masked_mean(smooth_l1, valid)


def _infonce(embeddings, global_indices, embedding_bank, positives, target_indices,
             global_to_local, temperature, row_masks, *, exclude_self):
    anchors = normalize(embeddings)
    total, count = jnp.float32(0), jnp.int32(0)
    covered = jnp.zeros(embeddings.shape[0], dtype=bool)
    for code, target_global in enumerate(target_indices):
        # Empty species banks have no contrastive rows.
        if target_global.shape[0] == 0:
            continue
        target_positive = positives[:, code]
        valid = (target_positive >= 0) & row_masks[code][:, None]
        active = jnp.any(valid, axis=-1)
        bank = normalize(jax.lax.stop_gradient(embedding_bank)[target_global])
        logits = jnp.matmul(anchors, bank.T, precision=jax.lax.Precision.HIGHEST) / temperature
        denominator_mask = jnp.broadcast_to(active[:, None], logits.shape)
        if exclude_self:
            denominator_mask &= target_global[None, :] != global_indices[:, None]
        local = global_to_local[code][jnp.maximum(target_positive, 0)]
        local = jnp.where(valid, local, 0)
        positive_logits = jnp.take_along_axis(logits, jnp.maximum(local, 0), axis=1)
        numerator = _masked_logsumexp(positive_logits, valid)
        denominator = _masked_logsumexp(logits, denominator_mask)
        loss = jnp.where(active, denominator - numerator, 0)
        total += jnp.sum(loss)
        count += jnp.sum(active)
        covered |= active
    return total / jnp.maximum(count, 1), jnp.sum(covered)


def multi_positive_infonce_loss(embeddings, global_indices, embedding_bank,
                                positive_indices, target_indices, global_to_local,
                                temperature=0.1, *, valid_mask=None):
    valid = _valid_rows(embeddings, valid_mask)
    return _infonce(embeddings, global_indices, embedding_bank,
                    positive_indices[global_indices], target_indices, global_to_local,
                    temperature, [valid] * len(target_indices), exclude_self=False)


def within_species_graph_infonce_loss(embeddings, global_indices, species_codes,
                                     embedding_bank, teacher_neighbors, target_indices,
                                     global_to_local, positive_k=5, temperature=0.1, *, valid_mask=None):
    valid = _valid_rows(embeddings, valid_mask)
    neighbors = teacher_neighbors[global_indices, :positive_k]
    positives = jnp.broadcast_to(neighbors[:, None, :],
                                 (len(embeddings), len(target_indices), neighbors.shape[-1]))
    rows = [valid & (species_codes == code) for code in range(len(target_indices))]
    return _infonce(embeddings, global_indices, embedding_bank, positives,
                    target_indices, global_to_local, temperature, rows, exclude_self=True)


def multi_species_mmd(embeddings, species_codes, base_bandwidth, *, num_species, valid_mask=None):
    embeddings = normalize(embeddings)
    valid = _valid_rows(embeddings, valid_mask)
    norms = jnp.sum(embeddings ** 2, axis=-1)
    similarities = jnp.matmul(embeddings, embeddings.T, precision=jax.lax.Precision.HIGHEST)
    distances = jnp.maximum(norms[:, None] + norms[None, :] - 2 * similarities, 0)
    kernel = sum(jnp.exp(-distances / jnp.maximum(scale * base_bandwidth, 1e-6))
                 for scale in (0.5, 1., 2.)) / 3
    total, pairs = jnp.float32(0), jnp.int32(0)
    for code_a, code_b in combinations(range(num_species), 2):
        a, b = valid & (species_codes == code_a), valid & (species_codes == code_b)
        active = (jnp.sum(a) >= 2) & (jnp.sum(b) >= 2)
        loss = (masked_mean(kernel, a[:, None] & a[None, :])
                + masked_mean(kernel, b[:, None] & b[None, :])
                - 2 * masked_mean(kernel, a[:, None] & b[None, :]))
        total += jnp.where(active, loss, 0)
        pairs += active
    return total / jnp.maximum(pairs, 1)


def partial_sinkhorn(cost, epsilon=0.05, transported_mass=0.8, iterations=100, *, row_mask=None, col_mask=None):
    """Reference KL projections with zero capacity for any padded rows/columns."""
    cost = jnp.asarray(cost, dtype=jnp.float32)
    if cost.ndim != 2 or not cost.size:
        raise ValueError("cost must be a non-empty matrix")
    if not 0 < transported_mass <= 1 or epsilon <= 0 or iterations < 1:
        raise ValueError("Sinkhorn requires mass in (0,1], epsilon > 0 and iterations >= 1")
    rows = jnp.ones(cost.shape[0], dtype=bool) if row_mask is None else row_mask
    cols = jnp.ones(cost.shape[1], dtype=bool) if col_mask is None else col_mask
    valid = rows[:, None] & cols[None, :]
    active = jnp.any(valid)
    row_cap = rows / jnp.maximum(jnp.sum(rows), 1)
    col_cap = cols / jnp.maximum(jnp.sum(cols), 1)
    minimum = jnp.where(active, jnp.min(jnp.where(valid, cost, jnp.inf)), 0)
    shifted = jnp.where(valid, cost - minimum, 0)
    plan = jnp.where(valid, jnp.maximum(jnp.exp(-shifted / epsilon), 1e-30), 0)

    def candidate(plan, correction):
        return jnp.where(valid, jnp.maximum(plan * correction, 1e-30), 0)

    def correction(before, after):
        return jnp.where(valid, jnp.minimum(before / jnp.maximum(after, 1e-30), 1e30), 0)

    def iteration(_, state):
        plan, q_row, q_col, q_mass = state
        before = candidate(plan, q_row)
        factors = jnp.minimum(1, row_cap / jnp.maximum(before.sum(axis=1), 1e-30))
        plan = before * factors[:, None]
        q_row = correction(before, plan)
        before = candidate(plan, q_col)
        factors = jnp.minimum(1, col_cap / jnp.maximum(before.sum(axis=0), 1e-30))
        plan = before * factors[None, :]
        q_col = correction(before, plan)
        before = candidate(plan, q_mass)
        plan = before * (transported_mass * active / jnp.maximum(before.sum(), 1e-30))
        q_mass = correction(before, plan)
        return plan, q_row, q_col, q_mass

    ones = jnp.ones_like(plan)
    return jax.lax.fori_loop(0, iterations, iteration, (plan, ones, ones, ones))[0]


def partial_ot_alignment_loss(embeddings, teacher_embeddings, macrogenes, species_codes,
                              epsilon=0.05, transported_mass=0.8, iterations=100, *,
                              num_species, valid_mask=None):
    student = normalize(embeddings)
    teacher = normalize(jax.lax.stop_gradient(teacher_embeddings))
    macro = normalize(jax.lax.stop_gradient(macrogenes))
    valid = _valid_rows(embeddings, valid_mask)
    teacher_similarity = jnp.matmul(teacher, teacher.T, precision=jax.lax.Precision.HIGHEST)
    macro_similarity = jnp.matmul(macro, macro.T, precision=jax.lax.Precision.HIGHEST)
    teacher_cost = 0.5 * (1 - teacher_similarity) + 0.5 * (1 - macro_similarity)
    student_cost = 1 - jnp.matmul(student, student.T, precision=jax.lax.Precision.HIGHEST)
    total, pairs, masses = jnp.float32(0), jnp.int32(0), []
    for a, b in combinations(range(num_species), 2):
        rows, cols = valid & (species_codes == a), valid & (species_codes == b)
        active = jnp.any(rows) & jnp.any(cols)
        plan = jax.lax.stop_gradient(partial_sinkhorn(teacher_cost, epsilon, transported_mass,
                                                     iterations, row_mask=rows, col_mask=cols))
        mass = jnp.sum(plan)
        total += jnp.where(active, jnp.sum(plan * student_cost) / jnp.maximum(mass, 1e-12), 0)
        pairs += active
        masses.append(mass)
    # Fixed pair order, with zero mass for absent species, supports compilation.
    return total / jnp.maximum(pairs, 1), jnp.asarray(masses, dtype=jnp.float32)

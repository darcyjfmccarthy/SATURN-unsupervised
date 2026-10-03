"""Reconstruction, ranking and cosine triplet losses in float32."""

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln


def normalize(values, *, axis=-1, epsilon=1e-12):
    values = jnp.asarray(values, dtype=jnp.float32)
    # Flooring before sqrt keeps the gradient at a zero vector finite.
    norm = jnp.sqrt(jnp.maximum(jnp.sum(values ** 2, axis=axis, keepdims=True), epsilon ** 2))
    return values / norm


def masked_mean(values, mask):
    mask = jnp.asarray(mask, dtype=bool)
    return jnp.sum(jnp.where(mask, values, 0)) / jnp.maximum(jnp.sum(mask), 1)


def _softplus_value(values):
    return jnp.where(values > 20, values, jnp.log1p(jnp.exp(jnp.minimum(values, 20))))


@jax.custom_vjp
def _reference_softplus(values):
    """PyTorch beta=1, threshold=20 softplus, including backward arithmetic.

    logaddexp's gradient can round to one earlier than PyTorch's native
    exp/(exp+1) rule. ZINB subtracts saturated softplus gradients, so this
    difference can change tiny dropout gradients and their Adam updates.
    """
    return _softplus_value(values)


def _softplus_forward(values):
    return _softplus_value(values), values


def _softplus_backward(values, cotangent):
    # PyTorch's native kernel multiplies the cotangent before division.
    # https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/ActivationSoftplusKernel.cu
    # Clamp the unused branch to prevent overflow above the linear threshold.
    exponential = jnp.exp(jnp.minimum(values, 20))
    return (jnp.where(values > 20, cotangent, cotangent * exponential / (exponential + 1)),)


_reference_softplus.defvjp(_softplus_forward, _softplus_backward)


def zinb_log_prob(x, mu, theta, zi_logits, *, epsilon=1e-8):
    """Match scvi log_zinb_positive, including eps and count threshold behavior."""
    x, mu, theta, logits = [jnp.asarray(v, dtype=jnp.float32) for v in (x, mu, theta, zi_logits)]
    log_total = jnp.log(theta + mu + epsilon)
    pi_theta_log = -logits + theta * (jnp.log(theta + epsilon) - log_total)
    softplus_pi = _reference_softplus(-logits)
    zero = _reference_softplus(pi_theta_log) - softplus_pi
    nonzero = (-softplus_pi + pi_theta_log + x * (jnp.log(mu + epsilon) - log_total)
               + gammaln(x + theta) - gammaln(theta) - gammaln(x + 1))
    return jnp.where(x < epsilon, zero, 0) + jnp.where(x > epsilon, nonzero, 0)


def zinb_reconstruction_loss(x, mu, theta, zi_logits, *, valid_mask=None, weights=None):
    per_cell = -jnp.sum(zinb_log_prob(x, mu, theta, zi_logits), axis=-1)
    if weights is not None:
        per_cell = per_cell * jnp.asarray(weights, dtype=jnp.float32)
    if valid_mask is not None:
        per_cell = jnp.where(valid_mask, per_cell, 0)
    return jnp.sum(per_cell)


def l1_loss(log_gene_to_macrogene):
    return jnp.sum(jnp.abs(jnp.exp(jnp.asarray(log_gene_to_macrogene, dtype=jnp.float32))))


def gene_weight_ranking_loss(learned_embeddings, protein_embeddings, paired_indices=None, *, key=None):
    """Use explicit sampled indices for parity or an explicit key for training."""
    n = learned_embeddings.shape[0]
    if paired_indices is None:
        if key is None:
            raise ValueError("Ranking loss needs paired_indices or an explicit PRNG key")
        paired_indices = jax.random.randint(key, (n,), 0, n)
    learned = normalize(learned_embeddings, epsilon=1e-8)
    protein = normalize(protein_embeddings, epsilon=1e-8)
    predicted = jnp.sum(learned * learned[paired_indices], axis=-1)
    target = jnp.sum(protein * protein[paired_indices], axis=-1)
    return jnp.sum((predicted - target) ** 2)


def cosine_similarity(left, right=None):
    right = left if right is None else right
    return jnp.matmul(normalize(left), normalize(right).T, precision=jax.lax.Precision.HIGHEST)


def triplet_filter_mask(embeddings, indices, *, margin=0.2, kind="semihard", valid_mask=None):
    anchors, positive, negative = indices
    normalized = normalize(embeddings)
    difference = jnp.sum(normalized[anchors] * (normalized[positive] - normalized[negative]), axis=-1)
    if kind == "unfiltered":
        selected = jnp.ones(difference.shape, dtype=bool)
    elif kind == "easy":
        selected = difference > margin
    elif kind in {"all", "hard", "semihard"}:
        selected = difference <= margin
        if kind == "hard":
            selected &= difference <= 0
        elif kind == "semihard":
            selected &= difference > 0
    else:
        raise ValueError(f"Unknown triplet filter: {kind}")
    return selected if valid_mask is None else selected & valid_mask


def triplet_margin_loss(embeddings, indices, *, margin=0.2, valid_mask=None, active_only=True):
    """Training averages positive losses; evaluation averages all frozen triplets."""
    anchors, positive, negative = indices
    normalized = normalize(embeddings)
    ap = jnp.sum(normalized[anchors] * normalized[positive], axis=-1)
    an = jnp.sum(normalized[anchors] * normalized[negative], axis=-1)
    losses = jax.nn.relu(an - ap + margin)
    selected = jnp.ones(losses.shape, dtype=bool) if valid_mask is None else valid_mask
    if active_only:
        selected = selected & (losses > 0)
    return masked_mean(losses, selected)

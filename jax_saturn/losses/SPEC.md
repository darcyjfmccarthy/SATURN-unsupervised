# Losses Spec

## Status
Phase 3 pure JAX losses and deterministic CPU fixture parity implemented.
Training integration and full HMM acceptance remain pending.

## Notebook Scope
This spec covers losses and objectives touched by notebook cells 7, 13, 16, 18, 20, 22, and 24.

In scope source files:
- `model/saturn_model.py`
- `train-saturn.py`
- `scripts/prepare_label_agnostic_artifacts.py`
- `scripts/train_label_agnostic.py`
- `label_agnostic/objectives.py`
- `label_agnostic/metrics.py`
- `losses/triplet_margin_loss.py`
- `miners/triplet_margin_miner.py`
- `distances/cosine_similarity.py`
- `utils/loss_and_miner_utils.py`

## Current PyTorch Contract
Pretraining losses:
- ZINB reconstruction loss from `scvi.distributions.ZeroInflatedNegativeBinomial`.
- Optional VAE KL loss, inactive in notebook defaults.
- L1/lasso loss on `exp(p_weights)`.
- Protein-embedding ranking loss.

Labeled baseline metric losses:
- Cosine similarity distance with normalized embeddings.
- Cross-species MNN triplet mining.
- Semihard triplet filtering by margin.
- Triplet margin loss with margin `0.2`.

Label-agnostic losses:
- Preservation distillation loss.
- Within-species graph InfoNCE loss.
- Multi-positive cross-species InfoNCE loss.
- Multi-species MMD loss.
- Partial OT alignment loss with entropy-regularized partial Sinkhorn.

Evaluation-only loss:
- Fixed triplet margin loss over frozen triplets in `label_agnostic.metrics.fixed_triplet_margin_loss`.

## JAX Design Target
All future JAX losses must be pure functions over arrays and explicit configuration values.

ZINB reconstruction:
- Inputs:
  - `x`: counts, float32 or int-like float, shape `[batch, genes]`.
  - `mu`: positive mean, float32, shape `[batch, genes]`.
  - `theta`: positive inverse dispersion, float32, shape `[genes]` or `[batch, genes]`.
  - `zi_logits`: dropout logits, float32, shape `[batch, genes]`.
- Negative binomial log probability:
  - `nb = lgamma(x + theta) - lgamma(theta) - lgamma(x + 1)`
  - `nb += theta * (log(theta) - log(theta + mu))`
  - `nb += x * (log(mu) - log(theta + mu))`
- Zero-inflated log probability:
  - For `x == 0`: `logaddexp(log_sigmoid(zi_logits), log_sigmoid(-zi_logits) + nb)`
  - For `x > 0`: `log_sigmoid(-zi_logits) + nb`
- Reconstruction loss per cell: `-sum(log_prob, axis=-1)`.
- Training reconstruction loss: sum over cells, optionally weighted by inverse label-frequency weights.

Implementation follows scvi's actual `log_zinb_positive` stabilizers (`eps=1e-8`)
and its count thresholds, rather than dropping eps from the simplified formula
above. Masked graph losses must avoid zero targets multiplied by negative
infinity and return finite zero gradients for all-invalid rows.

VAE KL, if enabled later:
- `kld = -0.5 * sum(1 + log_var - mu^2 - exp(log_var), axis=1)`.
- Total loss adds `kld_weight * sum(kld)`.
- Notebook defaults use `vae=False`, so v1 tests can skip VAE training.

L1/lasso:
- `l1 = sum(abs(exp(log_gene_to_macrogene)))`.
- Weighted by `l1_penalty`.

Protein-embedding ranking loss:
- `weights = exp(log_gene_to_macrogene)`, shape `[num_macrogenes, total_hv_genes]`.
- Apply `p_weights_embeddings` to `weights.T`, producing learned gene-weight embeddings `[total_hv_genes, 256]`.
- Sample one paired index per gene from a deterministic PRNG.
- Target similarity is cosine similarity between protein embeddings and sampled protein embeddings.
- Predicted similarity is cosine similarity between learned weight embeddings and sampled learned weight embeddings.
- Loss is sum of squared differences.
- For large runs, allow a spec-approved sampled subset, but HMM parity starts with full current behavior.

Cosine similarity:
- Normalize embeddings by L2 norm with epsilon floor.
- Similarity matrix: `sim = normalized_left @ normalized_right.T`.
- Larger is closer.

Triplet mining for labeled baseline:
- Candidate positives use same label across species, with MNN filtering when `mnn=True`.
- Candidate negatives exclude both the anchor and positive label and come from
  either the anchor's or positive's species, matching the active reference
  `get_species_triplet_indices` implementation.
- For cosine similarity, `ap_sim = sim[anchor, positive]` and `an_sim = sim[anchor, negative]`.
- Miner triplet margin is `ap_sim - an_sim`.
- Semihard candidates satisfy `0 < ap_sim - an_sim <= margin`.
- Hard candidates satisfy `ap_sim - an_sim <= 0`.
- All candidates satisfy `ap_sim - an_sim <= margin`.
- Unfiltered candidates bypass thresholding.

Triplet margin loss:
- `loss = relu(an_sim - ap_sim + margin)`.
- Reduction follows current threshold reducer behavior: average active nonzero losses where current reducer would count active triplets.
- If no triplets exist, return zero with gradient connected to embeddings.

Preservation distillation:
- Inputs:
  - batch embeddings `[batch, dim]`.
  - global indices `[batch]`.
  - embedding bank `[n_cells, dim]`.
  - candidate indices `[n_cells, k]`.
  - teacher probabilities `[n_cells, k]`.
  - teacher similarities `[n_cells, k]`.
- For valid rows, compute cosine logits from batch anchors to bank candidates.
- `distribution_loss = KL(log_softmax(logits / temperature), teacher_probabilities)`.
- `distortion_loss = smooth_l1(student_similarities, teacher_similarities)` over valid entries.
- Return `distribution_loss + distortion_loss`.

Within-species graph InfoNCE:
- For each species in the batch, contrast anchors against the complete same-species memory bank.
- Exclude the anchor itself from the denominator.
- Positives are first `positive_k` teacher neighbors.
- Loss per row: `logsumexp(all_logits) - logsumexp(positive_logits)`.

Multi-positive cross-species InfoNCE:
- For each target species, contrast active anchors against the full target-species bank.
- Positives come from `positive_indices[global_anchor, target_species, :]`.
- Loss per active row: `logsumexp(target_logits) - logsumexp(positive_logits)`.
- Coverage is count of batch rows with at least one valid positive in any target species.

MMD:
- Normalize embeddings.
- Bandwidths are `(0.5 * base_bandwidth, base_bandwidth, 2.0 * base_bandwidth)`.
- RBF kernel: `exp(-squared_distance / max(bandwidth, 1e-6))`.
- For every species pair:
  - `MMD = mean(k(a,a)) + mean(k(b,b)) - 2 * mean(k(a,b))`.
- Loss is average over valid species pairs.

Partial Sinkhorn:
- Input cost matrix shape `[n_rows, n_cols]`.
- `row_cap = ones(n_rows) / n_rows`.
- `col_cap = ones(n_cols) / n_cols`.
- Initialize `plan = exp(-(cost - min(cost)) / epsilon)`, clipped to at least `1e-30`.
- Iterate KL projections onto row caps, column caps, and total transported mass.
- Return plan with total mass close to `transported_mass`.

Partial OT alignment:
- Compute teacher cost with no gradient:
  - `0.5 * (1 - teacher_embeddings_a @ teacher_embeddings_b.T)`
  - plus `0.5 * (1 - macrogenes_a @ macrogenes_b.T)`.
- Compute plan by partial Sinkhorn.
- Student cost: `1 - student_embeddings_a @ student_embeddings_b.T`.
- Pair loss: `sum(plan * student_cost) / sum(plan)`.
- Total loss is average over valid species pairs.

Fixed triplet margin evaluation:
- Use the same formula as triplet margin loss over frozen global triplets.
- Report mean loss, active fraction, mean AP similarity, mean AN similarity, and triplet count.

## Decisions
- Keep losses mathematically equivalent before optimizing.
- Keep KNN graph construction outside JAX losses for v1.
- Keep loss reductions fp32, even when model matmuls use bf16.
- Return zero losses with gradient connectivity when no valid rows or pairs exist.
- Treat coverage values as host metrics, not differentiable outputs.

## Non-Goals
- Porting unused PyTorch metric-learning losses.
- Adding new biological objectives.
- Changing loss weights from notebook defaults during first parity.
- Making Sinkhorn sparse in v1.

## Acceptance Criteria
- Future unit tests compare JAX ZINB to scvi/PyTorch on fixed inputs.
- Future triplet loss parity tests match PyTorch for candidate index tuples.
- Future InfoNCE, MMD, and OT tests match PyTorch on small deterministic fixtures.
- Loss functions are pure, jit-compatible, and mask padded rows.

## Open External Facts
- Final accepted numeric tolerances for bf16 TPU loss parity.
- Whether large 25-species OT should use sub-batched pair matrices in v1 or only after HMM parity.

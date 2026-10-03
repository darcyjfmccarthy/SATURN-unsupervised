# Models Spec

## Status
Phase 2 modules and copied-weight CPU fixture parity implemented. Full HMM
and TPU bf16 scientific parity remain pending.

## Notebook Scope
This spec covers model behavior used by notebook cells 7, 16, 18, 20, and 22.

In scope source files:
- `model/saturn_model.py`
- `train-saturn.py`
- `scripts/train_label_agnostic.py`

## Current PyTorch Contract
The current notebook path uses:
- `SATURNPretrainModel`
- `SATURNMetricModel`

`SATURNPretrainModel` parameters:
- `p_weights`: trainable log gene-to-macrogene weights, shape `[num_macrogenes, total_hv_genes]`.
- `cl_layer_norm`: LayerNorm over macrogenes.
- `encoder`: non-VAE path is two `full_block`s:
  - `Dense(num_macrogenes -> hidden_dim)`, LayerNorm, ReLU, Dropout.
  - `Dense(hidden_dim -> embed_dim)`, LayerNorm, ReLU, Dropout.
- `px_decoder`: one `full_block` from `embed_dim + num_species + num_batch_labels` to `hidden_dim`.
- `cl_scale_decoder`: one `full_block` from `hidden_dim` to `num_macrogenes`.
  This block retains `full_block`'s default dropout 0.1 independently of the
  model-wide configured dropout.
- `px_dropout_decoders`: species-specific linear layers from `hidden_dim` to `n_genes_species`.
- `px_rs`: species-specific trainable log dispersion values, shape `[n_genes_species]`.
- `p_weights_embeddings`: `full_block(num_macrogenes -> 256)` used by protein-embedding ranking loss.

`SATURNPretrainModel.forward(inp, species, batch_labels=None)`:
1. Builds a zero-padded expression matrix with shape `[batch, total_hv_genes]`.
2. Inserts species expression into the species slice.
3. Applies `log1p`.
4. Computes macrogenes with a positive linear projection using `exp(p_weights)`.
5. Applies LayerNorm, ReLU, Dropout.
6. Encodes macrogenes to `embed_dim`.
7. Creates a one-hot species covariate.
8. Current reference behavior sets `spec_idx = 0` for every species.
9. Decodes to hidden dimension.
10. Computes library size as `log(inp.sum(axis=1))`.
11. Decodes hidden to macrogene scale.
12. Projects decoded macrogenes back through `exp(p_weights).T` and slices genes for the active species.
13. Applies softmax over genes to get `px_scale_decode`.
14. Computes `px_rate = exp(library) * px_scale_decode`.
15. Computes `px_drop` with species-specific dropout decoder.
16. Computes `px_r = exp(px_rs[species])`.
17. Returns `(encoder_input, encoded, mu, log_var, px_rate, px_r, px_drop)`.

`SATURNMetricModel` parameters:
- `encoder`: same encoder block structure as pretrain.
- Optional VAE fields exist but are not used by the notebook defaults.

For frozen-macrogene metric learning, `train-saturn.py` copies:
- `pretrain_model.cl_layer_norm`
- `pretrain_model.encoder`

The label-free trainer loads only:
- `encoder.*`
- `cl_layer_norm.*`

`SATURNMetricModel.forward` calls only `encoder`; the stored `cl_layer_norm`
is unused and must not be applied to already-transformed macrogenes.
Reference LayerNorm uses epsilon 1e-5 and centered variance.

## JAX Design Target
Future model implementation uses Flax Linen modules:
- `SaturnPretrainModule`
- `SaturnMetricModule`
- `FullBlock`
- `SpeciesDropoutDecoder`

Parameter tree target for `SaturnPretrainModule`:

```text
params/
  macrogene/
    log_gene_to_macrogene: float32 [num_macrogenes, total_hv_genes]
  cl_layer_norm/
    scale: float32 [num_macrogenes]
    bias: float32 [num_macrogenes]
  encoder/
    block_0/
      dense/kernel: float32 [num_macrogenes, hidden_dim]
      dense/bias: float32 [hidden_dim]
      layer_norm/scale: float32 [hidden_dim]
      layer_norm/bias: float32 [hidden_dim]
    block_1/
      dense/kernel: float32 [hidden_dim, embed_dim]
      dense/bias: float32 [embed_dim]
      layer_norm/scale: float32 [embed_dim]
      layer_norm/bias: float32 [embed_dim]
  decoder/
    px_decoder/block_0/...
    cl_scale_decoder/block_0/...
    px_dropout_decoders/{species}/kernel: float32 [hidden_dim, n_genes_species]
    px_dropout_decoders/{species}/bias: float32 [n_genes_species]
    px_rs/{species}: float32 [n_genes_species]
  p_weights_embeddings/
    block_0/...
```

Parameter tree target for `SaturnMetricModule`:

```text
params/
  cl_layer_norm/
    scale: float32 [num_macrogenes]
    bias: float32 [num_macrogenes]
  encoder/
    block_0/...
    block_1/...
```

Forward API targets:
- `pretrain_apply(params, batch, species_code, train, rngs) -> PretrainOutputs`
- `metric_apply(params, macrogenes, train, rngs) -> embeddings`

`PretrainOutputs` fields:
- `macrogenes`: float32 `[batch, num_macrogenes]`.
- `embedding`: float32 `[batch, embed_dim]`.
- `mu`: optional float32 `[batch, embed_dim]`.
- `log_var`: optional float32 `[batch, embed_dim]`.
- `px_rate`: float32 `[batch, n_genes_species]`.
- `px_r`: float32 `[n_genes_species]` or broadcastable `[batch, n_genes_species]`.
- `px_drop`: float32 `[batch, n_genes_species]`.

Species-sliced implementation:
- Future JAX code should avoid materializing `[batch, total_hv_genes]` when possible.
- For species `s`, use `log_gene_to_macrogene[:, start_s:end_s]`.
- This is algebraically equivalent to zero-padding then applying the full matrix.
- The output must match the PyTorch full-matrix computation within numerical tolerance.

Dtype policy:
- Parameters and optimizer state remain fp32.
- Matmuls may use bf16 on TPU.
- Loss accumulation, ZINB, softmax logits, KL, and Sinkhorn remain fp32.

Dropout and RNG:
- Dropout uses explicit JAX PRNG streams.
- Training state must carry RNG state or folded-in step seeds.
- Evaluation and embedding-bank emission must run with `train=False`.

Parity behavior:
- First JAX implementation must reproduce the current PyTorch species one-hot bug/behavior where `spec_idx = 0` for every species.
- A later `--fix-species-onehot` flag may use the true sorted species code, but this is outside first parity.

## Decisions
- Checkpoint conversion takes independent snapshots of source arrays; subsequent
  NumPy or PyTorch mutations must not alter converted JAX weights on CPU.
- Use Flax Linen, not Equinox or Haiku.
- Use a named parameter tree that is intentionally close to PyTorch state dict names.
- Keep VAE fields out of first implementation unless needed for checkpoint compatibility. Notebook defaults use `vae=False`.
- Do not make the model itself own AnnData or manifest logic.
- Use species names in checkpoint metadata and species codes in compiled functions.

## Non-Goals
- Porting `TransferModel`.
- Porting unused model classes later in `model/saturn_model.py`.
- Fixing the species one-hot behavior during first parity.
- Changing model architecture for speed before HMM parity is accepted.

## Acceptance Criteria
- With copied weights, a future JAX pretrain forward pass matches PyTorch outputs for a fixed HMM mini-batch within tolerance.
- Future JAX metric embeddings match PyTorch metric model embeddings for copied encoder weights within tolerance.
- Parameter shapes are fully determined by manifest preprocessing and hyperparameters.
- Orbax checkpoints can be restored without importing PyTorch.

## Open External Facts
- Final tolerance thresholds for forward parity on TPU bf16 versus CPU/GPU fp32.
- Whether compatibility conversion from existing `.pt` pretrain checkpoints is required for first TPU run.

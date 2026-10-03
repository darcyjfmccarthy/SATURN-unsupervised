# Pretraining Spec

## Status
Phase 4 CPU pretraining, AnnData output and Orbax resume implemented and tested
with synthetic atlases. GPU, full HMM and TPU execution remain unverified.

## Notebook Scope
This spec covers notebook cell 7 and the pretrain outputs consumed by cells 8, 10, 11, 13, 24, and 31.

In scope source files:
- `scripts/human_monkey_mouse.sh`
- `train-saturn.py`
- `pretrain_utils.py`
- `model/saturn_model.py`
- `data/multi_species_data.py`

## Current PyTorch Contract
`train-saturn.py` pretraining:
- Builds `SATURNPretrainModel`.
- Uses `torch.optim.Adam` with `pretrain_lr`.
- Uses `ExperimentDatasetMultiEqual` and `multi_species_collate_fn`.
- Iterates epochs `1..pretrain_epochs`.
- Within each batch, visits species in a random permutation of sorted species names.
- Accumulates per-species reconstruction loss divided by species batch size.
- Adds global L1 and protein-embedding ranking penalties.
- Backpropagates once per mixed batch.
- Saves `pretrain_model.state_dict()` to `PRETRAIN_MODEL_PATH`.
- Reloads the checkpoint if `PRETRAIN_MODEL_PATH` exists.
- Writes final gene-to-macrogene weights pickle.
- Emits `adata_pretrain.h5ad` with `.X` as pretrained embeddings and `.obsm["macrogenes"]`.
- Writes `pretrain_losses.csv` if loss history exists.

Current `adata_pretrain.h5ad` contract:
- `.X`: float embedding matrix `[n_cells, model_dim]`.
- `.obs`: `labels`, `labels2`, `ref_labels`, `species`.
- `.obsm["macrogenes"]`: float matrix `[n_cells, num_macrogenes]`.
- `.obs_names`: sorted species order output names.

## JAX Design Target
Future `train_saturn_jax.py` pretraining stage must:
1. Load normalized manifest and CPU-preprocessed arrays from `jax_saturn/data`.
2. Initialize `SaturnPretrainModule` params from centroid scores.
3. Create Optax Adam optimizer with `pretrain_lr`.
4. Create `PretrainTrainState`:
   - `params`
   - `opt_state`
   - `step`
   - `epoch`
   - `rng`
   - `species_names`
   - `config`
5. For every epoch:
   - Build deterministic per-epoch species permutation keys.
   - Iterate fixed-shape species-grouped batches.
   - Compute per-species ZINB loss and batch-level regularization losses.
   - Average or sum losses according to the PyTorch contract.
   - Apply gradients through Optax.
   - Record per-species average loss, L1, ranking loss, and total loss.
6. Save Orbax checkpoint at least once per epoch and at final pretrain.
7. Emit notebook-facing `adata_pretrain.h5ad`.

Jitted functions:
- `pretrain_step(state, batch) -> (state, metrics)`.
- `embed_pretrain_batch(params, batch, species_code) -> embeddings, macrogenes`.

The pretrain step must be compiled with fixed shapes. Species-specific gene counts may differ, so either:
- Compile one function per species gene shape, or
- Use padded species gene dimensions with masks.

For HMM v1, compile-per-species is acceptable because there are only three species.

Embedding-bank emission:
- Run model in eval mode with `train=False`.
- Iterate sorted species and local cell order.
- Write embeddings and macrogenes to host NumPy arrays.
- Preserve `obs_names`.

Pretrain checkpoint path:
- Public notebook compatibility path remains `shared/pretrain_model.pt` only if compatibility export is enabled.
- JAX native checkpoint path defaults to `shared/pretrain_orbax/`.
- `run_summary` or `config` metadata must point from public artifacts to JAX checkpoint path.

## Decisions
- Use Optax Adam with PyTorch-matching learning rate.
- Preserve current regularization placement: L1 and ranking loss are added once per mixed batch.
- Keep full ranking loss for HMM parity; sampled ranking can be introduced for large 25-species runs after parity.
- Use the current species one-hot parity behavior.
- Write `pretrain_losses.csv` with at least `epoch` and one column per species.

## Non-Goals
- Changing HVG, KMeans, or centroid score calculation during pretraining.
- Implementing VAE pretraining in v1.
- Scoring AnnData during pretraining. The notebook default has `score_adatas=False`.

## Acceptance Criteria
- Future tiny pretrain run writes an AnnData that passes the same artifact builder as PyTorch.
- Future HMM JAX pretrain writes `adata_pretrain.h5ad` with matching row count, embedding dimension, macrogene dimension, and obs schema.
- Future PyTorch-vs-JAX forward parity test passes before full HMM training is attempted.
- Resume from Orbax checkpoint yields the same next-step metrics as uninterrupted training for a deterministic tiny fixture.

## Open External Facts
- Whether checkpoint export to `.pt` is required for mixed PyTorch/JAX workflows during transition.
- Final location for JAX compilation cache on TPU VMs.

## GPU compilation evidence

Full HMM-shaped GPU profiling located an oversized monolithic step in XLA
priority-fusion analysis. The implementation compiles reconstruction gradients
per species and regularization separately, sums gradients at unchanged params,
and applies Adam once per mixed batch. Protein embeddings are runtime buffers.
The dropout/ranking key schedule and public artifacts are preserved. A full
HMM-shaped step compiled in 276.469 seconds and ran warm in 0.309 seconds.
CPU artifact, exact resume and fresh-process next-update tests pass after this
change. Full HMM scientific acceptance remains a separate verification gate.

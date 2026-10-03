# Label-Agnostic Training Spec

## Status
CPU InfoNCE, MMD and OT trainers implemented with strict label-free inputs,
reference CPU graphs, native checkpoint selection and exact resume tests.
Full HMM scientific parity and accelerator execution remain unverified.

## Notebook Scope
This spec covers notebook cells 13, 14, 16, 18, 20, 22, 24, 28, 32, and the report cells that compare label-free trials.

In scope source files:
- `scripts/prepare_label_agnostic_artifacts.py`
- `scripts/train_label_agnostic.py`
- `scripts/evaluate_label_agnostic_benchmark.py`
- `label_agnostic/artifacts.py`
- `label_agnostic/objectives.py`
- `label_agnostic/metrics.py`

## Current PyTorch Contract
The artifact builder creates:
- Strict label-free training artifact:
  - `embeddings`
  - `macrogenes`
  - `species`
  - `obs_ids`
- Frozen label-aware evaluation triplets:
  - `anchor`
  - `positive`
  - `negative`
  - `obs_ids`
- Metadata JSON with schema version, dimensions, species, seed, and evaluation triplet count.

The label-free trainer:
- Loads strict artifact and rejects unexpected keys.
- Factorizes species with `np.unique`.
- Loads metric model from pretrain checkpoint by copying encoder and layer norm weights.
- Builds preservation graph from pretrained embeddings.
- Builds cross-species positives only for `infonce`.
- Builds target species global indices and global-to-local maps.
- Estimates MMD bandwidth from fused pretrained embedding/macrogene view.
- Initializes embedding bank by embedding all macrogenes.
- Computes initial species mixing and teacher-neighborhood recall.
- Calibrates preservation weight from gradient norm ratio.
- For every epoch:
  - Rebuilds embedding bank.
  - Trains over shuffled label-free batches.
  - Computes objective-specific alignment loss.
  - Adds preservation distillation loss.
  - Adds within-species graph InfoNCE loss.
  - Updates model.
  - Computes epoch species mixing and teacher recall.
  - Selects best checkpoint by maximizing mixing subject to recall floor.
- Saves final selected model, final embeddings, metric history, and run summary.

Objective-specific current behavior:
- `infonce`: multi-positive cross-species InfoNCE over target species memory banks.
- `mmd`: multi-bandwidth pairwise species MMD.
- `ot`: partial OT alignment over species pairs using frozen teacher/macrogene cost.

## JAX Design Target
Future `scripts/train_label_agnostic_jax.py` must:
1. Load the same strict artifact.
2. Load JAX metric checkpoint from pretrain stage.
3. Build CPU-side graphs with NumPy/sklearn for v1:
   - preservation candidates and probabilities.
   - teacher neighbors.
   - cross-species positives for InfoNCE.
4. Transfer fixed graph arrays to device or shard/replicate them.
5. Build `LabelAgnosticTrainState`.
6. Compute initial embedding bank in eval mode.
7. Calibrate preservation weight using JAX gradients.
8. Train for `epochs`.
9. Select best checkpoint using the same label-free rule.
10. Write the same public artifacts as PyTorch.

Required train state:
- `params`
- `opt_state`
- `step`
- `epoch`
- `rng`
- `best_params`
- `best_epoch`
- `best_mixing`
- `best_recall`
- `preservation_weight`
- `config`

Required batch fields:
- `macrogenes`: float32 `[batch, num_macrogenes]`.
- `species_codes`: int32 `[batch]`.
- `global_indices`: int64 or int32 `[batch]`.
- `valid_mask`: bool `[batch]`.

Required metric history columns:
- `epoch`
- `metric_loss`
- `alignment_loss`
- `preservation_loss`
- `local_graph_loss`
- `mean_local_coverage_per_batch`
- `preservation_weight`
- `local_graph_weight`
- `mean_coverage_per_batch`
- `species_mixing_fraction`
- `teacher_top15_recall_at_50`
- `checkpoint_feasible`
- `checkpoint_selected`
- `objective`

Required `run_summary.json` fields:
- Preserve all current fields from `scripts/train_label_agnostic.py`.
- Add `implementation: "jax"`.
- Add `orbax_checkpoint_path`.
- Add `jax_devices` and `global_batch_size`.

Embedding bank:
- For HMM, full embedding bank `[n_cells, model_dim]` may be replicated on each device.
- For 25-species high-dimensional runs, spec allows sharded embedding bank only after HMM parity.
- Bank is rebuilt at epoch start, matching current PyTorch behavior.

Checkpoint selection:
- Feasible checkpoint if `teacher_top15_recall_at_50 >= preservation_target`.
- Selected checkpoint if feasible and `species_mixing_fraction > best_mixing`.
- If no epoch beats initial feasible value, selected epoch remains `0` and final params are initial params.

## Decisions
- Do not let label arrays enter the JAX label-free trainer.
- Keep graph construction CPU-side for v1.
- Preserve the current preservation-weight calibration rule.
- Keep the evaluator post-hoc and label-aware; do not merge it into training.
- Use the same strict artifact keys as the current PyTorch path.

## Non-Goals
- Porting phase-2 objectives from `phase2_train.py`.
- Porting adversarial or online MNN label-free objectives from `train-saturn.py`.
- Replacing sklearn nearest-neighbor graph construction with FAISS or TPU KNN in v1.
- Changing checkpoint selection criteria.

## Acceptance Criteria
- Future JAX label-free trainer never loads labels or evaluation triplets.
- `artifact_keys_seen_by_trainer` remains exactly `embeddings, macrogenes, obs_ids, species`.
- Future `infonce`, `mmd`, and `ot` runs write `final_embeddings.npz`, `metric_history.csv`, and `run_summary.json` with current schemas.
- The existing evaluator can compare JAX label-free trials to the JAX or PyTorch baseline.

## Open External Facts
- Whether HMM TPU memory permits fully replicated embedding banks at larger `num_macrogenes`.
- Whether 25-species InfoNCE needs sharded target banks in the first large run.

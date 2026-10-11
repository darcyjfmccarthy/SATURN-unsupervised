# Contracts Spec

## Status
Artifact validation, Phase 4 pretraining CLI and Orbax epoch save/resume
implemented. Metric/label-free CLIs and full notebook switching remain pending.

## Notebook Scope
This spec covers notebook cells 2, 7, 13, 16, 18, 20, 22, 24, 28, and all report cells that consume outputs from the benchmark directory.

In scope scripts:
- `scripts/human_monkey_mouse.sh`
- Future `scripts/human_monkey_mouse_jax.sh`
- `train-saturn.py`
- Future `scripts/train_saturn_jax.py`
- `scripts/train_label_agnostic.py`
- Future `scripts/train_label_agnostic_jax.py`
- `scripts/prepare_label_agnostic_artifacts.py`
- `scripts/evaluate_label_agnostic_benchmark.py`

## Current PyTorch Contract
`human_monkey_mouse.ipynb` controls the run through environment variables passed to `scripts/human_monkey_mouse.sh`.

Current environment variables used by the notebook baseline stage:
- `SEED`
- `DEVICE`
- `DEVICE_NUM`
- `WORK_DIR`
- `CENTROIDS_INIT_PATH`
- `PRETRAIN_MODEL_PATH`
- `METRIC_MODEL_PATH`
- `PRETRAIN`
- `METRIC_EPOCHS`
- `METRIC_BATCH_SIZE`
- `POLLING_FREQ`

`scripts/human_monkey_mouse.sh` also defines:
- `IN_DATA`
- `HV_GENES`
- `NUM_MACROGENES`
- `MODEL_DIM`
- `HIDDEN_DIM`
- `PRETRAIN_EPOCHS`
- `PRETRAIN_BATCH_SIZE`
- `PRETRAIN_LR`
- `METRIC_LR`
- `PE_SIM_PENALTY`
- `L1_PENALTY`
- `CENTROID_SCORE_FUNC`

Current `train-saturn.py` notebook-relevant arguments:
- `--in_data`
- `--work_dir`
- `--device`
- `--device_num`
- `--seed`
- `--ref_label_col`
- `--hv_genes`
- `--num_macrogenes`
- `--model_dim`
- `--hidden_dim`
- `--pretrain_epochs`
- `--epochs`
- `--pretrain_batch_size`
- `--batch_size`
- `--pretrain_lr`
- `--metric_lr`
- `--pretrain`
- `--pretrain_model_path`
- `--metric_model_path`
- `--polling_freq`
- `--embedding_model`
- `--centroid_score_func`
- `--centroids_init_path`
- `--pe_sim_penalty`
- `--l1_penalty`

Current `scripts/train_label_agnostic.py` notebook-relevant arguments:
- `--objective` in `{infonce,mmd,ot}`
- `--artifact`
- `--pretrain-checkpoint`
- `--output-dir`
- `--device`
- `--device-num`
- `--seed`
- `--epochs`
- `--batch-size`

Current AnnData output contract:
- `.X`: float embedding matrix with shape `[n_cells, model_dim]`.
- `.obs_names`: stable observation IDs in the same order as pretrain truth.
- `.obs["labels"]`: categorical species-prefixed labels.
- `.obs["labels2"]`: categorical original fine labels without species prefix.
- `.obs["ref_labels"]`: categorical reference labels.
- `.obs["species"]`: categorical species name.
- `.obsm["macrogenes"]`: float macrogene matrix with shape `[n_cells, num_macrogenes]`.
- Evaluated AnnData also contains `.obsm["X_umap"]` with shape `[n_cells, 2]`.

Current strict label-free artifact schema:
- `embeddings`: float32, shape `[n_cells, model_dim]`.
- `macrogenes`: float32, shape `[n_cells, num_macrogenes]`.
- `species`: string, shape `[n_cells]`.
- `obs_ids`: string, shape `[n_cells]`.

Current evaluation triplet artifact schema:
- `anchor`: int64, shape `[n_triplets]`.
- `positive`: int64, shape `[n_triplets]`.
- `negative`: int64, shape `[n_triplets]`.
- `obs_ids`: string, shape `[n_cells]`.

Current label-free final embedding artifact schema:
- `embeddings`: float32, shape `[n_cells, model_dim]`.
- `species`: string, shape `[n_cells]`.
- `obs_ids`: string, shape `[n_cells]`.

Current metric history CSV minimum notebook contract:
- `epoch`: integer epoch, expected to be exactly `1..METRIC_EPOCHS`.
- `metric_loss`: float objective loss.

Current label-free `run_summary.json` notebook contract:
- `objective`
- `label_free`
- `artifact_keys_seen_by_trainer`
- `selected_epoch`
- `selected_species_mixing_fraction`
- `selected_teacher_top15_recall_at_50`
- `selection_uses_labels`
- `epochs`
- `seed`

`selected_epoch` may be 0 when the initial teacher remains the best model.
Reference trainer diagnostics are permitted alongside these minimum fields.
Phase 1 validators recognize the diagnostic keys emitted by
`scripts/train_label_agnostic.py` and reject unrecognized keys.

Current shared evaluator outputs:
- `comparison.csv`
- `acceptance.json`
- `{trial}/metrics.json`
- `{trial}/history.csv`
- `{trial}/evaluated_adata.h5ad`
- `{trial}/umap.npz`

## JAX Design Target
Future `scripts/human_monkey_mouse_jax.sh` must accept the same environment variables as `scripts/human_monkey_mouse.sh` and add only JAX-specific optional variables:
- `JAX_PLATFORM_NAME`
- `JAX_ENABLE_X64`
- `XLA_FLAGS`
- `JAX_COMPILATION_CACHE_DIR`
- `ORBAX_CHECKPOINT_DIR`
- `TPU_TOPOLOGY`
- `GLOBAL_BATCH_SIZE`
- `PER_DEVICE_BATCH_SIZE`
- `MIXED_PRECISION`
- `RESUME`
- `KEEP_PYTORCH_COMPAT_CHECKPOINT`

Future `scripts/train_saturn_jax.py` must mirror notebook-used `train-saturn.py` arguments. It may reject out-of-scope options with a clear error if they are not touched by the notebook.

Future `scripts/train_label_agnostic_jax.py` must mirror notebook-used `scripts/train_label_agnostic.py` arguments for `infonce`, `mmd`, and `ot`.

Checkpoint contracts:
- JAX-native checkpoints use Orbax under a directory, not `.pt`, with this layout:
  - `metadata.json`
  - `params/`
  - `opt_state/`
  - `rng/`
  - `training_state/`
- `metadata.json` must include:
  - `schema_version`
  - `implementation: "jax"`
  - `model_kind`
  - `epoch`
  - `step`
  - `seed`
  - `hyperparameters`
  - `species_names`
  - `input_shapes`
  - `source_manifest_sha256`
  - `git_commit` if available
- `.pt` files are not required for JAX execution. If `KEEP_PYTORCH_COMPAT_CHECKPOINT=1`, a future converter may emit a PyTorch-shaped state dict for comparison only.

The future JAX scripts must write the same notebook-facing `.h5ad`, `.npz`, `.csv`, and `.json` files as the PyTorch path. This is the public compatibility boundary.

## Decisions
- Keep the existing output directory structure so notebook cells 24 through 38 remain unchanged.
- Treat Orbax checkpoint directories as internal implementation artifacts and `.h5ad`/`.npz` files as public artifacts.
- Preserve strict label-free artifact keys exactly.
- Use JSON metadata sidecars for any new JAX cache formats.
- Allow additional diagnostic columns in CSV outputs, but never remove or rename notebook-used columns.
- CLI defaults must match the PyTorch notebook defaults unless a spec explicitly records a JAX-specific default.

## Non-Goals
- Supporting every `train-saturn.py` argument in v1.
- Changing the evaluator schema.
- Replacing `scripts/prepare_label_agnostic_artifacts.py` in v1.
- Making notebooks call JAX internals directly.

## Acceptance Criteria
- A future JAX HMM run can be evaluated by `scripts/evaluate_label_agnostic_benchmark.py` without modification.
- `human_monkey_mouse.ipynb` can switch from `scripts/human_monkey_mouse.sh` to `scripts/human_monkey_mouse_jax.sh` by changing the command only.
- All JSON and NPZ artifacts reject unexpected keys where the PyTorch path currently rejects them.
- Orbax resume can restore params, optimizer state, RNG state, epoch, and step.

## Open External Facts
- Whether compatibility `.pt` files are still needed after the notebook is switched to JAX outputs.
- Final GCS checkpoint path.
- Required checkpoint retention count and cleanup policy.

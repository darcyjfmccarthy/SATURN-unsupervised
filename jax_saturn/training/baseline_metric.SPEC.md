# Labeled Baseline Metric Training Spec

## Status
CPU baseline implemented with reference mining, Adam parity, padded batches,
AnnData/history output and native checkpoint resume tests. GPU/TPU and full HMM
scientific parity remain unverified.

## Notebook Scope
This spec covers the labeled baseline portion of notebook cell 7 and outputs consumed by cells 24, 28, 32, 33, 34, 35, 36, 37, and 38.

In scope source files:
- `train-saturn.py`
- `miners/triplet_margin_miner.py`
- `losses/triplet_margin_loss.py`
- `distances/cosine_similarity.py`
- `utils/loss_and_miner_utils.py`

## Current PyTorch Contract
After pretraining, `train-saturn.py` starts labeled metric learning:
- If `unfreeze_macrogenes=False`, replace species expression arrays with precomputed macrogenes.
- Build `SATURNMetricModel`.
- Copy `cl_layer_norm` and `encoder` from the pretrain model.
- Use `torch.optim.Adam` with `metric_lr`.
- Build a shuffled DataLoader over the metric dataset.
- Use cosine similarity distance.
- Use `TripletMarginMiner` with:
  - `margin=0.2`
  - `miner_type=args.metric_miner_type`, default `cross_species`
  - `type_of_triplets=semihard` under notebook defaults.
  - `mnn=True` under notebook defaults.
- Use `TripletMarginLoss` with margin `0.2`.

The active cross-species miner augments each species-prefixed label group with
MNN matches from other species, retaining duplicate matched indices. It samples
cross-species positives with replacement. Negatives exclude both anchor and
positive labels and may belong to either anchor or positive species. Preserve
these semantics when implementing Phase 5; do not substitute an exhaustive
positive enumeration or anchor-species-only negatives.
- Train epochs `1..epochs`.
- Save intermediate AnnData every `polling_freq` epochs.
- Save final metric model to `METRIC_MODEL_PATH`.
- Write `final_adata.h5ad`.
- Write `triplets.csv`, `epoch_scores.csv`, `celltype_id.pkl`, and `metric_history.csv`.

Notebook/evaluator requires:
- `baseline/saturn_results/final_adata.h5ad`
- `baseline/saturn_results/metric_history.csv`
- `baseline/final_model.pt`

Evaluator uses only final `metric_loss` from `metric_history.csv` and final embeddings from `final_adata.h5ad`.

## JAX Design Target
Future JAX baseline metric training must:
1. Load pretrain JAX checkpoint or copied pretrain encoder params.
2. Build `SaturnMetricModule`.
3. Create `MetricTrainState` with params, Optax Adam state, RNG, epoch, and step.
4. Use fixed-shape batches of macrogenes, labels, ref labels, species codes, global indices, and valid masks.
5. Mine cross-species MNN triplets per batch.
6. Apply semihard filtering.
7. Compute triplet margin loss.
8. Update model params.
9. Emit metric history and final AnnData in PyTorch-compatible schema.

Triplet mining implementation choices:
- HMM v1 may mine triplets in JAX eager code outside the main jitted loss if dynamic tuple sizes make jit awkward.
- The loss application must be jit-compatible with padded triplet arrays and `triplet_valid_mask`.
- Large-scale implementation may move mining into a jitted top-k or nearest-neighbor routine after parity.

Required batch fields:
- `macrogenes`: float32 `[batch, num_macrogenes]`.
- `labels`: int32 `[batch]`, species-prefixed label IDs.
- `ref_labels`: int32 `[batch]`.
- `species_codes`: int32 `[batch]`.
- `global_indices`: int64 or int32 `[batch]`.
- `valid_mask`: bool `[batch]`.

Required history columns:
- `epoch`
- `metric_loss`
- `cross_metric_loss`
- `cross_candidate_triplets`
- `cross_active_triplets`
- `cross_active_triplet_fraction`
- `metric_miner_type`
- `metric_triplet_type`

Additional diagnostic columns may be included as long as existing notebook/evaluator columns remain.

Final AnnData creation:
- Use the same helper contract as pretraining.
- `.X` is metric embeddings.
- `.obsm["macrogenes"]` is frozen macrogenes.
- `obs` and `obs_names` match pretrain truth order.

## Decisions
- Keep labeled baseline in scope because the notebook evaluates label-free trials against it.
- Preserve semihard cross-species MNN mining before optimizing.
- Allow dynamic mining outside jit for first HMM parity.
- Do not port local triplet regularization or divergence teacher penalties for v1 because notebook defaults do not use them.

## Non-Goals
- Porting all miner types.
- Porting `equalize_triplets_species`.
- Porting local triplet regularizer for first parity.
- Changing checkpoint selection for baseline. Baseline uses final epoch.

## Acceptance Criteria
- Future JAX baseline produces `metric_history.csv` with epochs exactly `1..METRIC_EPOCHS`.
- Future JAX baseline final AnnData can be loaded by `scripts/evaluate_label_agnostic_benchmark.py`.
- Future small triplet mining fixture matches PyTorch candidate and semihard filtering semantics.
- Future HMM JAX baseline metrics satisfy evaluator acceptance thresholds relative to PyTorch reference.

## Open External Facts
- Whether first TPU baseline run should mine triplets on host CPU for simplicity or on TPU for speed.
- Maximum acceptable host-device transfer overhead for HMM and 25-species runs.

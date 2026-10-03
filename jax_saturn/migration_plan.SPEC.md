# Migration Plan Spec

## Status
Implementation authorized. Phases 0–6 implemented on CPU. Phases 7–10 pending.

## Notebook Scope
This plan covers the migration of the execution path used by `notebooks/human_monkey_mouse.ipynb`. It does not cover legacy vignettes, unused loss families, or phase-2 experiments outside the defended notebook.

## Current PyTorch Contract
Current state:
- PyTorch implementation is the reference.
- Defended notebook runs end to end on CUDA.
- Label-free benchmark outputs are already standardized enough to support a parallel JAX implementation.
- No existing docs/spec convention exists in the repository.

## JAX Design Target
Migration proceeds in phases after this SDD tree is reviewed.

Phase 0: Specs.
- Create this `jax_saturn/` Markdown spec tree.
- No code.
- Review architecture and contracts.

Phase 1: Data contracts and fixtures.
- Implement manifest normalization.
- Implement `.pt` gene embedding cache conversion.
- Implement centroid cache import/export.
- Add schema validators for AnnData, NPZ, CSV, JSON, and Orbax metadata.
- Add tiny fixtures for unit tests.

Phase 1 implementation: `data/manifest.py`, `data/cache.py`,
`contracts/validation.py`, and `verification/test_contracts.py`. Usage and limits
are documented in `README.md`. Checkpoint validators cover metadata, not Orbax
state restoration. AnnData preprocessing and batching are implemented in Phase 4.

Phase 2: Flax model parity.
- Implement `FullBlock`, `SaturnPretrainModule`, and `SaturnMetricModule`.
- Implement PyTorch-to-JAX parameter mapping for reference tests.
- Add forward parity tests on small HMM batches.

Phase 2 implementation: `models/saturn.py`, `models/conversion.py` and
`verification/test_models.py`. Three-species HMM-shaped fixtures check all
outputs, singleton batches, copied encoder transfer, JIT, dropout RNG and
parameter gradients against the actual PyTorch model. Full HMM parity remains
Phase 7; Orbax restoration remains a training-phase requirement.

Phase 3: Loss parity.
- Implement ZINB, L1, ranking loss, triplet margin loss, preservation, InfoNCE, MMD, and partial OT.
- Add deterministic parity tests against PyTorch functions.

Phase 3 implementation: `losses/core.py`, `losses/objectives.py` and
`verification/test_losses.py`. Value and gradient parity use the repository
PyTorch losses and installed scvi ZINB. Tests cover padding, empty candidates,
triplet filtering, Sinkhorn plans and frozen OT teachers. Cross-species MNN
candidate construction remains Phase 5.

Phase 4: CPU/GPU JAX pretraining.
- Implement `train_saturn_jax.py` pretrain path.
- Emit `adata_pretrain.h5ad`.
- Verify artifact builder accepts the output.

Phase 4 implementation: `data/preparation.py`, `data/batching.py`,
`training/pretrain.py`, `distributed/checkpoint.py`,
`scripts/train_saturn_jax.py`, and `verification/test_pretrain.py`. Tests verify
CPU preprocessing/KMeans against the reference, equal-species padded sampling,
one-epoch CLI output accepted by the existing artifact builder, exact epoch
resume and fresh-process restoration without Torch. GPU model/loss and small-fixture training/resume checks pass locally;
exact baseline GPU resume requires deterministic XLA operations. Use
`--epochs 0` for pretraining only, or positive epochs for the labeled baseline.

Phase 5: CPU/GPU JAX labeled baseline.
- Implement cross-species MNN mining and metric training.
- Emit `final_adata.h5ad` and `metric_history.csv`.
- Run evaluator against tiny output.

Phase 5 implementation: `training/mining.py`, `training/baseline.py` and
`verification/test_baseline.py`. CPU tests verify reference candidate sampling,
semihard filtering, Adam updates, exact resume and public AnnData/history output.
`scripts/human_monkey_mouse_jax.sh` exposes the notebook environment arguments
and optional PyTorch checkpoint exports.

Phase 6: CPU/GPU JAX label-free objectives.
- Implement `train_label_agnostic_jax.py`.
- Run `infonce`, `mmd`, and `ot` on tiny data.
- Verify strict label-free contract.

Phase 6 implementation: `data/graphs.py`, `training/label_agnostic.py` and
`scripts/train_label_agnostic_jax.py`. All three objectives run on synthetic and
repository tiny data. Tests verify strict input rejection, CPU graph parity,
adaptive-weight resume and checkpoint selection including epoch zero.
`scripts/run_label_agnostic_benchmark_jax.sh` connects native checkpoints to
the existing artifact builder and evaluator. Full scientific parity remains
a Phase 7 requirement.

Phase 7: HMM parity.
- Run bounded HMM implementation checks: two epochs by default, matched
  starting weights/batches, numerical updates, output contracts and resume.
- Reuse shared pretraining for baseline and all three label-free objectives.
- Compare scientific metrics only at matched training horizons. Full 20/30-epoch
  evaluation is optional and explicitly selected, not a gate for implementation.
- Freeze public artifact schema.

Phase 7 progress: matching CUDA runtime configured in an isolated directory;
GPU model/loss parity and small-fixture training/resume checks verified. Full
HMM pretraining/baseline launched at notebook settings.
`scripts/verify_saturn_jax_parity.py` audits completed settings/histories and
evaluates candidate embeddings on the reference truth and frozen triplets.
Reference agreement and benchmark acceptance are separate required gates;
neither is established for the full JAX HMM run yet. The user subsequently
requested bounded runs: shared 20-epoch pretraining is preserved, the long
baseline was stopped, and no full label-free runs were launched. Shell runners
now default to two epochs; `VALIDATION_PROFILE=full` opts into notebook lengths.
Bounded pretraining and all three label-free Adam trajectories now match the
PyTorch reference across three updates on CPU and local CUDA. The baseline
three-update comparison passes on CPU. Weight conversion takes independent
snapshots after the bounded test exposed CPU source-buffer aliasing. Full-atlas
matched short-horizon evaluation and TPU execution remain unverified.

Real HMM baseline updates now have a bounded verifier using the reference
artifact, pretraining checkpoint and frozen triplets at the defended dimensions
(200 macrogenes, hidden/model dimension 256, batch 512). Three CPU updates
passed loss/gradient/parameter comparisons across 1,516 sampled real cells.
This does not establish complete matched epochs or scientific acceptance.
The same real-data GPU check failed one first-update encoder parameter against
the CPU PyTorch reference (2.8039e-5 maximum absolute discrepancy); losses and
gradient norm passed. No tolerances were relaxed. GPU real-data parity remains
pending investigation.

The failure persists with both frameworks on CUDA and is amplified by Adam
at a tiny gradient. Disabling GPU `priority-fusion` now passes all three real
HMM baseline updates at unchanged tolerances and cuts the first update including
compile to 6.424s (following updates about 0.1s). This is verified with explicit
compiler flags; default-fusion real-data GPU parity remains unestablished.

A real-count pretraining verifier now checks sampled atlas rows in cached HV
gene order. It exposed saturated softplus backward differences in ZINB; the
loss now matches PyTorch's threshold/backward rule and all 22 targeted loss,
update, single-device integration and two-device pretraining tests pass.
Real HMM pretraining still exceeds strict parameter tolerances in two entries
from fresh initialization and ten entries in the trained-weight CUDA stress
case. This remains an open numerical gate, with failure receipts preserved.

Phase 8: TPU pilot.
- Add TPU runbook or scripts.
- Run one-step smoke on Tokyo v6e or verified fallback.
- Prove Orbax resume.
- Run tiny benchmark on TPU.

Phase 8 progress: single-host baseline pmap is implemented in
`distributed/metric.py` and exposed through `--metric-distributed`. Three
two-device CPU checks verify global triplets/Adam parity, identical preview and
update dropout, padded output and checkpoint resume. The five existing baseline
tests still pass (combined run: 28.168 seconds). Single-host pretraining pmap
is now implemented in `distributed/pretrain.py`, with unequal-shard valid-count
and once-only regularizer/Adam parity, absent species, singleton cells and
exact epoch resume. All five two-device checks pass (43.899 seconds), and the
five single-device pretraining checks still pass (76.165 seconds). A one-epoch
CLI smoke ran both distributed stages and verified 68 public rows and native
checkpoint topology metadata. Single-host label-free pmap now gathers global
embeddings and metadata for InfoNCE/MMD/OT and uses the same global objective for
calibration. Two additional CPU checks passed (77.689 seconds), and seven
existing label-free integration/PyTorch update tests still pass (88.266 seconds).
A one-epoch distributed label-free CLI smoke verified 16 unpadded rows and
topology metadata. Distributed inference now supports pretrain exports,
baseline artifacts and label-free embedding banks/selected outputs, with cached
static kernels and dynamic weights. All nine checks inside three two-device
CPU subprocesses passed in 150.739 seconds after inference integration.
Mixed-precision training is now exposed in library configs and both CLIs/shells.
One-epoch CPU checks cover all five stages on one and two devices, fp32 state
and outputs, and precision-change rejection on resume. Eleven existing fp32
baseline/label-free tests still pass. A CLI smoke reused an fp32 checkpoint
without retraining and ran a bf16 baseline. Actual TPU execution and accelerator
bf16 scientific parity remain pending.

The TPU runbook and synthetic hardware smoke command are implemented. Initial
and fresh-process resume together perform only three updates. The smoke test
passed on two CPU devices with bf16 in 48.123 seconds, including optimizer/RNG
resume and immutable receipts. Actual TPU execution awaits project or VM details
and a spending limit; the candidate TPU dependency stack is not hardware-tested.
No cloud resources were created.

Phase 9: 25-species pilot.
- Use `data/datatable.csv`.
- Start with conservative dimensions.
- Profile memory, compile time, and epoch time.

Phase 9 preflight: `scripts/preflight_saturn_jax.py` inspects all atlas shapes
without loading expression matrices and reports every missing input. The real
25-species manifest contains 212,189 cells across 25 readable atlases, but
embeddings are absent for a_thaliana, n_vectensis, s_mediterranea, s_pistillata
and z_mays. No species were dropped. Two preflight tests pass, covering dense
and sparse metadata, missing files, malformed atlases and dimension checks.
Actual 25-species training and measured accelerator profiling remain pending.

Phase 10: High-dimensional scaling.
- Increase `num_macrogenes`, `hv_genes`, and batch size only after profiling.
- Consider sharded embedding banks, sampled ranking loss, and `pjit` if needed.

## Decisions
- Do not merge code phases until their upstream spec exists and is accepted.
- Keep PyTorch path available until JAX HMM parity is accepted.
- Treat TPU spending as gated by tiny and resume tests.
- Prioritize notebook-compatible artifacts over internal code elegance.

## Non-Goals
- Big-bang rewrite of the entire repository.
- Removing PyTorch implementation.
- Rewriting notebooks before JAX output compatibility exists.
- Reserving TPU capacity before code passes smoke tests.

## Acceptance Criteria
- This SDD phase is complete when all listed spec files exist and are internally consistent.
- Later code phases must cite the relevant spec section in PR or commit notes.
- A future implementation phase should not need to decide public artifact schemas or TPU region strategy.

## Open External Facts
- Who reviews and signs off on each spec.
- Whether implementation should happen in one branch or stacked branches.
- Whether TPU cloud scripts belong in this repo or a private operations repo.

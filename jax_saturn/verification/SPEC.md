# Verification Spec

## Status
Phases 1–6 static contracts, model/loss CPU parity, baseline mining and all
label-free objective integrations are implemented with checkpoint resume tests.
Full HMM scientific acceptance and TPU execution remain pending.

## Notebook Scope
This spec covers future tests for notebook cells 7, 13, 16, 18, 20, 22, 24, and report validation cells 28 through 38.

In scope reference outputs:
- Existing PyTorch tiny and HMM benchmark artifacts under `out/`.
- Future JAX benchmark artifacts under a separate output directory.

## Current PyTorch Contract
The existing verification path is mostly notebook and evaluator based:
- Cell 14 audits strict label-free artifact keys.
- Cell 24 runs shared post-hoc evaluation.
- Cell 28 checks all histories contain epochs `1..METRIC_EPOCHS`.
- `scripts/evaluate_label_agnostic_benchmark.py` writes:
  - `comparison.csv`
  - `acceptance.json`
  - per-trial metrics JSON.
- Acceptance thresholds are relative to baseline:
  - fixed triplet margin loss no more than baseline plus `0.02`.
  - label same-neighbor fraction at least baseline minus `0.05`.
  - species mixing fraction at least baseline minus `0.03`.

## JAX Design Target
Future verification layers:

1. Static contract tests.
   - Manifest normalization.
   - Artifact schema validation.
   - Checkpoint metadata validation.
   - No label arrays in label-free trainer batches.

2. Unit parity tests.
   - ZINB log-prob against scvi/PyTorch.
   - `SaturnPretrainModule` forward with copied weights against PyTorch.
   - `SaturnMetricModule` forward with copied weights against PyTorch.
   - Cosine triplet mining/filtering against PyTorch on fixtures.
   - InfoNCE, MMD, and partial OT against PyTorch on fixtures.
   - Three consecutive production Adam updates for pretraining, baseline and
     each label-free objective, with copied weights and matched random inputs.
   - Padded rows must not affect losses, gradient norms or parameter updates.

3. Tiny integration tests.
   - Use `data/saturn_run_tiny.csv`.
   - Run 1 or 2 epochs.
   - Assert output schemas and finite losses.
   - Assert evaluator completes.

4. Bounded HMM implementation parity.
   - Use `data/human_monkey_mouse.csv`.
   - Match notebook hyperparameters.
   - Default to two epochs of shared pretraining, then two epochs each of
     baseline, `infonce`, `mmd`, and `ot`; reuse the shared pretraining checkpoint.
   - Compare PyTorch and JAX at the same short horizon, using copied starting
     weights and matched batches/random inputs for numerical update checks.
   - Check losses, gradients, parameter updates, finite outputs and resume.
   - Run existing evaluator for diagnostics. Do not compare a short candidate
     to the completed 20/30-epoch reference as evidence of implementation parity.
   - Full scientific evaluation is a separate opt-in run using
     `VALIDATION_PROFILE=full`; it is not required for implementation progress.

5. TPU smoke/resume.
   - Single step compile and train.
   - Save Orbax checkpoint.
   - Restart process.
   - Resume and complete next step.
   - Verify no padded rows in artifacts.

6. 25-species pilot.
   - Use `data/datatable.csv`.
   - Start with `hv_genes=2000`, `num_macrogenes=512`, `model_dim=256`.
   - Verify memory, compile time, and artifact schema.

Phase 1 test command: `python -m unittest discover -s jax_saturn/verification -p 'test_*.py' -v`.

## Decisions
- Use metric parity rather than bitwise parity.
- Tiny tests gate TPU spending.
- Bounded HMM implementation parity gates 25-species scaling.
- Existing evaluator remains the top-level scientific acceptance check.
- Tests must distinguish CPU preprocessing failures from JAX/TPU training failures.

## Non-Goals
- Running full HMM or TPU training during Phase 1 contract verification.
- Requiring exact same UMAP coordinates across PyTorch and JAX.
- Treating loss curves as exact parity targets.
- Adding CI for TPU in v1.

## Acceptance Criteria
- Implementation parity requires matched short-run numerical checks and output
  contracts. Full HMM scientific acceptance requires the existing evaluator to
  succeed and must be reported separately; short runs do not establish it.
- Future tests catch schema drift in public artifacts.
- Future TPU smoke proves resume before any long Spot/Flex-start run.
- Future 25-species pilot records memory and runtime diagnostics.

## Open External Facts
- Final numeric tolerances for unit-level parity.
- Whether CI runners will have the `saturn` conda environment.
- Whether TPU smoke tests can be automated or must remain manual due to quota/cost.

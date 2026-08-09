# Migration Plan Spec

## Status
Draft.

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

Phase 2: Flax model parity.
- Implement `FullBlock`, `SaturnPretrainModule`, and `SaturnMetricModule`.
- Implement PyTorch-to-JAX parameter mapping for reference tests.
- Add forward parity tests on small HMM batches.

Phase 3: Loss parity.
- Implement ZINB, L1, ranking loss, triplet margin loss, preservation, InfoNCE, MMD, and partial OT.
- Add deterministic parity tests against PyTorch functions.

Phase 4: CPU/GPU JAX pretraining.
- Implement `train_saturn_jax.py` pretrain path.
- Emit `adata_pretrain.h5ad`.
- Verify artifact builder accepts the output.

Phase 5: CPU/GPU JAX labeled baseline.
- Implement cross-species MNN mining and metric training.
- Emit `final_adata.h5ad` and `metric_history.csv`.
- Run evaluator against tiny output.

Phase 6: CPU/GPU JAX label-free objectives.
- Implement `train_label_agnostic_jax.py`.
- Run `infonce`, `mmd`, and `ot` on tiny data.
- Verify strict label-free contract.

Phase 7: HMM parity.
- Run full HMM locally if hardware permits, otherwise on on-demand TPU.
- Compare with PyTorch reference using existing evaluator.
- Freeze public artifact schema.

Phase 8: TPU pilot.
- Add TPU runbook or scripts.
- Run one-step smoke on Tokyo v6e or verified fallback.
- Prove Orbax resume.
- Run tiny benchmark on TPU.

Phase 9: 25-species pilot.
- Use `data/datatable.csv`.
- Start with conservative dimensions.
- Profile memory, compile time, and epoch time.

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

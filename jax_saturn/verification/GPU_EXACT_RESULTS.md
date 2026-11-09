# Local GPU-exact continuation validation

Validation date: 2026-10-17. Device: RTX 4060 Laptop GPU, 8 GB, driver 566.24.
JAX/jaxlib/CUDA plugin 0.6.2. All work was local; no A100 restart, remote
deployment or cloud training/benchmark was performed.

## Deployment packaging correction

The initial archive filter excluded every directory named `data`, inadvertently
omitting the runtime modules in `jax_saturn/data/`. That archive must be replaced.
The corrected packager excludes biological data only at the repository root,
requires the native runtime modules in both packaging and installed-tree
verification, and passes two deployment regression checks. These verify every
Python file under `jax_saturn/` is included and reject the old incomplete manifest.
Imports of `NeighborSearch` and `WorkCache` also pass from a fresh extraction.
Rebuild/retransfer the archive before running cloud preflight; no environment or
foundation retraining is required for this correction.

The isolated workspace venv inherits the existing `saturn` environment read-only
and installs CUDA plugin/PJRT 0.6.2 and SciPy 1.15.3 locally. CUDA compiler 12.6.85
is pinned for this driver's CUDA 12.7 support; retained CUDA library directories
are linked into the isolated overlay. The base conda environment was unchanged.
Exact package identities are in the [execution receipt](receipts/gpu_exact_execution.json).

## Required gates

The final test summary and log hashes are recorded in the execution receipt.
The real-data [accuracy receipt](receipts/gpu_exact_real.json) uses the downloaded
original JAX foundation, not a replacement PyTorch run.

| Check | Evidence |
|---|---|
| Foundation reuse | Local GPU preflight validates epoch-20 pretraining, epoch-30 baseline, all 148,727 observations, 300/384/384 dimensions, seed/batch/precision, file hashes, artifact/truth alignment and frozen triplets; neither completed trainer is called |
| Non-tied CPU/GPU neighbors | Exact indices; distance tolerance 1e-5; multiple query/reference block boundaries |
| Ties / duplicates / zeros / small species | Explicit self exclusion, ascending reference-index ties, chunk-invariant results and zero-width neighbor result |
| Complete-population real queries | 64 queries against all 148,727 references plus 32 queries per species against that complete species population, k=50 |
| Real numerical differences | Maximum distance error 7.16e-7; two within-species tied/roundoff ordering rows reported separately; no non-tied disagreement |
| Representative metrics | 400 real cells/species with original labels2; all seven metric differences zero, within tolerance 1e-4 |
| Preservation negatives | Exact seeded draw/graph parity against the original numpy/sklearn implementation across three seeds; no species-sized pool rebuilt per anchor |
| Cache behavior | Cached/uncached graphs agree; preservation/bandwidth reused across objectives; corrupt and incomplete entries rejected |
| Label leakage | Strict artifact contract rejects labels; training/selection sees exactly four label-free fields |
| Objectives and resume | Finite InfoNCE/MMD/OT outputs; 2→4 epoch resume matches uninterrupted parameters, optimizer, RNG, selected state, diagnostics and adaptive weight under the same backend |
| Completion | Epoch-2 exports cannot satisfy a 30-epoch trial; summaries, histories, selected state and checksum completion records are checked |
| Orchestration | One launch orders 2/2/2 then 30/30/30, resumes at 3, skips completed stages/epochs, preserves checkpoints when the time gate fails, invokes no cloud command |
| Notebook recovery | Streams/errors forwarded; completed-cell snapshot and failing-cell traceback retained; child processes cleaned up |
| Static report cache | Repeated pretrained metrics/UMAP identical; original source h5ad checksum unchanged |
| Documentation/package | Shell examples parse, CLI help checks pass, notebook code compiles; archive includes new source/docs/tests and excludes data/runs/environments/credentials |

The complete existing CPU verification suite and the final GPU test results are
listed with exact counts in the execution receipt. A first restricted-sandbox CPU
attempt stalled during checkpoint work; the final CPU suite ran locally outside
that process sandbox with `JAX_PLATFORMS=cpu`.

## Reproduce locally

Use an environment containing the retained SATURN dependencies plus JAX CUDA
0.6.2, nbclient 0.10.4 and psutil. For this laptop, retain CUDA 12.6 libraries and
compiler rather than installing a compiler newer than its driver supports.

```bash
export NUMBA_CACHE_DIR=/tmp/saturn-validation-numba
export MPLCONFIGDIR=/tmp/saturn-validation-mpl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
JAX_PLATFORMS=cpu python -m unittest discover -v \
  -s jax_saturn/verification -p 'test_*.py'
SATURN_TEST_NEIGHBOR_DEVICE=gpu python -m unittest -v \
  jax_saturn.verification.test_gpu_neighbors
JAX_PLATFORMS=cpu python -m unittest -v \
  jax_saturn.verification.test_gpu_runner \
  jax_saturn.verification.test_notebook_execution
python scripts/validate_gpu_exact_real.py
python scripts/run_hmmff_gpu.py preflight \
  --output out/gpu_exact_validation/preflight_output
python scripts/check_saturn_workflow_docs.py
python scripts/package_saturn_source.py
```

These accuracy/unit checks are local validation, not a cloud throughput benchmark.
The production run obtains its runtime estimate from useful checkpointed epochs.

## Scope and limits

The A100 production experiment has not been run with these changes. Thirty-epoch
scientific acceptance, cloud throughput, the timing gate's eventual outcome and
final remote backup remain later user-run stages. The time gate uses measured
production epoch/update/search-compilation timings plus declared setup/final-work
reserves; it is an estimate, and the six-hour ceiling remains authoritative.

Historical sklearn self handling dropped its first neighbor, which can differ
from explicit self exclusion on tied/zero vectors. GPU tie behavior is declared
and tested; bitwise historical equivalence is asserted on non-tied fixtures.
Small numerical differences can affect near ties and selected epochs in a full
run even when local metric tolerances pass. Execution success and scientific
acceptance must be recorded separately.

Current raw input-file hashes were captured locally. The inherited foundation
lacks original fit-time raw-file hashes; its manifest hash alone cannot prove
historical raw contents. Saved foundation contents are independently protected
by their verified inventory. The unchanged-input limitation remains explicit in
the experiment record. Unseen-species/frozen held-out transform workflows remain
unsupported, as documented in the [reuse guide](../EXPERIMENT_WORKFLOW.md).

# Verification evidence

Verified on 2026-10-02 in the existing `saturn` environment, with CPU JAX 0.6.2,
Flax 0.10.7, Optax 0.2.8 and Orbax 0.11.36.

## Phases 1–6

All 54 tests passed: the 53-test core suite in 245.483 seconds and the
benchmark shell integration in 122.355 seconds:

```bash
NUMBA_CACHE_DIR=/tmp/jax-saturn-numba MPLCONFIGDIR=/tmp/jax-saturn-mpl \
  JAX_PLATFORMS=cpu python -m unittest discover \
  -s jax_saturn/verification -p 'test_*.py' -v
```

Coverage includes manifest schemas, portable cache round trips, strict public
artifacts, copied-weight forward and parameter gradient parity against the
actual PyTorch model, scvi ZINB and all active loss families, masked padding,
CPU HVG/KMeans reference parity, CLI output accepted by the existing artifact
builder, exact epoch resume and fresh-process next-step restoration without
importing Torch. CLI checkpoint inference does not add another training epoch.

Orbax tests ran outside the restricted Codex sandbox. An isolated diagnostic
showed its asyncio filesystem work hanging inside the sandbox and completing
outside it. The checkpoint implementation uses the normal Orbax API.

## Repository tiny manifest

One real-data pretraining epoch completed with reduced dimensions:

```bash
NUMBA_CACHE_DIR=/tmp/jax-saturn-numba MPLCONFIGDIR=/tmp/jax-saturn-mpl \
  JAX_PLATFORMS=cpu python scripts/train_saturn_jax.py \
  --in_data data/saturn_run_tiny.csv --device cpu \
  --work_dir /tmp/saturn-jax-smoke.4rKBpw --epochs 0 \
  --pretrain_epochs 1 --pretrain_batch_size 256 --hv_genes 64 \
  --num_macrogenes 8 --model_dim 16 --hidden_dim 16
```

The run completed 83 updates with finite epoch loss 453.029663. The pretrain
AnnData has shape `[11074, 16]` and macrogenes `[11074, 8]`. Observation IDs and
species order were checked directly against the source H5AD files, in sorted
manifest order:

| Species | Cells |
| --- | ---: |
| a_queenslandica | 804 |
| c_elegans | 5,275 |
| c_gigas | 2,207 |
| c_hemisphaerica | 2,788 |

The existing `scripts/prepare_label_agnostic_artifacts.py` accepted this AnnData
and generated a strict label-free artifact and 38,156 frozen evaluation
triplets, both validated by the new schema readers. The native epoch checkpoint
contains params, optimizer state, RNG, training history and validated metadata.
Smoke artifacts are in `/tmp` and may be removed by the environment.

## Baseline and label-free training

CPU tests additionally verify cross-species candidate/draw parity with the
reference miner, semihard filtering, a copied-weight Adam update, baseline
AnnData/history and exact epoch resume. Label-free tests compare CPU graphs and
selection metrics to the reference, exercise all three objectives, reject
labels, and prove adaptive-weight and earlier-best-epoch resume. Optional
PyTorch state exports round-trip exactly. Selected checkpoints cannot be used
as training resume checkpoints. The public benchmark shell passed end to end
on a 68-cell synthetic atlas, including optional PyTorch exports, all three
JAX objectives and the unchanged evaluator. Shell syntax and `git diff --check`
also passed.

One baseline metric epoch completed on the repository tiny manifest with
loss 0.100363, 52,098 candidates and 14,431 active triplets. The three label-free
one-epoch runs consumed the same strict artifact and native pretrain checkpoint.
The unchanged evaluator completed using all 38,156 frozen triplets:

| Trial | Objective loss | Species mixing | Label same-neighbor | Fixed triplet loss | Accepted |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline | 0.100363 | 0.120751 | 0.285185 | 0.064746 | yes |
| infonce | 7.741944 | 0.121763 | 0.284089 | 0.063114 | yes |
| mmd | 1.828130 | 0.130076 | 0.280748 | 0.097453 | no |
| ot | 1.986834 | 0.129095 | 0.277154 | 0.133546 | no |

InfoNCE passed the evaluator's three tolerances against this JAX baseline.
MMD and OT failed its fixed-triplet loss threshold; all other thresholds passed.
These reduced one-epoch runs establish integration, not full HMM parity.
Comparison outputs are under `/tmp/saturn-jax-benchmark.IW9N3y`; they may be
removed by the environment.

## GPU verification and full HMM run

JAX 0.6.2 CUDA plugin/PJRT and CUDA 12.6.85 assembler wheels were installed under
`/tmp/jax-saturn-cuda`, with the existing NVIDIA libraries made visible through
symlinks. The conda environment was not modified. JAX reports `CudaDevice(id=0)`
and executes compiled operations on the RTX 4060 Laptop GPU (8 GiB).

All 12 copied-weight model tests passed on GPU. Initial GPU loss tests exposed
contrastive dot-product precision errors of about 7.8e-5. Explicit
`Precision.HIGHEST` in loss dot products fixed these errors without relaxing the
assertions; all 12 loss tests then passed on GPU (58.341 seconds) and CPU
(25.637 seconds).

GPU integration verified all objective outputs, MMD adaptive-weight resume,
earlier selected-epoch resume, reference miner sampling/filtering, and the Adam
update. Default GPU reductions produced a baseline resume parameter difference
of 2.3283064e-10 in one element. With
`XLA_FLAGS=--xla_gpu_deterministic_ops=true`, the same unchanged baseline resume
test passed exact equality (32.855 seconds). GPU resume assertions were not
weakened.

Two additional reference-comparison tests passed. The parity command checks
histories/configuration and calls the existing evaluator using reference truth
and the same frozen triplets before comparing each trial. It requires both
reference agreement and benchmark acceptance.

Full HMM pretraining (20 epochs) and baseline training (30 epochs), using the
notebook dimensions and 512-cell batches, were launched under
`out/human_monkey_mouse_jax_parity`. Preprocessing completed. The original monolithic GPU step spent over 30
minutes compiling without producing a checkpoint. A native stack profile
located the delay in XLA `PriorityFusionQueue::ComputeAndSetPriorities`.
Passing proteins as a runtime buffer alone did not resolve the long compile;
neither did disabling Triton GEMM.

Per-species reconstruction gradient programs, a separate regularizer gradient
and one combined Adam update completed a full HMM-shaped step with real protein
vectors in 276.469 seconds, followed by a 0.309-second warm step. A diagnostic
that disabled `priority-fusion` compiled in 30.036 seconds but had a slower
0.382-second warm step; the full run uses the normal compiler passes. The five
CPU pretraining/artifact/resume tests passed after the change (89.987 seconds).
The original compilation was stopped with no checkpoint; caches were retained
and full training restarted in the benchmark layout.

Full HMM acceptance has not been established. The existing
PyTorch walkthrough comparison has 575,128 frozen evaluation triplets and no
passing label-free trial; reference agreement alone will not satisfy the gate.

## Remaining gates

### Distributed baseline

`distributed/metric.py` implements single-host baseline pmap with replicated
train state, batch shards, gathered global embeddings, global triplet loss and
averaged gradients. Global host mining and shard dropout use the same preview
as the update. Checkpoint and public artifact boundaries remove the replica
axis. Metadata records pmap execution and local device count; resume rejects
changed topology. CLI flags expose execution and an expected-device-count
check before preprocessing.

The subprocess test forces two CPU devices and requires all three inner checks
to run without skips: three Adam updates (including cross-shard triplets and an
empty loss), dropout preview/update consistency, and padded epoch training with
exact checkpoint resume. The wrapper and five existing single-device baseline
tests passed in 28.168 seconds. A separate CLI smoke completed one pretraining
epoch and two distributed baseline epochs on a synthetic 68-cell atlas, wrote
all 68 final rows with five embedding dimensions, and recorded pmap/two devices
in the epoch-two checkpoint. An incorrect expected device count exited before
reading a deliberately nonexistent input manifest.

### Distributed pretraining

`distributed/pretrain.py` implements separate per-species gradient programs
with global valid-cell denominators. Replica gradient means recover the global
reconstruction average; identical regularizer RNGs retain one L1/ranking
contribution before a single combined Adam update. State remains replicated
through training and is unreplicated for checkpoints and returned outputs.
`PRETRAIN_DISTRIBUTED=1` / `--pretrain-distributed` enables this optional path.

Two additional CPU checks compare three steps against the single-device
trainer with unequal shard counts, a completely absent species, singleton
cells, extreme padded values and both regularizers enabled; they check losses,
gradient norms, all parameter/optimizer/RNG state and identical replicas. The
second check verifies exact epoch resume and 16 unpadded output rows. All five
distributed checks ran without skips and passed in 43.899 seconds; the five
existing single-device pretraining tests passed in 76.165 seconds. A CLI smoke
ran one epoch each of distributed pretraining and baseline on two CPU devices,
verified 68-by-5 pretrain/final AnnData and execution/device-count metadata in
both native checkpoints.

### Distributed label-free training

`distributed/label_agnostic.py` shards encoders and gathers embeddings plus
species/index/valid-mask metadata before assembling the global objective.
InfoNCE, MMD, OT, preservation and local graph losses share the single-device
loss assembly. Global calibration gradients and training gradients use pmean.
Only params, optimizer, RNG and step are replicated; adaptive weights and best
selection statistics retain their existing host precision and checkpoint form.

Two CPU checks passed in 77.689 seconds. They compare calibration and three Adam
updates for every objective using shards containing different species, varying
weights and extreme masked padding, then verify all objective outputs and
native resume. A strengthened integration check passed separately in 40.538
seconds: MMD resumes the actual second epoch exactly, and a forced earlier
selected epoch survives resume with best parameters and a host statistic
`.40000000000013` preserved exactly. All seven existing label-free integration
and PyTorch update checks passed after the shared-loss refactor (88.266 seconds).
A one-epoch two-device MMD CLI smoke verified 16-by-5 output and topology
metadata. CLI `--distributed` / benchmark `LABEL_DISTRIBUTED=1` enables the path.

### Distributed inference

`distributed/inference.py` adds cached pmap kernels for metric banks/outputs and
pretrain embeddings/macrogenes. Params replicate once per inference pass and
remain dynamic kernel arguments; caches depend on model, devices and species
code. Fixed global batches shard over local devices and outputs concatenate in
input order, excluding padded rows. Distributed training stages use this path
for all inference and artifact emission.

Two dedicated CPU checks compare single-device and two-device outputs on
non-divisible row counts, preserve species/observation order, verify float32
outputs, reject indivisible batch sizes, and change weights while reusing the
cached kernels. All three subprocess wrappers passed in 150.739 seconds with
nine inner checks required to execute without skips, including distributed
training, earlier selection and exact resume after the inference change.

### Mixed precision

`models/precision.py` resolves fp32/bf16 matmul policy. Both CLIs and shell
runners expose precision, with bf16 default on TPU and fp32 on CPU/GPU;
library APIs default to fp32. Parameters, optimizer moments and public outputs
remain fp32. Baseline/label-free checkpoints record bf16 policy and reject a
switch to fp32 on resume; default fp32 metadata stays compatible with previous
native checkpoints. Pretraining already records its module dtype.

Two CPU precision tests passed in 38.359 seconds, including one epoch of
pretraining, baseline and every label-free objective. The same five stages
passed with two local CPU devices (51.729 seconds); state arrays and exports
were fp32 and finite. Eleven existing fp32 baseline/label-free tests passed in
89.339 seconds. A CLI smoke explicitly reused fp32 pretraining with
`--pretrain false --pretrain-mixed-precision fp32 --mixed-precision bf16`,
performed no pretraining updates, preserved pretrain embeddings exactly and
produced a bf16 baseline checkpoint with fp32 final embeddings.

These checks validate CPU dtype handling and artifact/resume policy, not
accelerator bf16 scientific parity. Actual TPU execution
and multi-host execution are still pending. This verification uses local CPU devices and
does not establish TPU performance or scientific parity.

### Bounded update checks

The new `test_updates.py` compares three consecutive production pretraining
updates against the actual PyTorch model, with copied starting weights,
three species, rotating unequal valid counts, padded batches, nonzero L1 and
protein-ranking penalties, and identical ranking draws. Dropout is controlled
to identity in both frameworks; dropout/RNG and resume have separate tests.
Each step checks total/per-species losses, both regularizers, gradient norm and
every updated parameter at the existing 1e-4 relative / 1e-5 absolute tolerance.

This exposed CPU aliasing in `torch_to_flax`: contiguous reference arrays could
share storage with JAX arrays, so a reference optimizer step mutated the copied
starting state. Conversion now snapshots each array before device transfer.
A regression test mutates every source array after conversion and verifies the
converted weights stay unchanged. All 14 model/update tests passed on CPU in
21.346 seconds. The baseline Adam comparison also now spans three updates.

Both direct Python CLIs default to two epochs, like the shell runners;
`--validation-profile full` selects reference lengths, and explicit epoch
arguments override either profile. Library API defaults are unchanged.

The label-free update test now compares three Adam steps for each of InfoNCE,
MMD and OT against the PyTorch metric model and reference objective functions.
It uses the same copied weights, frozen inference bank and teacher graphs,
three species, varying preservation weights, and two extreme padded rows
excluded by the JAX valid mask. Each update checks all component/combined losses,
coverage, gradient norm, every updated parameter and step count. The three
objective trajectories passed on CPU in 7.732 seconds without relaxing numeric
tolerances. The reference fixture uses fully populated small preservation
graphs because the PyTorch KL expression produces NaNs for absent candidates;
padding checks on trainer batches still exercise the production JAX mask.
Both bounded update tests (pretraining and all three label-free objectives)
also passed on the local CUDA GPU in 85.498 seconds. JAX executes its update
on GPU while the PyTorch reference stays on CPU. These establish deterministic
small-fixture numerical update parity; they do not establish full-atlas
scientific quality or TPU execution.

The user requested bounded validation rather than repeated full training.
All 20 shared HMM pretraining epochs completed, with loss decreasing from
3237.858887 to 1794.478394. The saved pretraining AnnData has 117,104 cells,
256 embedding dimensions and 200 macrogenes; observation IDs, species and
label metadata match the reference exactly. All completed checkpoints and
the label-free artifact were retained. The remaining 30-epoch baseline process
was stopped; full InfoNCE/MMD/OT runs were not launched.

Shell runners now default to two epochs, with explicit epoch overrides and
`VALIDATION_PROFILE=full` for notebook-length scientific evaluation. Matched
short-run numerical parity is the implementation gate; full HMM scientific
acceptance remains unestablished and is a separate optional assessment.
TPU execution, accelerator bf16 scientific parity, and distributed batches and
resume on actual TPU hardware remain unverified. Local two-device CPU batches
and resume are verified above. The HMM source contains 117,104 cells. No cloud
capacity was provisioned.

### Bounded hardware smoke and runbook

`scripts/smoke_saturn_jax.py` checks synthetic distributed pretraining without
an epoch loop or atlas download. The initial process saves step one and computes
the expected next update; a fresh process restores and compares step two,
including parameters, optimizer moments, RNG, step and loss. There are three
updates total across both processes. Device/count checks precede output creation,
receipts are immutable, and exported arrays follow the strict label-free schema.

`test_smoke.SmokeChecks` passed in 48.123 seconds using two CPU devices and bf16.
It also checks rejected receipt overwrites and unavailable TPU requests. This
does not establish TPU runtime compatibility. `cloud/TPU_RUNBOOK.md` documents
the pinned candidate TPU stack, bounded commands, receipts, cost accounting and
resource cleanup. Project, existing VM, quota and spending limit remain unknown;
no cloud resources were created.

### 25-species input preflight

`scripts/preflight_saturn_jax.py --expected-species 25 --hv-genes 64
--embed-dim 16 --num-macrogenes 8` inspected all 25 atlases without reading
expression values. They contain 212,189 cells. All atlas paths are available;
embedding paths are missing for a_thaliana, n_vectensis, s_mediterranea,
s_pistillata and z_mays. The command correctly returned status 2 and retained
all 25 species in its report.

The requested pilot dimensions imply 54,320,384 bytes for one dense float32 HV
expression array, 13,580,096 for one embedding bank and 6,790,048 for one
macrogene export before filtering. These are host array sizes, not measured peak
memory or an accelerator-fit claim. Two fixture checks passed in 0.012 seconds,
including dense/sparse shapes, missing inputs and malformed atlas reporting.
Embedding contents and gene overlap are not checked by this metadata preflight.

### Real HMM bounded baseline updates

`scripts/verify_saturn_jax_updates.py` compares three baseline Adam updates using
the real reference HMM artifact, its saved pretraining weights and its frozen
575,128 triplets. The artifact has 117,104 cells, 200 macrogenes and 256-dimensional
embeddings; the check retains hidden dimension 256 and batch size 512. Both
frameworks start from the same copied weights and receive identical triplets;
dropout is disabled. Each update compares loss, gradient norm and every updated
parameter at rtol 1e-4 / atol 1e-5.

The CPU run passed all three updates, sampling 170 triplets per update and
1,516 distinct cells total, with all three species represented in every batch.
Losses progressed from 0.09415127 to 0.08044501 to 0.07480006 in the reference;
the maximum parameter discrepancy was 3.4571e-6. The receipt records source
hashes, dimensions, coverage and devices. This is real-data update parity,
not full-epoch parity, candidate-mining parity or scientific acceptance.

The corresponding CUDA run did not pass. Its first update passed loss and
gradient-norm comparisons but failed one of 51,200 entries in the first encoder
Dense kernel (maximum absolute discrepancy 2.8039e-5). The PyTorch reference
ran on CPU while JAX ran on GPU. The first update compilation took 6m45s;
the process exited on the assertion and wrote no success receipt. The numerical
tolerances were not changed. Real HMM GPU update parity remains unestablished;
this failure needs investigation separately from the passing small-fixture
GPU checks above.

The verifier now supports a CUDA PyTorch reference, identical padded shapes in
both frameworks, explicit TF32 disabling, persistent compilation caching and
per-update timings. Numerical failures write a non-success receipt with all
mismatched parameter fields before exiting with status 1. The updated CPU
comparison still passes all three real HMM updates, with unchanged losses and
maximum parameter discrepancy. First CPU update (including compile) took 1.273s;
the next two took 0.046s and 0.035s. A ten-minute-capped same-GPU comparison
was launched and failed one first-update kernel entry (maximum discrepancy
3.7871e-5), despite identical loss. A cached single-update diagnostic found
first Adam moments -2.6921e-10 in PyTorch and -2.1100e-10 in JAX at that entry.
The corresponding gradient difference is about 5.8e-10, but both gradients are
smaller than Adam's 1e-8 epsilon, amplifying the difference in the update.

Disabling the GPU `priority-fusion` HLO pass resolved this bounded check.
With `XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0
--xla_disable_hlo_passes=priority-fusion'`, both frameworks on CUDA passed
all three real HMM updates at the original rtol 1e-4 / atol 1e-5. The first
update including compilation took 6.424s, followed by 0.098s and 0.102s;
the previous first-update compile alone took 6m35s. Maximum parameter
discrepancy across the passing trajectory was 1.0999e-5, within the combined
relative and absolute bound at each parameter. This establishes sampled
real-data baseline update parity with that compiler configuration. It does
not establish parity under default GPU fusion or whole-epoch scientific quality.

### Real HMM pretraining stress check and saturated ZINB gradients

`scripts/verify_saturn_jax_pretrain_updates.py` loads sampled expression rows
in the cached 6,000-gene order, validates manifest/embedding/centroid provenance,
and compares matched ranking draws, losses, gradient norms and parameter updates.
It defaults to fresh PyTorch initialization copied into JAX. An optional trained
checkpoint performs a separate stress check with reset Adam moments. Dropout is
controlled to identity; no whole-atlas preprocessing or epoch training is run.

The trained-checkpoint stress case exposed early saturation in JAX's default
softplus derivative. ZINB now uses PyTorch's threshold-20 forward branch and
native backward arithmetic, including multiplication before division. A focused
regression compares saturated dropout gradients within one float32 ULP of the
upstream cotangent. All 15 loss/three-update tests passed in 42.035s; five
pretraining integration/resume checks passed in 79.926s, and both two-device
pretraining gradient/resume checks passed in 32.433s.

The correction reduced the trained-checkpoint CPU maximum parameter discrepancy
from 4.3626e-4 to 1.5564e-4, but this stress case still fails strict parameter
tolerances. With both frameworks on CUDA and priority-fusion disabled, loss
matched exactly at 1721.6072998; ten dropout-head kernel entries failed, with
maximum discrepancy 2.0266e-4. Optimizer moments at the largest discrepancy
were 1.6298e-10 and -3.6089e-10, again showing tiny-gradient sensitivity with a
reset Adam optimizer. Trained-weight/reset-optimizer parity remains unestablished.

The fresh-initialization CPU check also stops at its first update: loss and
gradient norm pass, but one lemur and one mouse dropout-head kernel entry
exceed the unchanged parameter bound (maximum discrepancy 2.2758e-5). This is
therefore not solely a trained-weight/reset-optimizer issue. Real HMM
pretraining parameter-update parity remains open despite the passing smaller
fixtures and integration checks; no tolerance was relaxed for this gate.

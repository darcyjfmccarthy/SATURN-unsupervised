# Distributed TPU Spec

## Status
Orbax epoch save/resume implemented for CPU pretraining, labeled baseline and
label-free objectives. Tests cover fresh-process pretrain restoration, adaptive
weight state and selected parameters. Single-host baseline pmap is implemented
with two-device CPU checks for global triplets, Adam updates, dropout reuse,
padding and resume. Single-host pretraining pmap is implemented with global
valid-count reductions, shared regularizers and epoch resume checks.
Single-host label-free pmap gathers global embeddings and batch metadata for
InfoNCE/MMD/OT, with replicated banks/graphs and calibration gradients.
Actual TPU and multi-host execution remain pending.
Mixed-precision training policy is implemented through CLI/shell precision
flags and library model/config options. Short CPU bf16 checks cover every
stage on one and two devices; accelerator scientific bf16 parity is pending.

## Notebook Scope
This spec covers future execution of notebook cells 7, 16, 18, 20, and 22 on TPU capacity. It does not change report or evaluator cells.

In scope future components:
- `scripts/human_monkey_mouse_jax.sh`
- `scripts/train_saturn_jax.py`
- `scripts/train_label_agnostic_jax.py`
- Orbax checkpoints for all training stages.

## Current PyTorch Contract
Current notebook execution assumes a single CUDA GPU:
- Cell 2 sets `DEVICE="cuda"` and `DEVICE_NUM=0`.
- `train-saturn.py` uses PyTorch tensors, DataLoader, and `torch.optim.Adam`.
- `scripts/train_label_agnostic.py` uses a single PyTorch device and CPU-side sklearn graph construction.
- No current notebook path uses distributed PyTorch.

There is a `utils/distributed.py` helper, but it is not used by the defended notebook path.

## JAX Design Target
Primary first TPU target:
- Single-host TPU VM.
- `pmap` over local devices.
- Replicated params and optimizer state.
- Data-parallel batches sharded over the leading batch axis.
- Gradient `lax.pmean` across axis name `data`.

Implemented baseline entry point: `--metric-distributed` on
`scripts/train_saturn_jax.py`, or `METRIC_DISTRIBUTED=1` in the HMM shell.
`--batch_size` remains the global batch size and must divide local device count.
Host triplet mining uses the global preview; train steps gather embeddings
across shards before evaluating triplets and average replica gradients. Preview
and update fold the same replica index into their shared dropout key. Parameters
and optimizer state stay replicated throughout training; checkpoint/output
boundaries use one replica. Execution and local device count are recorded in
checkpoint configuration, so changing topology on resume is rejected.
`--expected-local-device-count` fails before preprocessing on a mismatch.
This flag distributes only the baseline. `--pretrain-distributed` or
`PRETRAIN_DISTRIBUTED=1` enables pretraining across local devices. Its padded
per-species batch size must divide the device count. Reconstruction gradients
use the global valid-cell denominator, then `pmean`; identical ranking and
regularizer RNGs across replicas retain one shared contribution. Per-species
gradient programs remain separate, followed by one combined Adam update.
Pretraining checkpoints record execution/topology and remove replica axes.

Label-free execution uses `--distributed` on its Python CLI or
`LABEL_DISTRIBUTED=1` in the benchmark shell. Batch size remains global.
Encoders run on shards; embeddings and valid-row/species/index metadata are
gathered before computing every objective on the global batch. Banks and teacher
graphs are replicated runtime buffers. Calibration and updates average replica
gradients. Parameters, optimizer, RNG and step remain replicated within training;
adaptive weights and selected-checkpoint statistics retain host precision.
Epoch boundaries restore the existing public checkpoint and selection format.
Distributed inference uses `distributed/inference.py` for pretrain exports,
metric snapshots/final artifacts, label-free banks and selected embeddings.
Cached pmap kernels take weights dynamically and preserve global observation
order. Params replicate once per inference pass; fixed global batches shard
over local devices and returned NumPy arrays omit all padded rows.

Future multi-host target:
- Multi-host `pmap` first if the code remains simple.
- `pjit`/GSPMD only after HMM and 25-species pmap runs reveal a memory or scaling limit.

Global batch rules:
- `global_batch_size = per_device_batch_size * jax.device_count()`.
- Every compiled step sees fixed shapes.
- Last batch is padded to global batch size.
- Losses and metrics must multiply by `valid_mask` before reduction.
- Per-species pretrain batches may compile per species if gene dimensions differ.

Sharding boundaries:
- Model params: replicated in v1.
- Optimizer state: replicated in v1.
- Current batch: sharded along batch axis.
- Embedding bank: replicated for HMM v1.
- Preservation graphs and positive indices: replicated for HMM v1.
- Large 25-species sharding is deferred to post-parity optimization.

Dtype policy:
- `MIXED_PRECISION=bf16` on TPU by default.
- Params stored fp32.
- Optax state stored fp32.
- Dense matmuls may compute bf16.
- Losses, reductions, softmax/logsumexp, ZINB, KL, and Sinkhorn compute fp32.
- Public artifacts are float32 NumPy arrays.

CLI `--mixed-precision` / shell `MIXED_PRECISION` chooses fp32 or bf16. Default
CLI policy is bf16 for TPU, fp32 for CPU/GPU; library APIs default to fp32.
The baseline CLI accepts `--pretrain-mixed-precision` separately so an existing
fp32 pretraining checkpoint can initialize a bf16 baseline without retraining.
Precision is recorded in checkpoint configuration; changing resumed training
precision is rejected. Default fp32 baseline/label-free metadata omits the new
precision field to preserve compatibility with existing native checkpoints.

Compile boundaries:
- Separate jitted/pmap functions for:
  - pretrain step.
  - metric train step.
  - label-free train step per objective if objective branches are not static.
  - embedding-bank inference.
- Objective names, species count, dimensions, and padding shapes are static compile-time values.
- Python control flow may remain around epoch loops, checkpointing, sklearn graph construction, and artifact writing.

Orbax resume rules:
- Checkpoint at epoch boundaries and final selected checkpoint.
- Restore must include params, opt state, RNG, epoch, step, best checkpoint metadata, and preservation weight where applicable.
- A resumed run must append or rewrite metric history deterministically without duplicate epochs.
- Spot/Flex-start preemption handling depends on epoch checkpoints, not step-level checkpoints, unless future profiling shows epochs are too long.

Device setup:
- Future scripts must log:
  - `jax.platforms`
  - `jax.device_count()`
  - `jax.local_device_count()`
  - device kind strings
  - process index and process count
- Training must fail fast if expected TPU devices are unavailable and `DEVICE=tpu`.

## Decisions
- Use `pmap` for the first TPU implementation because SATURN training is data-parallel and the HMM model fits comfortably with replicated params.
- Defer `pjit` until a concrete memory/scaling problem appears.
- Replicate banks and graph arrays for first HMM parity.
- Prefer epoch-level checkpointing for simplicity.
- Keep CPU preprocessing and report generation off TPU.

## Non-Goals
- Ray, GKE, or Kubernetes orchestration in v1.
- Tensor parallelism or model parallelism in v1.
- Sharded AnnData readers in v1.
- TPU-based UMAP or KNN graph construction.

## Acceptance Criteria
- Future TPU smoke test reports TPU devices and does not silently run training on CPU.
- Future single-host pmap tiny benchmark compiles once per expected step type after warmup.
- Future HMM run can resume from an Orbax checkpoint after process restart.
- Future padded batches produce identical public row counts and no padded rows in artifacts.

## Open External Facts
- Actual local device count for chosen TPU topology.
- Whether HMM epoch duration requires more frequent than epoch-level checkpointing.
- Whether 25-species high-dimensional run fits with replicated embedding banks.

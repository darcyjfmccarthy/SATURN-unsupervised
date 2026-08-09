# Distributed TPU Spec

## Status
Draft.

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

# JAX SATURN implementation

Phases 1–6 of [the migration plan](migration_plan.SPEC.md) are implemented:
host-side contracts, Flax models, losses, pretraining, labeled baseline and
InfoNCE/MMD/OT training with Orbax resume. The PyTorch path remains the scientific
reference. Full HMM parity and TPU execution remain pending.

This phase needs NumPy and Pandas. AnnData validation also needs AnnData; `.pt`
conversion needs PyTorch. The existing `saturn` environment provides these.
No JAX runtime is needed for host-side data contracts.
Models and losses use the CPU versions listed in `requirements.txt`, verified
in the existing `saturn` environment. Reference parity tests also import the
repository's PyTorch model and scvi distributions.

To compare a bounded baseline trajectory on the saved real HMM inputs, use
the reference artifact, checkpoint and frozen triplets together:

```bash
JAX_PLATFORMS=cpu python scripts/verify_saturn_jax_updates.py \
  --artifact out/human_monkey_mouse_benchmark_walkthrough/shared/label_free_artifact.npz \
  --triplets out/human_monkey_mouse_benchmark_walkthrough/shared/evaluation_triplets.npz \
  --pretrain-checkpoint out/human_monkey_mouse_benchmark_walkthrough/shared/pretrain_model.pt \
  --output /tmp/hmm-update-parity.json
```

The command defaults to three updates and caps requests at ten. It checks
losses, gradient norms and every updated parameter against PyTorch, starting
from identical copied weights. Dropout is disabled and triplets are fixed to
isolate numerical update parity. The receipt records input hashes and sampled
cell coverage; this check does not evaluate whole epochs or scientific quality.
Pass only trusted local reference checkpoints.

For a comparison with both frameworks on CUDA, set `JAX_PLATFORMS=cuda` and
pass `--reference-device cuda`. The checker disables PyTorch TF32, passes the
same padded batch to both models, writes a failure receipt on numerical mismatch
and exits with status 1. `--compilation-cache-dir /tmp/saturn-jax-compile-cache`
allows JAX to reuse compatible compiled updates. Wrap accelerator invocations
with `timeout --kill-after=30s 600s` to cap wall-clock time as well as updates.

On the tested local CUDA stack, the real HMM update check passes with:

```bash
export XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0 --xla_disable_hlo_passes=priority-fusion'
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Use these flags together with `JAX_PLATFORMS=cuda` and
`--reference-device cuda` for the command above. Disabling `priority-fusion`
reduced first-update compile/runtime from several minutes to 6.4 seconds and
passed the existing parameter tolerances. The default-fusion real HMM check
failed one parameter due to small gradient differences amplified by Adam.
This result covers three sampled baseline updates, not all objectives or epochs.

Pretraining has a corresponding bounded real-count checker, using the saved HV
gene order and fresh reference initialization:

```bash
JAX_PLATFORMS=cpu python scripts/verify_saturn_jax_pretrain_updates.py \
  --manifest data/human_monkey_mouse.csv \
  --centroids out/human_monkey_mouse_jax_parity/shared/centroids.npz \
  --embedding-cache-dir out/human_monkey_mouse_jax_parity/shared/embedding_cache \
  --output /tmp/hmm-pretrain-update-parity.json
```

It defaults to three updates and caps requests at ten. Optional
`--pretrain-checkpoint` loads trained weights with fresh optimizer moments as a
separate stress check; that case currently exceeds parameter tolerances at tiny
dropout gradients. Neither mode establishes complete epoch trajectories.

Run the tiny fixtures and compatibility tests from the repository root:

```bash
NUMBA_CACHE_DIR=/tmp/jax-saturn-numba MPLCONFIGDIR=/tmp/jax-saturn-mpl \
  JAX_PLATFORMS=cpu python -m unittest discover \
  -s jax_saturn/verification -p 'test_*.py' -v
```

The tests create all artifacts in temporary directories and require no atlas
downloads, accelerator or cloud credentials. Orbax integration tests require an
environment where asyncio thread notifications work; the Codex restricted
sandbox hangs in Orbax directory creation, so these tests were run outside it.

## Manifests

```python
from jax_saturn.data.manifest import load_manifest

rows = load_manifest("data/human_monkey_mouse.csv", check_paths=True)
```

Rows are sorted by species and retain their zero-based CSV data row index.
Relative data paths resolve against the repository root, matching the PyTorch
scripts. Use `base_dir=...` for external manifests. Both repository CSV schemas
are supported; missing labels default to `cellType`. Duplicate species, missing
required values and conflicting aliases are rejected. Path existence checks
are optional so a manifest can be inspected before data is staged.

Before staging the 25-species pilot, inspect every input and atlas shape without
loading expression matrices or starting training:

```bash
python scripts/preflight_saturn_jax.py --expected-species 25 \
  --hv-genes 64 --embed-dim 16 --num-macrogenes 8 \
  --output /tmp/saturn-25-preflight.json
```

This requires h5py, collects every missing file and exits with status 2 if inputs
are incomplete. Host array sizes exclude graphs, optimizer state and temporary
copies; they do not establish accelerator memory fit. Available embedding files
still need content and gene-overlap validation during preparation.

## Portable caches

```bash
python -m jax_saturn.data.cache embeddings \
  data/gene_embeddings/esmc600_embeddings_summaries_cpu/h_sapiens_gene_all_esmc600.pt \
  /tmp/h_sapiens_embeddings.npz --species h_sapiens

python -m jax_saturn.data.cache centroids \
  /path/to/centroids_seed0.pkl /tmp/centroids.npz \
  --seed 0 --hv-genes 2000 \
  --embedding-cache-sha256 <digest> --source-manifest-sha256 <digest>
```

Only import trusted legacy centroid pickles. The importer uses the exact cache
fields produced by `train-saturn.py`. It requires seed, HVG limit and source
hashes explicitly because the legacy file does not contain them. The embedding
digest identifies the upstream embedding cache supplied by the caller. The
preprocessing pipeline defines multi-cache provenance as SHA256 of compact,
sorted-key JSON containing the sorted list of `{"species": ..., "sha256": ...}`
cache identities. Each cache digest hashes its NPZ bytes.

Each NPZ has a sibling JSON sidecar (`centroids.npz` → `centroids.json`). Runtime
readers use `allow_pickle=False` and validate keys, dimensions, dtypes, finite
values and provenance. Gene embeddings use sorted lowercase lookup keys;
lowercase collisions retain the last source entry, matching PyTorch. Centroids
retain the reference gene order and validate sorted, contiguous species blocks.
Repeated conversions produce identical arrays, NPZ bytes and metadata.

```python
from jax_saturn.data.cache import load_cache, save_cache

arrays, metadata = load_cache("/tmp/centroids.npz", kind="centroid")
save_cache("/tmp/copied_centroids.npz", arrays, metadata, kind="centroid")
```

## Public contracts

`jax_saturn.contracts.validation` exposes:

- `load_npz(path, kind)` / `validate_npz(arrays, kind)` for `label_free`,
  `final_embeddings` and `evaluation_triplets`, with exact keys and dtypes.
- `validate_anndata(adata, ...)` for pretrain/final or evaluated AnnData, including
  optional expected observation and species order. Sparse arrays stay sparse.
- `validate_metric_history(frame, epochs=...)` for contiguous epochs and finite
  losses; additional diagnostic columns are allowed.
- `load_json(path, kind)` for `run_summary` and `checkpoint`, rejecting duplicate
  JSON keys. Run summaries allow the reference trainer's diagnostic fields and
  epoch 0 selection. Checkpoint metadata uses schema version 1 and model kinds
  `pretrain`, `baseline_metric`, and `label_agnostic`.

Checkpoint validation verifies metadata; `distributed/checkpoint.py` saves and
restores params, optimizer, RNG and training history with Orbax. Metadata is
published only after Orbax commits. Checkpoints are immutable epoch directories.

## Models and losses

`models/saturn.py` provides `FullBlock`, `SaturnPretrainModule`,
`SaturnMetricModule`, `init_pretrain_params`, and apply helpers. Species names
and gene counts are immutable tuples in sorted order. `species_code` and `train`
must be static when using `jax.jit`; species-specific gene counts compile
separately. Initializing a pretrain model creates all species heads and the
protein-ranking block. Supply the `dropout` PRNG stream when training.

```python
import jax
import jax.numpy as jnp
from jax_saturn.models.saturn import SaturnPretrainModule, init_pretrain_params

model = SaturnPretrainModule(("h_sapiens", "m_murinus"), (4, 3),
                            num_macrogenes=6, hidden_dim=9, embed_dim=5)
params = init_pretrain_params(model, jax.random.key(0), jnp.ones((7, 6)))
outputs = model.apply({"params": params}, jnp.ones((2, 4)), 0)
```

The metric forward preserves the reference's unused `cl_layer_norm` parameters.
Pretrain preserves the species covariate index 0 and the scale decoder's fixed
dropout 0.1. LayerNorm uses the Torch epsilon and centered variance rather than
[Flax defaults](https://flax-linen.readthedocs.io/en/v0.10.1/api_reference/flax.linen/layers.html).
Singleton batches retain a two-dimensional output. Parameters stay fp32;
optional bf16 matmuls cast back to fp32 for normalization and public outputs.

`models/conversion.py` maps a reference non-VAE state dict onto initialized
Flax params, transposing dense kernels and checking shapes. Metric conversion
selects encoder and layer-norm parameters from a pretrain state dict. The
converter and model runtime do not import Torch.

`losses/core.py` implements scvi-compatible ZINB, L1, protein ranking, cosine
similarity, triplet filtering and triplet margin reduction. Ranking accepts
explicit paired indices for parity or an explicit PRNG key for training.
`losses/objectives.py` implements preservation distillation, cross-species and
within-species InfoNCE, MMD and partial Sinkhorn/OT. MMD and OT require a static
`num_species`; OT returns a fixed array of pair masses, with zero for absent
species. Banks and OT teachers are frozen. CPU-built positive indices must
belong to their target species and within-species positives must exclude self.

Padded rows are excluded through `valid_mask`. Candidate index -1 denotes a
missing graph edge. All-invalid candidate rows produce zero loss and gradients,
including when masked softmax would otherwise receive only negative infinity.
Loss accumulation is fp32. CPU unit tolerances are `rtol=3e-5, atol=3e-5` for
models and `rtol=1e-4, atol=1e-5` for losses (ZINB extreme values use `atol=3e-4`).
These tests establish small-fixture parity, not full HMM scientific parity or
TPU bf16 parity.

GPU model and loss parity tests also pass on the local RTX 4060. Install the
matching backend with `python -m pip install -r jax_saturn/requirements-cuda.txt`
in a dedicated environment, following the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).
Loss dot products explicitly request full fp32 precision: default CUDA dot
precision exceeded the contrastive-loss tolerances on the GPU fixture. For
bitwise GPU resume checks, set `XLA_FLAGS=--xla_gpu_deterministic_ops=true` before
starting Python. The default GPU scatter reductions can differ in their last
bit; the baseline resume fixture differed by 2.33e-10 without that flag and
passed its unchanged exact assertions with it.

## Pretraining

```bash
python scripts/train_saturn_jax.py \
  --in_data data/saturn_run_tiny.csv --device cpu \
  --work_dir /tmp/saturn-jax-tiny --epochs 0 \
  --pretrain_epochs 1 --pretrain_batch_size 32
```

The entry point mirrors notebook-used arguments. Use `--epochs 0` for pretraining
only; positive epochs run labeled metric training. `--pretrain_model_path`
selects an Orbax output directory. Optional
`--pytorch-compat-pretrain-checkpoint` and `--pytorch-compat-metric-checkpoint`
export `.pt` state dictionaries for legacy consumers.
Requesting an unavailable GPU/TPU fails before preprocessing.

`data/preparation.py` filters genes against portable embedding caches, performs
the same normalized Seurat v3 HVG selection as PyTorch, then restores raw counts
and densifies only the selected genes. This preserves the reference's
normalization-before-HVG behavior, including Scanpy's noninteger-count warning.
Observation IDs must already be globally unique. `labels2` preserves the actual
reference's final underscore-separated token rather than silently changing label
semantics. KMeans and all three centroid score functions stay on CPU.

`data/batching.py` shuffles the reference equal-species virtual dataset and
resamples smaller species, padding fixed-shape records with masks. Sampling is
deterministic by seed and epoch. `training/pretrain.py` compiles reconstruction gradients per species, adds
regularizers once per mixed batch and applies one combined Adam update.
Reconstruction is averaged by the valid species cell count. Protein vectors
are runtime device buffers. These boundaries avoid an oversized XLA GPU fusion
program while preserving the objective and dropout key schedule.
Loss history is checkpointed with params, optimizer state and the next RNG key.
Resume verifies configuration, source manifest, embeddings, centroid scores and
selected gene order. A fresh-process test checks the next update without Torch.

Outputs include `saturn_results/adata_pretrain.h5ad`, `pretrain_losses.csv`,
`pretrain_summary.json`, portable shared caches and native Orbax epoch
checkpoints. The existing `scripts/prepare_label_agnostic_artifacts.py` accepts
the generated AnnData and builds strict artifacts and evaluation triplets.

Continue after epoch 1 with identical arguments and:

```bash
--pretrain_epochs 2 \
--resume /tmp/saturn-jax-tiny/shared/pretrain_orbax/checkpoints/epoch_0001
```

`--pretrain false --resume ...` emits embeddings from an existing checkpoint
without another update. CPU and small-fixture GPU execution are verified;
full HMM acceptance and TPU execution remain pending.

## Baseline and label-free benchmark

Run the complete pipeline with native checkpoints and the existing evaluator:

```bash
PYTHON=python DEVICE=cpu IN_DATA=data/saturn_run_tiny.csv \
  OUT_DIR=/tmp/saturn-jax-benchmark \
  PRETRAIN_EPOCHS=1 EPOCHS=1 HV_GENES=64 NUM_MACROGENES=8 \
  HIDDEN_DIM=16 MODEL_DIM=16 PRETRAIN_BATCH_SIZE=256 BATCH_SIZE=256 \
  bash scripts/run_label_agnostic_benchmark_jax.sh
```

The default manifest is HMM and the default device is CUDA. Device availability
is checked explicitly. The shell runners default to `VALIDATION_PROFILE=short`:
two shared pretraining epochs and two epochs per baseline/label-free trial.
Explicit `PRETRAIN_EPOCHS`, `METRIC_EPOCHS` (baseline runner) and `EPOCHS`
(benchmark runner) override those defaults. Use `VALIDATION_PROFILE=full` only
when a full scientific run is wanted: 20 shared pretraining epochs and 30 epochs
per trial. Pretraining is shared across all four trials.
Direct Python CLIs also default to two epochs; use `--validation-profile full`
or explicit epoch arguments to request longer training. The library APIs retain
the reference defaults for programmatic callers.

Short implementation checks compare copied starting weights, matched batches,
losses, gradients and updates, plus output contracts and checkpoint resume.
Compare training runs at the same epoch count; agreement with a completed
20/30-epoch reference is a separate scientific assessment. Short validation is
enough to progress implementation once those checks pass. Compilation and CPU
graph preparation can still dominate a short run.

Use `scripts/human_monkey_mouse_jax.sh` for the baseline
alone. `KEEP_PYTORCH_COMPAT_CHECKPOINT=1` exports legacy state dictionaries for
external PyTorch consumers. The current [benchmark notebooks](../notebooks/README.md)
and the benchmark shell above use JAX throughout with native Orbax checkpoints.

Each label-free trial consumes only `embeddings`, `macrogenes`, `species` and
`obs_ids`. Evaluation labels enter only the existing evaluator. Selection uses
species mixing subject to topology preservation, with epoch zero eligible.
Checkpoint state includes the adaptive preservation weight, selected parameters,
optimizer, RNG and history. `final_embeddings.npz`, `metric_history.csv` and
`run_summary.json` follow the reference contracts; the summary additionally
identifies JAX and its selected Orbax checkpoint.

Set `RESUME=1` to restore the latest committed native checkpoints. Resuming
requires identical data, source checkpoint and training configuration, except
the total number of epochs. Selected checkpoints are inference artifacts;
resume training from a trial's `checkpoints/epoch_*` directory. The selected
checkpoint is stored separately under `selected_orbax/`.

## Distributed training

`METRIC_DISTRIBUTED=1` in the HMM shell, or `--metric-distributed` on
`scripts/train_saturn_jax.py`, runs the labeled baseline over all local devices.
The batch size is global and must be divisible by the local device count.
Use `EXPECTED_LOCAL_DEVICE_COUNT` (shell) or `--expected-local-device-count`
(CLI) to check the topology before preprocessing. Device kinds, local/global
counts and process information are logged. Only single-host execution is
supported. Distributed checkpoints require the same device count on resume.

Triplet mining remains on CPU and sees all batch embeddings, including pairs
across device shards. Training gathers embeddings and averages gradients;
public artifacts and checkpoints keep their existing unreplicated format.
`PRETRAIN_DISTRIBUTED=1` or `--pretrain-distributed` also distributes pretraining.
Its global padded per-species batch size must divide the device count.
Reconstruction losses use global valid-cell counts; L1 and protein-ranking
regularizers contribute once per combined Adam update. Both stages keep the
same device count and execution mode on resume.

`LABEL_DISTRIBUTED=1` in the benchmark shell, or `--distributed` on
`scripts/train_label_agnostic_jax.py`, distributes InfoNCE/MMD/OT updates.
Embedding banks and teacher graphs are replicated. Encoded embeddings and batch
metadata are gathered globally before loss evaluation, preserving cross-shard
species comparisons for MMD and OT. Calibration uses the same global objective.
Adaptive weights and selected-checkpoint statistics retain host precision.

Bounded two-device CPU checks cover numerical updates/calibration, padding,
baseline dropout, absent species during pretraining, and epoch resume.
Distributed stages also use all local devices for embedding-bank inference and
public embedding export. Inference kernels cache static model/device/species
configuration while weights remain runtime inputs; epoch updates reuse kernels.
Rows retain their input order, and padded rows are removed before export.
Actual TPU execution is unverified. Preprocessing and evaluation remain on CPU.

`MIXED_PRECISION=bf16` in either shell, or `--mixed-precision bf16` on the Python
CLIs, enables bf16 matmuls for every stage. CLIs default to bf16 for TPU and fp32
for CPU/GPU; use `fp32` explicitly for a TPU reference check. Library APIs
default to fp32. Parameters, Adam state, loss arithmetic and exported arrays
remain float32. Checkpoints record bf16 execution and reject precision changes
on resume. Existing fp32 checkpoint configuration remains compatible.
Enabling bf16 does not establish biological or numerical parity on TPU.
To reuse an existing fp32 pretrain checkpoint while running a bf16 baseline,
set `PRETRAIN=false PRETRAIN_MIXED_PRECISION=fp32 MIXED_PRECISION=bf16` (CLI:
`--pretrain false --pretrain-mixed-precision fp32 --mixed-precision bf16`).
Keep the pretraining execution mode and topology consistent with its checkpoint.
The label-free CLI can initialize bf16 training directly from fp32 pretrain
parameters; its own resumed training precision must remain unchanged.

For the hardware pilot, follow [the TPU runbook](cloud/TPU_RUNBOOK.md).
`scripts/smoke_saturn_jax.py` executes a fixed three-update synthetic check
across separate initial/resume processes, records runtime/topology and optional
cloud cost assumptions, and rejects missing devices or existing receipts.
CPU smoke validation does not establish TPU compatibility.

## Full HMM reference comparison

After completing all stages at the notebook settings, compare them against a
completed PyTorch reference. This optional scientific comparison requires
`VALIDATION_PROFILE=full` when launching the benchmark; it does not accept a
two-epoch candidate against a full-length reference:

```bash
python scripts/verify_saturn_jax_parity.py \
  --reference-root out/human_monkey_mouse_benchmark_walkthrough \
  --candidate-root out/human_monkey_mouse_jax_parity
```

This command checks completed histories, model dimensions, hyperparameters,
labels and observation order before invoking the unchanged evaluator with the
reference truth and frozen triplets. It writes `reference_parity.csv` and
`reference_parity.json`. Each trial permits degradation of at most 0.02 in
fixed-triplet loss, 0.05 in label same-neighbor fraction and 0.03 in species
mixing, matching the defended evaluator's budgets. Improvements are allowed.
The report requires both reference agreement and the candidate benchmark's own
acceptance; a failing reference alone cannot establish scientific acceptance.

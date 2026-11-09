# Reusable SATURN experiment workflow

Start with the [canonical from-zero GCP runbook](../GCP_RUNBOOK.md) for cloud
setup and execution. This guide expands experiment design and reuse policy.

Use the [experiment record](EXPERIMENT_RECORD_TEMPLATE.md) for every experiment.
The [five-species worked example](../notebooks/jax/GCP_FIVE_SPECIES_RUNBOOK.md)
contains the concrete stopped-VM deployment handoff. Keep foundations immutable
and put each changed trial or foundation in a new output directory.

## Choose the operation

**Resume** means continue the same trial from its latest committed epoch with
optimizer moments, RNG, selected parameters, history and adaptive preservation
weight restored. Pass `--resume .../checkpoints/epoch_NNNN`. An exported selected
checkpoint is for inference/reporting, and cannot resume training. Increasing
the target epoch count is supported; changing scientific parameters is rejected.
The five-species runner discovers the checkpoint automatically and never repeats
completed epochs.

**New trial from a foundation** means start a fresh optimizer/RNG/selection
state from compatible **unlabelled pretraining**, not from a labelled baseline.
Use a new trial directory and omit `--resume`. Label-free training accepts exactly
`embeddings`, `macrogenes`, `species`, `obs_ids`; labels remain in separate
post-hoc truth/triplet artifacts. Foundations contain fitted preprocessing,
centroids, pretraining, teacher representations and a declared population.

**Report regeneration** means reuse complete representations and selection
results, run the evaluator/notebook again, and fit no model. For the worked
example, set `HMMFF_REUSE_FOUNDATION=1`, `HMMFF_OUTPUT_DIR` to the completed run,
`HMMFF_NEIGHBOR_BACKEND=jax`, and `HMMFF_CACHE_DIR`, then use
`scripts/execute_saturn_notebook.py`. Its reuse branch fails if a trial is
incomplete. Reports read labels only after training/selection.

| Change | Policy |
|---|---|
| Fine-tuning objective, temperature or loss weight | Reuse compatible foundation; rebuild the changed trial and affected artifacts |
| Fine-tuning seed | Reuse foundation only for an explicitly defined fine-tuning-seed experiment; rebuild seeded graphs and trial state |
| Full-pipeline seed, dimensions, HVGs, embedding model or pretraining settings | New foundation and downstream artifacts |
| Add/remove species or change training cells/expression | New foundation by default |
| Report formatting only | Reuse completed representations |

A fine-tuning-seed study does not estimate variability of the full pipeline.
Record both the inherited foundation seed and the varied fine-tuning seed.
Changing the teacher population, graph definitions, or selection data is an
experimental change even when the model architecture stays fixed.

## Stage only declared inputs

Create a CSV with `species,path,embedding_path,in_label_col`, one row/species.
Paths are relative to the repository under `data/`. Validate that species are
unique and all files exist. Preview selective commands before executing:

```bash
python scripts/stage_saturn_inputs.py \
  --manifest data/human_monkey_mouse_frog_fish.csv \
  --bucket macrogenes-saturn-artifacts --direction upload
python scripts/stage_saturn_inputs.py \
  --manifest data/human_monkey_mouse_frog_fish.csv \
  --bucket macrogenes-saturn-artifacts --direction upload --execute
python scripts/record_saturn_inputs.py \
  --manifest data/human_monkey_mouse_frog_fish.csv \
  --output out/EXPERIMENT/input_record.json
```

Download the CSV itself first on a new machine, then use the same staging script
with `--direction download --execute`. It copies only the manifest's atlases and
protein files. Do not rsync entire species directories. For a foundation-only
resume, reuse retained data and download only missing checkpoint/representation
files; the worked example supplies `download_hmmff_foundation.sh`.

**A manifest hash alone does not establish unchanged input contents.** Its paths
can point to overwritten files. Record actual input-file SHA256 values, ordered
source observation/gene identities, exported observation IDs and explicit split
CSV hashes before fitting. Legacy foundations may lack fit-time input hashes:
record the gap; hashing files now cannot retrospectively close it. Checksums of
the saved foundation still establish whether that foundation was modified.

## Create and preserve a foundation

For new populations, fit preprocessing/HVGs, centroids and pretraining on the
population declared in the record. Use the JAX wrapper's `IN_DATA`, `SEED`,
`NUM_MACROGENES`, `HIDDEN_DIM`, `MODEL_DIM`, `PRETRAIN_MODEL_PATH`,
`CENTROIDS_INIT_PATH`, `METRIC_MODEL_PATH`, `WORK_DIR` and epoch variables.
The [JAX README](README.md) documents the wrapper. Inspect its outputs and native
metadata, all-cell observation alignment, scientific settings and histories.
Export the strict teacher artifact and frozen evaluation triplets separately.
Back up a checksum inventory before downstream experiments.

For full-pipeline seed/dimension/species/cell changes, allocate a **new** centroid
cache, preprocessing output, pretrain checkpoint, baseline and teacher artifact.
Do not replace the completed foundation in place. A teacher artifact cannot be
trimmed to manufacture an unseen-species foundation: its representations and
parameters have already seen the removed cells/species.

## Exact-search policy and caches

The accelerated backend uses exhaustive cosine search, host float32 L2
normalization with a `1e-12` floor, float32 JAX matrix products at
`Precision.HIGHEST`, distance clipping to `[0,2]`, streaming exact top-k and
default query/reference blocks `1024 × 8192`. No full cell-by-cell matrix is
allocated. Species order remains sorted. Explicit JAX GPU requests fail if CUDA
is absent. sklearn remains the default reference in the standalone trainer.

GPU ties use increasing reference index. Self reference IDs are explicitly
excluded, including duplicates and zero vectors. Historical sklearn code drops
the first returned neighbor, which may not be self when vectors tie; therefore
bitwise identity with that historical behavior is claimed only on non-tied
fixtures. Tied or roundoff boundary changes are reported separately in validation.
Acceptance requires distance errors ≤`1e-5` and representative metric differences
≤`1e-4`. Tiny distance differences can reorder almost equal neighbors or later
selection outcomes; record the backend policy for the experiment.

Graph caches include actual teacher artifact content, seed, graph parameters,
backend/version/block/numerical policy and graph/search source hashes. Frozen
preservation targets, reciprocal InfoNCE positives and bandwidth can be shared
between compatible trials. Entries publish by atomic directory rename and carry
file and array checksums. Corrupt/incomplete entries fail with their path;
remove only that cache entry and rerun to rebuild. Changing a key builds a new
entry. Derived metrics/UMAP use separate content keys and never alter the
foundation `.h5ad`. Label-based post-hoc caches are separate from training caches.
Changing epoch embeddings always triggers fresh selection evaluation.

Execution settings (backend, blocks, paths, cache identities, source identity and
timings) are recorded separately from scientific checkpoint hyperparameters.
Resuming uses the same numerical/backend policy. To compare policies, create a
new trial from the same compatible foundation and record the numerical change.

## Run, time gate, backups and recovery

Deploy a verified source archive including new files; the VM need not have Git.
Exclude biological data, outputs, environments and credentials. Preserve the boot
disk, configure termination **STOP**, and set a **six-hour** max runtime before
starting the GPU VM. Record `lastStartTimestamp`; deployment and warm-up consume
that ceiling. The worked example provides exact commands and the tmux launch.

The single production runner validates both completed stages, runs two real
checkpointed epochs for InfoNCE, MMD and OT, then estimates the remaining work.
There is no disposable cloud benchmark. The gate is

```text
1.3 × (remaining setup and compilation
       + remaining InfoNCE epochs × measured steady InfoNCE epoch time
       + remaining MMD epochs × measured steady MMD epoch time
       + remaining OT epochs × measured steady OT epoch time
       + final evaluation, figures and backup)
≤ seconds before the actual six-hour deadline
```

On the first launch the remaining counts are 28 each. The first update is timed
separately as an upper bound on compilation plus one update; it is removed from
the steady epoch estimate when it is the first epoch of a process and reserved
for the next process. Measured neighbor-kernel compilation is reserved separately
for the 384-dimensional selection searches. The runner uses
the largest of its latest two available steady timings, a declared 600-second
setup reserve and a declared 3,600-second final evaluation/figure/backup reserve.
These are **estimates**, recorded in `timing_gate.json`, not a guarantee that all
work finishes in six hours. No automatic extension or experiment reduction is
allowed. A five-minute deadline reserve terminates computation and prints the
user's VM-stop command. Continue on a later launch using the retained checkpoints.

Run `backup_saturn_run.py --loop` in a separate tmux session: every five minutes
it snapshots committed checkpoints plus execution records and partial notebooks,
uploads, and checks a checksum-only dry run. It skips unpublished Orbax writes.
Inspect failures; periodic backup is not evidence of final completion. Finish
with a one-off verified snapshot, download the prefix locally, and verify every
file against `backup_manifest.json`. Final backup verification and VM stopping
are user actions in this workflow.

A checkpoint is committed only when `metadata.json` exists after all Orbax
components. The runner preserves incomplete directories under `interrupted/` before
retrying that epoch; committed directories are never moved. Standalone trainer
users must move an incomplete write out of `checkpoints/epoch_NNNN` before retrying. Never delete committed
epochs to force a rerun. Histories and summaries can be regenerated by restoring
the latest committed checkpoint with the same target; training resumes only
above its epoch counter. Completion requires exact epochs `1–30`, matching
summaries/checkpoint identities, checksum-verified completion records and selected representations. Epoch-2 exports
alone do not count.

## Worked experiment recipes

### Resume the five-species experiment

Use the [worked example](../notebooks/jax/GCP_FIVE_SPECIES_RUNBOOK.md). It reuses
pretrain epoch 20 and labelled baseline epoch 30; its new output is
`out/human_monkey_mouse_frog_fish_jax_gpu_exact`. Preflight must validate
148,727 cells, all five species, dimensions `300/384/384`, seed 0, batch 512,
fp32 and frozen evaluation triplets. The one-command runner first fills any
missing warm-up epochs, gates, then finishes each objective through epoch 30.

### Add a sixth species

1. Copy the five-species CSV to `data/SIX_SPECIES.csv`; add exactly one species
   row with its atlas, protein file and label column.
2. Stage that manifest's six required atlases/protein files with
   `stage_saturn_inputs.py`; already uploaded objects need not be recopied by
   hand. Record actual content hashes and observation identities.
3. Create a new centroid/preprocessing/foundation directory. Fit on all six
   declared species; train a new labelled baseline and teacher artifact.
4. Launch new fine-tuning trials using that new foundation, then post-hoc
   evaluation. Adapt the notebook's manifest/report source for six species.

The **five-species runner intentionally rejects six species**. General trainer
and wrapper commands are reusable; the hard-coded worked example's foundation
validator and figures are not a generic sixth-species runner. Adding a species
is not supported as an Orbax checkpoint resume or by appending teacher rows.

### Temperature and local-graph-weight ablations

These supported fine-tuning changes can reuse a compatible foundation. Record
question, inherited foundation seed, changed variable and fresh trial directory.
For example, compare InfoNCE temperature `.1` versus `.2` and local graph weight
`.3` versus `0`, holding all other settings fixed:

```bash
python scripts/train_label_agnostic_jax.py \
  --objective infonce --artifact out/FOUNDATION/shared/label_free_artifact.npz \
  --pretrain-checkpoint out/FOUNDATION/shared/pretrain_orbax/checkpoints/epoch_0020 \
  --output-dir out/ABLATION/infonce_t02 --device gpu --epochs 30 \
  --seed 0 --batch-size 512 --hidden-dim 384 --model-dim 384 \
  --mixed-precision fp32 --neighbor-backend jax --cache-dir out/ABLATION/cache \
  --infonce-temperature 0.2
python scripts/train_label_agnostic_jax.py \
  --objective infonce --artifact out/FOUNDATION/shared/label_free_artifact.npz \
  --pretrain-checkpoint out/FOUNDATION/shared/pretrain_orbax/checkpoints/epoch_0020 \
  --output-dir out/ABLATION/infonce_local0 --device gpu --epochs 30 \
  --seed 0 --batch-size 512 --hidden-dim 384 --model-dim 384 \
  --mixed-precision fp32 --neighbor-backend jax --cache-dir out/ABLATION/cache \
  --local-graph-weight 0
```

Do not add `--resume` from a differently configured trial. Compatible frozen
teacher graphs may be cached, but new optimizer/selection histories are required.
A zero preservation-gradient ratio does **not** disable preservation: the weight
is clipped to at least `1e-3`; do not call that a preservation-off ablation.
The specialized one-launch runner fixes the approved five-species configuration;
for ablations, schedule standalone trials within a separately recorded budget.

### Species holdouts

- **Evaluation subset:** train on all species, report one species separately.
  This measures performance on an exposed population; retain the full neighbor
  reference population if that is the metric definition. Record query/reference
  populations rather than silently recomputing neighbors on the subset.
- **Fine-tuning-only exclusion:** a species may have contributed to preprocessing,
  centroids, pretraining and teacher features but is excluded from fine-tuning.
  This is a transductive exposure design, not a strict unseen-species claim.
  The current JAX trainer requires artifact/checkpoint species to match, so
  simply trimming the artifact is rejected. A defined exclusion/batching and
  evaluation path needs additional implementation; **unsupported in this change**.
- **Strict inductive species holdout:** exclude the species from preprocessing,
  feature/HVG fitting, centroid fitting, pretraining, teacher graphs, bandwidth,
  fine-tuning and epoch selection. Fit a new foundation on training species only.
  JAX currently lacks a complete unseen-species transform/transfer workflow.
  Generating unseen-species macrogenes/representations with frozen fitted state
  and evaluating them is **unsupported** here. Root `inference.py` is a legacy
  PyTorch `.pt`/pickle workflow and cannot consume native Orbax checkpoints.

Defining and fitting the training-species foundation is supported; a complete
strict unseen-species experiment is not yet runnable end to end. Do not label
transductive removal of graph nodes as inductive transfer.

### Cell/donor holdouts

Create an explicit split CSV before any fitting, with stable source observation
IDs, species, donor/group IDs and split. Split by donor when donor generalization
is the question; keep donor/technical replicate groups together. Record overlap,
species/cell-type counts, missing donors and any external test data.

- **Evaluation subset:** reuse complete representations and report test IDs as
  a post-hoc subset. Those cells have been seen; this is not inductive validation.
- **Fine-tuning-only exclusion:** exclude designated observations from every
  fine-tuning batch, teacher positive/negative/local graph and bandwidth fitting,
  but acknowledge exposure during the inherited foundation. The current trainer
  has no explicit exclusion-mask workflow; preparing subset artifacts by hand
  also requires aligned evaluation triplets and a declared selection population.
  It is not a supported one-command recipe in this change.
- **Strict inductive cell/donor holdout:** fit HVGs, feature mappings, normalization
  statistics, centroids, pretraining, graphs/bandwidth and model selection using
  training observations only. Use validation donors for declared selection and
  keep test donors out of selection and hyperparameter tuning. Do not fit UMAP
  or preprocessing jointly and then claim frozen held-out inference. Materialize
  a separate training-only atlas and new foundation, retaining observation maps
  and hashes. The five-species cell workflow now implements this with
  `scripts/run_cell_holdout_jax.py` and the holdout notebook: training-only
  atlases, saved selected-gene order, frozen Orbax inference, training-fitted
  UMAP `transform`, and test-query/training-reference metrics. See the canonical
  runbook's strict cell holdout recipe. This split is by cell within species;
  donor-group splitting remains outside this implementation.

For every holdout, disclose externally pretrained protein-model exposure as well
as experiment-specific exposure. Reusing a full-data centroid or teacher leaks
held-out information even when no held-out cell reaches the fine-tuning batches.

### Selected methods for the RDEB cell holdout

`HoldoutConfig.trials` and `SATURN_TRIALS` select methods consistently through
training, timing, inference, evaluation, diagnostics and completion. The default
remains `baseline,infonce,mmd,ot` for historical runs; the RDEB configuration uses
`baseline,infonce`. The standalone evaluator accepts `--trials baseline infonce`.
A different method set requires a fresh output/configuration record.

`REPORT_NOTEBOOK` selects the artifact report; `DISEASE_METADATA` supplies post-hoc
mouse metadata. The portable input bundle records actual bytes and stages its
metadata explicitly with `stage_saturn_inputs.py --bundle-record ...`. See the
[canonical disease recipe](../GCP_RUNBOOK.md#rdeb-skin-disease-experiment).

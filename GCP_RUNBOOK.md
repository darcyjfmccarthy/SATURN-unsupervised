# SATURN on Google Cloud — from zero to a verified experiment

**This is the canonical operational runbook. Update it first when the workflow
improves.** It applies to any species manifest using this repository's JAX
training harness. Experiment names, inputs and scientific settings live in one
configuration file; the repository location and harness stay fixed.

Last reviewed: 2026-10-17. Local repository:
`/home/darcy/projects/SATURN-unsupervised`. VM repository:
`~/SATURN-unsupervised`. Persistent environment: `.venv-saturn-gpu`.

Start at step 1 for a new experiment. Skip resource creation and environment
installation when reusing the retained VM. For interrupted work, use step 12.
Keep GPU compute **stopped while preparing code, inputs and configuration locally**.
All cloud commands here are commands for the operator to run.

Related documents:

- [Experiment design, reuse and holdout policy](jax_saturn/EXPERIMENT_WORKFLOW.md).
- [Experiment record template](jax_saturn/EXPERIMENT_RECORD_TEMPLATE.md).
- [Existing experiment continuation example](notebooks/jax/GCP_FIVE_SPECIES_RUNBOOK.md).
- [Implementation and validation](jax_saturn/README.md).

## 1. LOCAL — define the experiment once

Use a new ID whenever the scientific question/configuration or training
population changes. A resume keeps the same ID. Create this file, edit its
values, then source it in each shell. It contains configuration, never credentials.

```bash
cd /home/darcy/projects/SATURN-unsupervised
mkdir -p experiments
export SATURN_CONFIG='experiments/experiment_001.env'
cat > "$SATURN_CONFIG" <<'ENV'
export EXPERIMENT='experiment_001'
export SATURN_PROJECT='macrogenes'
export SATURN_REGION='asia-southeast1'
export SATURN_ZONE='asia-southeast1-a'
export SATURN_VM='saturn-a100-pilot'
export SATURN_BUCKET="${SATURN_PROJECT}-saturn-artifacts"
export SATURN_SA="saturn-training-vm@${SATURN_PROJECT}.iam.gserviceaccount.com"

export IN_DATA="data/manifests/${EXPERIMENT}.csv"
export RUN_ROOT="out/${EXPERIMENT}"
export FOUNDATION_ROOT="$RUN_ROOT/foundation"
export RUN_DIR="$RUN_ROOT/run"
export CACHE_DIR="$RUN_ROOT/cache"
export BUCKET_PREFIX="gs://$SATURN_BUCKET/runs/$EXPERIMENT"

export SEED=0
export HV_GENES=2000
export NUM_MACROGENES=200
export HIDDEN_DIM=256
export MODEL_DIM=256
export PRETRAIN_EPOCHS=20
export METRIC_EPOCHS=30
export LABEL_FREE_EPOCHS=30
export PRETRAIN_BATCH_SIZE=512
export METRIC_BATCH_SIZE=512
export MIXED_PRECISION='fp32'
export PRETRAIN_MIXED_PRECISION='fp32'
export NEIGHBOR_BACKEND='jax'
export QUERY_BLOCK=1024
export REFERENCE_BLOCK=8192

export DEVICE='cuda'
export DEVICE_NUM=0
export VALIDATION_PROFILE='full'
export NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR='/tmp/macrogenes-numba-cache'
export MPLCONFIGDIR='/tmp/macrogenes-matplotlib'
export XLA_PYTHON_CLIENT_PREALLOCATE='false'
export CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_ENABLED='False'
ENV
source "$SATURN_CONFIG"
mkdir -p "$RUN_ROOT"
cp jax_saturn/EXPERIMENT_RECORD_TEMPLATE.md "$RUN_ROOT/EXPERIMENT_RECORD.md"
```

The dimensions/epochs above are explicit defaults, not inferred from the species
count. Change them deliberately and record the change. The wrappers otherwise
have shorter validation defaults; always pass the configured epoch counts.

### Strict five-species cell holdout

The implemented holdout workflow uses the original **200/256/256** architecture,
seed 0, 2,000 HVGs/species, fp32 and batch 512. It reserves 10% of each species
before any fitting (current inputs: 133,852 train / 14,875 test), then fits a new
20-epoch foundation and four 30-epoch fine-tuning trials. This is a cell holdout
within known species, not an unseen-species or donor-generalization experiment.

```bash
source experiments/hmmff_cell_holdout.env
python scripts/run_cell_holdout_jax.py prepare
DEVICE=cpu python scripts/run_cell_holdout_jax.py preflight
python scripts/record_saturn_inputs.py --manifest "$IN_DATA" \
  --split "$RUN_ROOT/split/split.csv" --output "$RUN_ROOT/input_record.json"
```

Complete local validation and source packaging before provisioning. Follow steps
2–7 below with this experiment configuration and its fresh VM name. Install
`jax_saturn/requirements-holdout-gpu.txt` plus the CPU PyTorch wheel; the included
`scripts/bootstrap_cell_holdout_vm.sh` installs a fresh VM, stages inputs,
verifies the prepared split, starts periodic backups and executes production.
Its fresh install pins NVIDIA **580.178.04** using NVIDIA's signed Ubuntu
repository and version-lock package. This avoids the floating production
installer selecting a different system driver; `gpu_driver.txt` records the
actual driver, kernel-module flavor and guest kernel. See NVIDIA's
[Ubuntu installation](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/ubuntu.html)
and [version locking](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/version-locking.html)
instructions. The first holdout setup encountered an R615 UVM initialization
hang before training and switched to this pinned branch within its original window.
Preserve the **initial** GCE start timestamp in `~/saturn-vm-start.txt`, including
across a driver reboot. For this experiment authorization is **one six-hour
window total**, including installation: never start another window automatically.

After activating the VM environment and sourcing the configuration:

```bash
export SATURN_VM_START="$(cat ~/saturn-vm-start.txt)"
python scripts/run_cell_holdout_jax.py preflight
python -u scripts/run_cell_holdout_jax.py run --vm-start-time "$SATURN_VM_START"
```

The controller measures real production epochs for foundation and fine-tuning,
records explicit reserves for unmeasured stages, gates remaining work, and keeps
a five-minute deadline reserve. All preprocessing, graphs, losses and selection
use training observations only. Frozen inference requires no cell labels.
Post-hoc test metrics query training references; UMAP fits only training cells
and transforms held-out cells. The executed notebook and PNG/PDF/HTML figures
live under `$RUN_ROOT`, with split/provenance, fitted gene order, checkpoints,
aligned embeddings, reducers and coordinates retained separately.

From your local terminal, follow setup and training output with:

```bash
source experiments/hmmff_cell_holdout.env
gcloud compute ssh "$SATURN_VM" --project="$SATURN_PROJECT" --zone="$SATURN_ZONE" \
  --command="tail -n 100 -f ~/SATURN-unsupervised/$RUN_ROOT/setup.log"
```

Ctrl-C closes the log viewer; the job continues in `saturn-production` tmux.
Epoch and stage records are also saved in `execution.jsonl` files beneath the
run directory. Backup progress is in `/tmp/saturn-holdout-backup.log` on the VM.

For report regeneration, run `python scripts/run_cell_holdout_jax.py report`,
then execute `notebooks/jax/human_monkey_mouse_frog_fish_holdout.ipynb` with
`SATURN_HOLDOUT_REPORT_ONLY=1`. Complete the verified backup/download and STOP
steps even when the time gate ends the experiment with partial checkpoints.

For completed artifacts, `DEVICE=cpu NEIGHBOR_BACKEND=jax python scripts/run_cell_holdout_jax.py diagnostics` regenerates the generalization
figures locally. It reuses the finalized embeddings, cached training neighbours
and saved coordinates. Training queries exclude themselves; test queries use
the training reference. The `diagnostics/` directory contains paired train/test
scores, species and cell-type gaps, per-cell scores and an HTML explorer that
filters wrong cell-type votes. Models and UMAPs remain frozen. These diagnostics
describe held-out cells from the input atlases; they do not test unseen donors.

Fill in the question, parent foundation, changes and acceptance criteria before
training. If an executed notebook/figure report is required, prepare its
experiment-specific report notebook locally now (see step 10), before GPU billing starts. The [reuse table](jax_saturn/EXPERIMENT_WORKFLOW.md#choose-the-operation)
determines whether a new foundation is required. Adding/removing species,
changing expression/cells, full-pipeline seed, preprocessing or architecture
normally requires a new foundation. A compatible fine-tuning ablation can reuse
one; step 8 explains that branch.

### RDEB skin disease experiment

This first disease experiment keeps all original atlas cells and adds the study's
WT1/WT2 and KO1/KO2 cells. It uses a fresh 200/256/256 foundation, 20 pretraining
epochs and **baseline + InfoNCE only**, 30 epochs each. The existing 90/10 cell
split produces 142,686 training / 15,857 test cells. Donors can occur in both
splits. The six-hour total window, including installation, remains unchanged.

With GPU compute stopped, prepare and validate locally:

```bash
export SATURN_CONFIG='experiments/hmmff_rdeb_skin_cell_holdout.env'
source "$SATURN_CONFIG"
python scripts/prepare_disease_cloud_inputs.py
python scripts/run_cell_holdout_jax.py prepare
DEVICE=cpu python scripts/run_cell_holdout_jax.py preflight
python scripts/record_saturn_inputs.py --manifest "$IN_DATA" \
  --split "$RUN_ROOT/split/split.csv" --output "$RUN_ROOT/input_record.json"
python scripts/stage_saturn_inputs.py --manifest "$IN_DATA" --bucket "$SATURN_BUCKET" \
  --bundle-record "$DISEASE_INPUT_BUNDLE" --direction upload
# After inspecting the declared transfer list:
python scripts/stage_saturn_inputs.py --manifest "$IN_DATA" --bucket "$SATURN_BUCKET" \
  --bundle-record "$DISEASE_INPUT_BUNDLE" --direction upload --execute
```

The bundle copies verified normalized expression from the full preparation
`exploration.csv` into portable `data/` paths; it does not use donor-fold inputs.
Its explicit inventory also stages evaluation metadata and preparation audits.
Run `python -m unittest disease_pilot.test_cloud` and the existing holdout tests,
then `python scripts/validate_disease_holdout.py` for the one-epoch, five-species
CPU fixture, including both methods and the executed disease notebook. This is
implementation validation, not a scientific training arm.

Upload the input record and completed experiment record to
`gs://$SATURN_BUCKET/experiment_specs/$EXPERIMENT/`, then package source and follow
steps 4–7. Transfer `~/saturn-experiment-config.txt` containing `$SATURN_CONFIG`,
the **initial** `~/saturn-vm-start.txt`, and `~/saturn-vm-deadline.txt` containing
that start plus six hours. Run the parameterized bootstrap in `saturn-production`
tmux. Reconnect after its expected driver reboot and rerun the bootstrap within
the original window; do not start another cloud window or replace its timestamp.

The controller executes `$REPORT_NOTEBOOK`, which is an artifact-only disease
report, and requires its disease figures and aligned metadata for completion.
`report` regenerates finalized representations, diagnostics and the disease
views without training. Complete verified backup/download, STOP and retirement
as in steps 11–13. Keep partial checkpoints and the stopped disk if the single
window ends before completion.

## 2. LOCAL — authenticate and prepare cloud resources

Install the [Google Cloud CLI](https://docs.cloud.google.com/sdk/docs/install-sdk)
if `gcloud version` fails. Choose a billing-enabled project and set a budget alert
in Cloud Billing. [Budget alerts do not impose a spending cap](https://docs.cloud.google.com/billing/docs/how-to/budgets).
The operational limit here is a six-hour VM ceiling with automatic **STOP**.

```bash
gcloud auth login
gcloud config set project "$SATURN_PROJECT"
gcloud services enable compute.googleapis.com storage.googleapis.com \
  --project="$SATURN_PROJECT"
gcloud projects describe "$SATURN_PROJECT" \
  --format='table(projectId,projectNumber,name)'
gcloud auth list --filter=status:ACTIVE
```

Before creating a GPU VM, check the intended region's GPU quota and current
[GPU locations](https://docs.cloud.google.com/compute/docs/regions-zones/gpu-regions-zones).
This guide uses one A100 via `a2-highgpu-1g`; Spot quota and available capacity
must cover that machine. Keep the requested GPU count at one. If capacity is
unavailable, choose another supported zone and update the configuration; do not
silently change the experiment or launch multiple machines.

**First setup only:** create the bucket/service account below if absent.
For existing resources, inspect and reuse them rather than recreating them.

```bash
gcloud storage buckets create "gs://$SATURN_BUCKET" \
  --project="$SATURN_PROJECT" --location="$SATURN_REGION" \
  --uniform-bucket-level-access
gcloud iam service-accounts create saturn-training-vm \
  --project="$SATURN_PROJECT" --display-name='SATURN training VM'
gcloud storage buckets add-iam-policy-binding "gs://$SATURN_BUCKET" \
  --member="serviceAccount:$SATURN_SA" --role='roles/storage.objectAdmin'
gcloud storage ls "gs://$SATURN_BUCKET"
```

The bucket-scoped [IAM binding](https://docs.cloud.google.com/sdk/gcloud/reference/storage/buckets/add-iam-policy-binding)
lets the VM stage inputs and write backups. Use its attached service account;
no downloaded service-account key or interactive VM login is needed. An IAM
administrator may need to perform initial provisioning/attach-account grants.

## 3. LOCAL — prepare, validate and selectively upload inputs

Create the CSV at `$IN_DATA`. Use one row per species, with any species names:

```csv
species,path,embedding_path,in_label_col
species_a,data/cell_atlases/species_a.h5ad,data/gene_embeddings/species_a.pt,cellType
species_b,data/cell_atlases/species_b.h5ad,data/gene_embeddings/species_b.pt,cellType
```

Use at least two species for these label-free objectives. Atlas `X` must contain
finite nonnegative counts; gene/observation names must be unique. Observation IDs
must also be globally unique across species. The current foundation wrapper
expects a `cellType` reference-label column even if `in_label_col` names another
column. All species must use compatible protein embeddings. For changed input
contents, use new/versioned input paths so older experiments remain recoverable.

Use your existing local SATURN environment. If none exists, a small local input
inspection environment is sufficient for the following steps:

```bash
python3 -m venv .venv-saturn-local
source .venv-saturn-local/bin/activate
python -m pip install 'numpy<2' pandas 'anndata==0.11.4' h5py
```

```bash
python scripts/preflight_saturn_jax.py --manifest "$IN_DATA" \
  --hv-genes "$HV_GENES" --embed-dim "$MODEL_DIM" \
  --num-macrogenes "$NUM_MACROGENES" --output "$RUN_ROOT/input_preflight.json"
python scripts/record_saturn_inputs.py --manifest "$IN_DATA" \
  --output "$RUN_ROOT/input_record.json"
python - <<'PY'
import os
import anndata as ad
from jax_saturn.data.manifest import load_manifest
rows = load_manifest(os.environ['IN_DATA'], check_paths=True)
assert len(rows) >= 2, 'At least two species required'
seen = set()
for row in rows:
    atlas = ad.read_h5ad(row.path, backed='r')
    assert atlas.obs_names.is_unique and atlas.var_names.is_unique, row.species
    identifiers = set(map(str, atlas.obs_names))
    assert not seen.intersection(identifiers), f'Observation IDs overlap: {row.species}'
    seen.update(identifiers)
    for column in (row.in_label_col, 'cellType'):
        assert column in atlas.obs and not atlas.obs[column].isna().any(), (row.species, column)
    atlas.file.close()
print(f'Validated metadata for {len(rows)} species and {len(seen)} cells.')
PY
python scripts/stage_saturn_inputs.py --manifest "$IN_DATA" \
  --bucket "$SATURN_BUCKET" --direction upload
```

The first preflight checks paths/shapes, not all matrix/embedding contents.
The metadata check adds label/ID validation; training preparation checks counts,
gene overlap and protein caches. Review the printed upload list: exactly the
manifest and its required atlas/protein files. Then upload:

```bash
python scripts/stage_saturn_inputs.py --manifest "$IN_DATA" \
  --bucket "$SATURN_BUCKET" --direction upload --execute
gcloud storage cp "$RUN_ROOT/input_record.json" \
  "gs://$SATURN_BUCKET/experiment_specs/$EXPERIMENT/input_record.json"
```

A CSV hash records paths/settings; it does **not** establish unchanged input
contents. `input_record.json` records actual file hashes and observation/gene
identities. Record split CSVs too (`record_saturn_inputs.py --split ...`) for
holdouts. See the [holdout recipes](jax_saturn/EXPERIMENT_WORKFLOW.md#worked-experiment-recipes)
for leakage controls and currently unsupported frozen-transform steps.

## 4. LOCAL — package the source before starting the GPU

```bash
python scripts/package_saturn_source.py
(cd dist && sha256sum -c saturn-gpu-exact.tar.gz.sha256)
printf '%s\n' "$SATURN_CONFIG" > /tmp/saturn-experiment-config.txt
```

The archive includes current edits, new files and the experiment configuration;
it excludes data, outputs, environments and credentials. Its source manifest is
verified after installation. The VM does not need a Git checkout. Finish code
changes and local checks now, while the GPU is stopped.

## 5. LOCAL — create the VM once, or start the retained VM

**Only when the VM does not exist**, create it with a retained boot disk and
six-hour STOP ceiling:

```bash
gcloud compute instances create "$SATURN_VM" \
  --project="$SATURN_PROJECT" --zone="$SATURN_ZONE" \
  --machine-type=a2-highgpu-1g \
  --provisioning-model=SPOT --maintenance-policy=TERMINATE \
  --max-run-duration=6h --instance-termination-action=STOP \
  --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
  --boot-disk-size=250GB --boot-disk-type=pd-balanced \
  --no-boot-disk-auto-delete \
  --service-account="$SATURN_SA" --scopes=cloud-platform
```

The machine type includes its GPU. See Google's [GPU VM overview](https://docs.cloud.google.com/compute/docs/gpus/create-vm-with-gpus)
and [creation flags](https://docs.cloud.google.com/sdk/gcloud/reference/compute/instances/create).
Creation starts compute immediately. Disk size must cover inputs, retained
foundations, checkpoints, outputs and backup staging; increase it deliberately
for larger experiments. Use a VPC that allows your intended SSH access. A project
without a default VPC needs its own network/subnet settings on this command.

**For an existing stopped VM**, inspect it and correct retention/scheduling
before starting:

```bash
gcloud compute instances describe "$SATURN_VM" \
  --project="$SATURN_PROJECT" --zone="$SATURN_ZONE" \
  --format='yaml(status,disks,scheduling.maxRunDuration,scheduling.instanceTerminationAction)'
```

Require the boot disk's `autoDelete: false`, `maxRunDuration: 21600s` and action
`STOP`. If necessary while stopped, use `set-disk-auto-delete --no-auto-delete`
and `set-scheduling --max-run-duration=6h --instance-termination-action=STOP`.
The boot-disk resource name is shown in `disks`; it need not equal the VM name.
Then, for the existing stopped VM only:

```bash
gcloud compute instances start "$SATURN_VM" \
  --project="$SATURN_PROJECT" --zone="$SATURN_ZONE"
```

After either creation or start:

```bash
gcloud compute instances describe "$SATURN_VM" \
  --project="$SATURN_PROJECT" --zone="$SATURN_ZONE" \
  --format='value(lastStartTimestamp)' > /tmp/saturn-vm-start.txt
gcloud compute ssh "$SATURN_VM" \
  --project="$SATURN_PROJECT" --zone="$SATURN_ZONE" --command='true'
```

If SSH is briefly refused during boot, wait about 20 seconds and retry. Persistent
failure: rerun with `--troubleshoot`. Once connected, transfer and log in:

```bash
gcloud compute scp dist/saturn-gpu-exact.tar.gz \
  dist/saturn-gpu-exact.tar.gz.sha256 /tmp/saturn-experiment-config.txt \
  /tmp/saturn-vm-start.txt "$SATURN_VM:~/" \
  --project="$SATURN_PROJECT" --zone="$SATURN_ZONE"
gcloud compute ssh "$SATURN_VM" \
  --project="$SATURN_PROJECT" --zone="$SATURN_ZONE"
```

Google's [runtime limit](https://docs.cloud.google.com/compute/docs/instances/limit-vm-runtime)
starts with the VM, including deployment, installation and warm-up. Each later
start has a new timestamp. Never extend the six-hour ceiling automatically.

## 6. VM — install source and set up the environment

```bash
cd ~
sha256sum -c saturn-gpu-exact.tar.gz.sha256
mkdir -p ~/SATURN-unsupervised
cd ~/SATURN-unsupervised
tar --no-same-owner -xzf ~/saturn-gpu-exact.tar.gz
source "$(cat ~/saturn-experiment-config.txt)"
python3 scripts/package_saturn_source.py --verify-tree source_manifest.json
```

Local exports do not cross SSH. Source the configuration again after every new
VM login. **Fresh machine only**, install the OS packages:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip curl tmux
```

Check `nvidia-smi`. If the driver is absent, use Google's
[driver installer](https://docs.cloud.google.com/compute/docs/gpus/install-drivers-gpu):

```bash
curl -fL https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz \
  --output /tmp/cuda_installer.pyz
sudo python3 /tmp/cuda_installer.pyz install_driver \
  --installation-mode=repo --installation-branch=prod
```

The installer can reboot; reconnect and repeat its command if it asks to continue.
After any reboot, recapture `lastStartTimestamp` from the local terminal and
transfer the refreshed `/tmp/saturn-vm-start.txt`. Require `nvidia-smi` to work
before Python GPU checks. The Python JAX CUDA wheels provide their CUDA libraries;
this recipe does not require a separately installed system CUDA Toolkit.

**Fresh Python environment only:**

```bash
python3 -m venv .venv-saturn-gpu
source .venv-saturn-gpu/bin/activate
python -m pip install --upgrade pip
python -m pip install -r jax_saturn/requirements-cuda.txt
python -m pip install 'numpy<2' pandas 'anndata==0.11.4' scanpy scipy \
  'scikit-learn==1.7.2' scikit-misc h5py matplotlib seaborn \
  'umap-learn==0.5.12' pyyaml plotly 'nbclient==0.10.4' nbformat nbconvert ipykernel psutil
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip check
```

A retained, working environment needs activation rather than reinstallation:

```bash
source .venv-saturn-gpu/bin/activate
python - <<'PY'
import jax
import jax.numpy as jnp
assert jax.__version__ == '0.6.2'
device = jax.devices('gpu')[0]
with jax.default_device(device):
    result = jnp.ones((4, 4)) @ jnp.ones((4, 4))
    result.block_until_ready()
print(device, result.shape)
PY
```

This checks actual execution, not just GPU discovery. If CUDA libraries/compiler
are newer than the driver's supported version, align their versions or install a
compatible driver; do not accept silent CPU fallback. Preserve the environment
for resumes, and record its versions instead of broadly upgrading between stages.

## 7. VM — stage the same inputs, validate and start backups

```bash
mkdir -p "$(dirname "$IN_DATA")" "$RUN_ROOT" "$RUN_DIR"
gcloud storage cp "gs://$SATURN_BUCKET/inputs/${IN_DATA#data/}" "$IN_DATA"
python scripts/stage_saturn_inputs.py --manifest "$IN_DATA" \
  --bucket "$SATURN_BUCKET" --direction download --execute
python scripts/preflight_saturn_jax.py --manifest "$IN_DATA" \
  --hv-genes "$HV_GENES" --embed-dim "$MODEL_DIM" --num-macrogenes "$NUM_MACROGENES" \
  --output "$RUN_ROOT/vm_input_preflight.json"
python scripts/record_saturn_inputs.py --manifest "$IN_DATA" --output "$RUN_ROOT/input_record.json"
gcloud storage cp "gs://$SATURN_BUCKET/experiment_specs/$EXPERIMENT/input_record.json" \
  "$RUN_ROOT/local_input_record.json"
python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['RUN_ROOT'])
local = json.loads((root / 'local_input_record.json').read_text())
vm = json.loads((root / 'input_record.json').read_text())
assert local['manifest_sha256'] == vm['manifest_sha256']
assert local['inputs'] == vm['inputs'], 'VM input contents differ from staged local inputs'
print('Verified actual atlas/protein hashes and observation/gene identities.')
PY
python -m pip freeze > "$RUN_ROOT/environment.txt"
cp source_manifest.json "$RUN_ROOT/source_manifest.json"
cp "$SATURN_CONFIG" "$RUN_ROOT/experiment.env"
test -f "$RUN_ROOT/EXPERIMENT_RECORD.md" || \
  cp jax_saturn/EXPERIMENT_RECORD_TEMPLATE.md "$RUN_ROOT/EXPERIMENT_RECORD.md"
tmux new-session -s saturn-backup
```

Inside that backup tmux session:

```bash
cd ~/SATURN-unsupervised
source "$(cat ~/saturn-experiment-config.txt)"
source .venv-saturn-gpu/bin/activate
python -u scripts/backup_saturn_run.py --run-dir "$RUN_ROOT" \
  --bucket-prefix "$BUCKET_PREFIX" --loop 2>&1 | tee /tmp/saturn-backup.log
```

Detach with **Ctrl-b, then d**. This snapshots committed checkpoints and partial
notebooks every five minutes, and verifies the upload with a checksum-only dry
run. Logs are outside the snapshotted directory to avoid a self-changing backup
log. Inspect backup failures; a running backup process alone is not proof of a
successful backup. Start a separate production tmux session:

```bash
tmux new-session -s saturn-production
cd ~/SATURN-unsupervised
source "$(cat ~/saturn-experiment-config.txt)"
source .venv-saturn-gpu/bin/activate
set -o pipefail
```

Use the following stages there. Detaching tmux leaves computation running;
closing SSH is safe only once the process is inside tmux.

## 8. VM — fit a new foundation, or reuse a compatible one

For a **new foundation**, use the existing manifest-aware wrapper. Its historical
filename does not restrict its species manifest:

```bash
export PYTHON="$(command -v python)"
export WORK_DIR="$FOUNDATION_ROOT/baseline"
export PRETRAIN_MODEL_PATH="$FOUNDATION_ROOT/shared/pretrain_orbax"
export METRIC_MODEL_PATH="$FOUNDATION_ROOT/baseline/metric_orbax"
export CENTROIDS_INIT_PATH="$FOUNDATION_ROOT/shared/centroids.npz"
export PRETRAIN=true
export RESUME=0
bash scripts/human_monkey_mouse_jax.sh 2>&1 | tee -a "$RUN_ROOT/foundation.log"
```

Expected outcome: pretrain checkpoint at the configured pretrain epoch, baseline
checkpoint at the configured metric epoch, `adata_pretrain.h5ad`, `final_adata.h5ad`,
complete loss histories and summaries under `$FOUNDATION_ROOT`. This stage is
new training and can require more than one bounded VM launch. Pretraining is
unlabelled; the labelled baseline and evaluation triplets use labels. Record
that exposure even though label-free fine-tuning itself excludes labels.

For **reuse**, point `FOUNDATION_ROOT` at the immutable compatible parent, and
skip the wrapper entirely. Restore the required foundation directory from its
verified bucket backup if absent; do not regenerate it implicitly. Confirm its
architecture, preprocessing, population, seeds and completed histories. Never
reuse a different population merely because its dimensions match.

Export the strict teacher artifact/triplets **once per foundation**, after both
stages complete. Skip this command if the compatible exports already exist:

```bash
python scripts/prepare_label_agnostic_artifacts.py \
  --pretrain-adata "$FOUNDATION_ROOT/baseline/saturn_results/adata_pretrain.h5ad" \
  --artifact "$FOUNDATION_ROOT/shared/label_free_artifact.npz" \
  --triplets "$FOUNDATION_ROOT/shared/evaluation_triplets.npz" \
  --metadata "$FOUNDATION_ROOT/shared/artifact_metadata.json" \
  --seed "$SEED" --batch-size "$METRIC_BATCH_SIZE"
```

Preserve an inventory before downstream work. For a reused foundation, verify
its original saved inventory first; hashing a changed parent now cannot prove
its historical identity. Capture this experiment's foundation inventory once:

```bash
python - <<'PY'
import json, os
from pathlib import Path
from jax_saturn.data.cache import file_sha256
foundation = Path(os.environ['FOUNDATION_ROOT'])
inventory = Path(os.environ['RUN_ROOT']) / 'foundation_manifest.json'
files = [{'path': str(p.relative_to(foundation)), 'sha256': file_sha256(p)}
         for p in sorted(foundation.rglob('*')) if p.is_file()]
assert files, 'Foundation is empty'
if inventory.exists():
    assert json.loads(inventory.read_text())['files'] == files, 'Foundation changed'
else:
    inventory.write_text(json.dumps({'files': files}, indent=2) + '\n')
print('Foundation inventory recorded/verified.')
PY
```

Keep every trial's output outside the parent foundation; copy its baseline
`saturn_results` into the new run so evaluation/reporting can write their own artifacts:

```bash
mkdir -p "$RUN_DIR/baseline"
test ! -e "$RUN_DIR/baseline/saturn_results" && \
  cp -a "$FOUNDATION_ROOT/baseline/saturn_results" "$RUN_DIR/baseline/"
```

Run that copy only when absent. For a parent outside `$RUN_ROOT`, keep its own
verified backup as well as this new experiment's backup.

## 9. VM — train the label-free trials with the shared harness

The current `run_hmmff_gpu.py` controller validates a particular saved population
and architecture; it is **not a generic experiment launcher**. The commands here
use the supported generic trainer. They preserve the same production warm-up
approach, with an operator timing gate. Generalizing the automated controller is
future harness work, not a capability implied by this runbook.

Define this function in the production shell. It always restores the latest
committed epoch when present, including optimizer/RNG/selected/adaptive state:

```bash
run_trial() {
  local objective="$1" target="$2" trial="$RUN_DIR/$1"
  local -a resume_args=()
  if compgen -G "$trial/checkpoints/epoch_*/metadata.json" > /dev/null; then
    local checkpoint
    checkpoint=$(python - "$trial" <<'PY'
import sys
from jax_saturn.distributed.checkpoint import latest_checkpoint
print(latest_checkpoint(sys.argv[1]))
PY
)
    resume_args=(--resume "$checkpoint")
  fi
  python -u scripts/train_label_agnostic_jax.py \
    --objective "$objective" --artifact "$FOUNDATION_ROOT/shared/label_free_artifact.npz" \
    --pretrain-checkpoint "$FOUNDATION_ROOT/shared/pretrain_orbax/checkpoints/$(printf 'epoch_%04d' "$PRETRAIN_EPOCHS")" \
    --output-dir "$trial" --device gpu --seed "$SEED" --epochs "$target" \
    --batch-size "$METRIC_BATCH_SIZE" --hidden-dim "$HIDDEN_DIM" --model-dim "$MODEL_DIM" \
    --mixed-precision "$MIXED_PRECISION" --neighbor-backend "$NEIGHBOR_BACKEND" \
    --query-block "$QUERY_BLOCK" --reference-block "$REFERENCE_BLOCK" \
    --cache-dir "$CACHE_DIR" "${resume_args[@]}" \
    2>&1 | tee -a "$RUN_ROOT/$objective.log"
}
run_all_trials() {
  local target="$1" objective
  for objective in infonce mmd ot; do
    run_trial "$objective" "$target" || return "$?"
  done
}
```

On a fresh trial, run two **real, checkpointed production epochs** each:

```bash
run_all_trials 2
```

Stop on any nonzero result; resolve it before evaluation or the next stage.
These epochs count toward the final target. On a resumed experiment, never set
a target below an existing checkpoint; skip this warm-up loop when trials have
already reached epoch 2 or later. Cache fixed graphs/positives/bandwidth across
trials. Model embeddings still receive fresh evaluation every epoch.

Before continuing, use `execution.jsonl` steady `epoch_complete` timings and
separate compilation events to record this gate:

```text
1.3 × (remaining setup and compilation
       + sum over objectives [(target − completed epochs) × steady epoch seconds]
       + final evaluation, figures and backup)
≤ time remaining before lastStartTimestamp + 6 hours
```

Read the actual start timestamp in `~/saturn-vm-start.txt`, not the time training
began. On the first 30-epoch launch, remaining epochs are 28 each. Include new
process/neighbor compilation separately; budget final UMAP/report work and backup
explicitly. If projection fails, back up and stop now; a later launch resumes
these useful checkpoints. There is no separate cloud benchmark and no automatic
runtime extension. The generic commands do not implement the specialized
controller's automatic gate/watchdog; GCE's six-hour STOP remains the hard ceiling.

Calculate and save the gate from the production records with this command. The
600-second setup and 3,600-second final-work reserves are estimates; increase
them deliberately if the experiment's reporting/backup work requires more time.
Missing timing evidence fails rather than guessing:

```bash
python - <<'PY'
import json, os, time
from datetime import datetime
from pathlib import Path
from jax_saturn.experiments.hmmff import committed_epoch
root = Path(os.environ['RUN_DIR'])
target = int(os.environ['LABEL_FREE_EPOCHS'])
start = datetime.fromisoformat(Path.home().joinpath('saturn-vm-start.txt').read_text().strip().replace('Z', '+00:00'))
details = {}
for objective in ('infonce', 'mmd', 'ot'):
    trial = root / objective
    completed = committed_epoch(trial)
    events = []
    for line in (trial / 'execution.jsonl').read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    steady = [e for e in events if e['event'] == 'epoch_complete' and e['epoch'] >= 2]
    assert completed >= 2 and steady, f'Missing checkpointed steady timing: {objective}'
    seconds = max(e['seconds'] - (e.get('first_update_seconds', 0) if e.get('invocation_first_epoch') else 0)
                  for e in steady[-2:])
    first_update = max(e.get('first_update_seconds', 0) for e in events if e['event'] == 'epoch_complete')
    kernels = {}
    for e in events:
        if e['event'] == 'neighbor_compilation' and e['dimension'] == int(os.environ['MODEL_DIM']):
            key = (e['k'], e['query_block'], e['reference_block'])
            kernels[key] = max(kernels.get(key, 0), e['seconds'])
    remaining = max(0, target - completed)
    details[objective] = {'completed_epochs': completed, 'remaining_epochs': remaining,
                          'steady_epoch_seconds': seconds,
                          'compilation_seconds': first_update + sum(kernels.values()) if remaining else 0}
work = sum(d['remaining_epochs'] * d['steady_epoch_seconds'] + d['compilation_seconds'] for d in details.values())
required = 1.3 * (600 + work + 3600)
available = start.timestamp() + 21600 - time.time()
record = {'objectives': details, 'setup_reserve_seconds': 600, 'final_reserve_seconds': 3600,
          'safety_factor': 1.3, 'required_seconds': required, 'available_seconds': available,
          'passes': required <= available}
Path(os.environ['RUN_ROOT']).joinpath('timing_gate.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
raise SystemExit(0 if record['passes'] else 3)
PY
```

If the projection passes:

```bash
run_all_trials "$LABEL_FREE_EPOCHS"
```

For an objective/temperature/weight ablation, create a new trial directory and
pass the changed supported CLI option; do not resume differently configured
checkpoints. The evaluator below expects the `baseline/infonce/mmd/ot` directory
layout, so give each complete comparison its own `RUN_DIR`. Changing objectives
or evaluator definitions beyond that layout requires adapting the report harness.

## 10. VM — evaluate and produce reports

After every trial reaches its full target:

```bash
python scripts/evaluate_label_agnostic_benchmark.py --root "$RUN_DIR" \
  --truth-adata "$FOUNDATION_ROOT/baseline/saturn_results/adata_pretrain.h5ad" \
  --triplets "$FOUNDATION_ROOT/shared/evaluation_triplets.npz" --seed "$SEED" \
  --neighbor-backend "$NEIGHBOR_BACKEND" --query-block "$QUERY_BLOCK" \
  --reference-block "$REFERENCE_BLOCK" --cache-dir "$CACHE_DIR" \
  2>&1 | tee -a "$RUN_ROOT/evaluation.log"
```

This generic evaluation writes evaluated `.h5ad` representations, cached UMAPs,
per-trial metrics/histories, `comparison.csv` and `acceptance.json`. The latter's
baseline-relative thresholds are implementation defaults; record whether they
answer the experiment's scientific question.

An executed notebook and PNG/PDF figures require an experiment-appropriate report
notebook. The existing presentation notebook contains fixed manifest/architecture
and report assumptions. Do not run it unchanged for another population. Configure
or adapt a report notebook locally before starting GPU compute, record its path
and use the same execution harness:

```bash
export REPORT_NOTEBOOK='notebooks/jax/my_experiment_report.ipynb'
python scripts/execute_saturn_notebook.py --notebook "$REPORT_NOTEBOOK" \
  --output "$RUN_DIR/executed.ipynb"
```

`REPORT_NOTEBOOK` is a file you prepare, not a supplied generic notebook. Its
report cells must read this run's completed representations, save PNG/PDF files
under `$RUN_DIR/report_assets`, and avoid retraining or altering the foundation.
The executor forwards streams/errors and saves a partial notebook after each
completed cell, including on failure. Report regeneration reuses representations;
changing report formatting is not a new training trial.

## 11. VM then LOCAL — verify, back up, download and stop

Before claiming completion, verify exact epoch coverage and checksum completion
records, not just the presence of an exported `.npz`:

```bash
python - <<'PY'
import os
from pathlib import Path
import pandas as pd
from jax_saturn.contracts.validation import validate_metric_history
from jax_saturn.experiments.hmmff import trial_complete
root = Path(os.environ['RUN_DIR'])
validate_metric_history(pd.read_csv(root / 'baseline/saturn_results/metric_history.csv'),
                        epochs=int(os.environ['METRIC_EPOCHS']))
for objective in ('infonce', 'mmd', 'ot'):
    validate_metric_history(pd.read_csv(root / objective / 'metric_history.csv'),
                            epochs=int(os.environ['LABEL_FREE_EPOCHS']))
    assert trial_complete(root / objective, objective,
                          epochs=int(os.environ['LABEL_FREE_EPOCHS'])), objective
for trial in ('baseline', 'infonce', 'mmd', 'ot'):
    for name in ('evaluated_adata.h5ad', 'umap.npz', 'metrics.json', 'history.csv'):
        assert (root / trial / name).is_file(), (trial, name)
for name in ('comparison.csv', 'acceptance.json'):
    assert (root / name).is_file(), name
import json
from jax_saturn.data.cache import file_sha256
foundation = Path(os.environ['FOUNDATION_ROOT'])
saved = json.loads((Path(os.environ['RUN_ROOT']) / 'foundation_manifest.json').read_text())
current = [{'path': str(p.relative_to(foundation)), 'sha256': file_sha256(p)}
           for p in sorted(foundation.rglob('*')) if p.is_file()]
assert current == saved['files'], 'Original foundation changed'
print('Training/evaluation artifacts and unchanged foundation verified; assess scientific acceptance separately.')
PY
```

When notebook/figure outputs are part of the experiment, also require the executed
notebook and all expected PNG/PDF files. Compare the parent's foundation hashes
with its pre-run inventory. Complete the experiment record with selected epochs,
metrics, timing gate decisions, source/environment identity and limitations.
Record **execution success separately from scientific acceptance**.

Stop periodic backups so they cannot race the final snapshot, then run one
verified final backup **on the VM**:

```bash
tmux kill-session -t saturn-backup
python scripts/backup_saturn_run.py --run-dir "$RUN_ROOT" --bucket-prefix "$BUCKET_PREFIX"
```

Download and verify **in the LOCAL terminal**, with the local configuration sourced:

```bash
cd /home/darcy/projects/SATURN-unsupervised
source "$SATURN_CONFIG"
mkdir -p "out/downloaded_$EXPERIMENT"
gcloud storage rsync --recursive "$BUCKET_PREFIX" "out/downloaded_$EXPERIMENT"
python scripts/backup_saturn_run.py --run-dir "out/downloaded_$EXPERIMENT" --verify-download
gcloud compute instances stop "$SATURN_VM" \
  --project="$SATURN_PROJECT" --zone="$SATURN_ZONE"
gcloud compute instances describe "$SATURN_VM" \
  --project="$SATURN_PROJECT" --zone="$SATURN_ZONE" --format='value(status)'
```

Require `TERMINATED`. Preserve the verified bucket/local copies. Once a completed
experiment's local download verifies, continue to step 13 and delete its VM and
disks. Retain the disk only while verification is pending or an authorized resume
needs it. Compute stopping does not remove storage charges. If download/verification
needs substantial time, first finish the verified bucket backup and stop the GPU;
complete the local download from the bucket while compute remains stopped.

## 12. Resume after interruption or return another day

1. Source the same configuration locally. Confirm retained disk, STOP and the
   six-hour limit; start the VM, capture its new timestamp, wait for SSH.
2. Deploy a newly verified archive only if source changed. Activate the retained
   environment, source the configuration on the VM and restart five-minute backups.
3. Restore missing files from a verified backup; fail rather than silently fit a
   replacement foundation. Check input/foundation/source identities before resuming.
4. If foundation training was interrupted, restore its completed native checkpoints:

```bash
export PYTHON="$(command -v python)"
export WORK_DIR="$FOUNDATION_ROOT/baseline"
export PRETRAIN_MODEL_PATH="$FOUNDATION_ROOT/shared/pretrain_orbax"
export METRIC_MODEL_PATH="$FOUNDATION_ROOT/baseline/metric_orbax"
export CENTROIDS_INIT_PATH="$FOUNDATION_ROOT/shared/centroids.npz"
export PRETRAIN=true
export RESUME=1
bash scripts/human_monkey_mouse_jax.sh 2>&1 | tee -a "$RUN_ROOT/foundation.log"
```

`RESUME=1` requires a committed pretrain checkpoint. If interruption occurred
before the first checkpoint, preserve the incomplete write and rerun that
unfinished epoch without `RESUME=1`. The wrapper also resumes baseline state if
its metric checkpoint directory exists; an empty/uncommitted directory must be
moved aside first. Completed pretraining updates are not repeated, though the
wrapper rebuilds/validates preparation and re-exports its representation.

For label-free work, redefine `run_trial` from step 9 and call it with the full
target. Existing completed updates are not repeated. Keep optimizer, RNG,
selected model and adaptive preservation state in `checkpoints/`; `selected_orbax`
contains inference exports and is not a resume source. Preserve incomplete epoch
directories outside their `checkpoints/epoch_NNNN` paths before retrying the
standalone trainer. Corrupt caches name the exact entry to remove/rebuild.

If only evaluation/reporting failed, repeat step 10 using completed representations.
Do not restart training. If leaving for the day, finish a backup, run the cloud
STOP command and confirm `TERMINATED`; detaching tmux alone keeps compute running.

## 13. LOCAL — retire the cloud resources permanently

Stopping a VM retains its disk and storage charges. The default for a completed
experiment is to delete its VM and disks after verifying the results; keep the
results bucket unless its deletion is separately requested. First verify the local downloaded backup with
`backup_saturn_run.py --verify-download` as in step 11. Keep that local copy.
The following commands are destructive operator actions; they are not part of
normal stop/resume. No GPU restart is needed.

Delete the VM and all of its attached disks, explicitly overriding their retained
auto-delete setting with [the deletion flag](https://docs.cloud.google.com/sdk/gcloud/reference/compute/instances/delete):

```bash
gcloud compute instances delete "$SATURN_VM" \
  --project="$SATURN_PROJECT" --zone="$SATURN_ZONE" --delete-disks=all
```

If retiring the whole dedicated SATURN bucket, this deletes **all cloud inputs
and backups in that bucket**, including other experiments. Complete their local
downloads first. Clear soft delete before deletion so new deletions are not
retained, and allow the policy change to propagate:

```bash
gcloud storage buckets update "gs://$SATURN_BUCKET" \
  --project="$SATURN_PROJECT" --clear-soft-delete
sleep 30
gcloud storage rm --recursive "gs://$SATURN_BUCKET" --project="$SATURN_PROJECT"
```

Recursive bucket deletion includes all object versions and the bucket itself
([CLI reference](https://docs.cloud.google.com/sdk/gcloud/reference/storage/rm)).
Objects already soft-deleted before that policy change retain their original
expiry and can accrue storage charges until then
([soft-delete policy and billing](https://docs.cloud.google.com/storage/docs/soft-delete)).

Inspect the project for orphaned resources from earlier attempts. Remove only
resources belonging to this work; an empty project can be kept for future runs:

```bash
gcloud compute instances list --project="$SATURN_PROJECT"
gcloud compute disks list --project="$SATURN_PROJECT"
gcloud compute snapshots list --project="$SATURN_PROJECT"
gcloud compute images list --no-standard-images --project="$SATURN_PROJECT"
gcloud compute machine-images list --project="$SATURN_PROJECT"
gcloud compute addresses list --project="$SATURN_PROJECT"
gcloud storage buckets list --project="$SATURN_PROJECT"
```

Deleting the VM does not automatically delete independent snapshots/images,
other detached disks, reserved IP addresses or other buckets. Use the displayed
names/scopes for any further targeted deletion, rather than bulk-deleting a
project that might contain unrelated work.

## Maintaining this golden runbook

When commands, environment pins, layout, caching, resume, timing or reporting
improve: update this file, then update the focused workflow guide and worked
examples where relevant. Keep experiment-specific facts in configuration and
records, and preserve historical records rather than rewriting past results.
Document implemented behavior and explicit gaps; do not present an experiment-
specific controller or notebook as generic until it accepts generic configuration.

Validate changes locally, with the GPU VM stopped:

```bash
python scripts/check_saturn_workflow_docs.py
python scripts/package_saturn_source.py
(cd dist && sha256sum -c saturn-gpu-exact.tar.gz.sha256)
```

The documentation checker checks shell/Python syntax, repository links and CLI
help. It does not provision cloud resources or establish that a new experiment
will finish within one six-hour launch. Cloud commands were checked against the
linked Google documentation; fresh provisioning is an operator action.

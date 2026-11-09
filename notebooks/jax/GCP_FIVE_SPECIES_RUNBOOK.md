# Five-species GPU-exact continuation: deployment handoff

For initial setup and new experiments, start with the
[canonical from-zero GCP runbook](../../GCP_RUNBOOK.md).

This is the worked example for the [reusable workflow](../../jax_saturn/EXPERIMENT_WORKFLOW.md)
and [experiment record](../../jax_saturn/EXPERIMENT_RECORD_TEMPLATE.md).
**The A100 stays stopped during local implementation and validation.** The steps
below are for the user after reviewing the source archive and validation receipt.
Implementation does not start, deploy to, or stop a cloud VM.

The saved run is
`out/human_monkey_mouse_frog_fish_jax_benchmark_walkthrough`. Preserve its epoch-20
unlabelled pretraining, epoch-30 labelled baseline, both representations, strict
teacher artifact and frozen evaluation triplets. Corrected work goes to
`out/human_monkey_mouse_frog_fish_jax_gpu_exact`. All five species/all 148,727
cells, 300 macrogenes, 384 hidden/embedding dimensions, seed 0, batch 512,
fp32, and 30 epochs per label-free objective remain fixed.

## 1. While the VM is stopped: obtain missing foundation files locally

```bash
cd /home/darcy/projects/SATURN-unsupervised
bash scripts/download_hmmff_foundation.sh
```

This selectively downloads the two completed checkpoint directories, required
representations/histories/summaries and teacher/triplet artifacts. It never copies
all species or starts the VM. It writes a checksum inventory. For new inputs,
use `stage_saturn_inputs.py` against the five-row manifest, as documented in the
reusable guide. Other species in the bucket can remain there.

## 2. Review the local handoff

The package command includes new source/docs/tests as well as tracked edits:

```bash
python scripts/package_saturn_source.py
cd dist
sha256sum -c saturn-gpu-exact.tar.gz.sha256
cd ..
```

Review `jax_saturn/verification/GPU_EXACT_RESULTS.md` and its receipts. The
archive excludes data, runs, environments, credentials and `.git`. The VM has
no Git checkout; deploy the archive, not `git pull`. Retained biological inputs
and environments remain on its protected boot disk.

## 3. Later, in the LOCAL terminal: confirm retention, STOP and six hours

```bash
export SATURN_PROJECT='macrogenes'
export SATURN_ZONE='asia-southeast1-a'
export SATURN_VM='saturn-a100-pilot'
export SATURN_BUCKET='macrogenes-saturn-artifacts'
gcloud compute instances describe "$SATURN_VM" \
  --project="$SATURN_PROJECT" --zone="$SATURN_ZONE" \
  --format='yaml(status,disks,scheduling.maxRunDuration,scheduling.instanceTerminationAction)'
```

Require status `TERMINATED`, boot disk `autoDelete: false`, max run duration
`21600s` and termination action `STOP`. If either retention/scheduling value
needs correction, apply it while stopped and inspect again:

```bash
gcloud compute instances set-disk-auto-delete "$SATURN_VM" \
  --disk="$SATURN_VM" --no-auto-delete \
  --project="$SATURN_PROJECT" --zone="$SATURN_ZONE"
gcloud compute instances set-scheduling "$SATURN_VM" \
  --max-run-duration=6h --instance-termination-action=STOP \
  --project="$SATURN_PROJECT" --zone="$SATURN_ZONE"
```

Do not extend the six-hour ceiling. Retained disks and bucket storage still
incur storage charges while GPU compute is stopped.

## 4. LOCAL: start, wait for SSH, transfer the verified package

```bash
gcloud compute instances start "$SATURN_VM" \
  --project="$SATURN_PROJECT" --zone="$SATURN_ZONE"
gcloud compute instances describe "$SATURN_VM" \
  --project="$SATURN_PROJECT" --zone="$SATURN_ZONE" \
  --format='value(lastStartTimestamp)' > /tmp/saturn-vm-start.txt

gcloud compute ssh "$SATURN_VM" \
  --project="$SATURN_PROJECT" --zone="$SATURN_ZONE" --command='true'
```

A brief `Connection refused` just after start can mean SSH has not finished
booting. Wait roughly 20 seconds and retry the last command. If it persists,
use the suggested `--troubleshoot` command; do not assume the upload failed.
Once SSH succeeds:

```bash
gcloud compute scp dist/saturn-gpu-exact.tar.gz \
  dist/saturn-gpu-exact.tar.gz.sha256 /tmp/saturn-vm-start.txt \
  "$SATURN_VM:~/" --project="$SATURN_PROJECT" --zone="$SATURN_ZONE"
gcloud compute ssh "$SATURN_VM" \
  --project="$SATURN_PROJECT" --zone="$SATURN_ZONE"
```

Deployment/warm-up have already begun consuming the six hours.

## 5. VM terminal: verify/install source and activate the retained environment

Local exports do not cross SSH. Set them again:

```bash
export SATURN_PROJECT='macrogenes'
export SATURN_ZONE='asia-southeast1-a'
export SATURN_VM='saturn-a100-pilot'
export SATURN_BUCKET='macrogenes-saturn-artifacts'
export SATURN_VM_START="$(cat ~/saturn-vm-start.txt)"
cd ~
sha256sum -c saturn-gpu-exact.tar.gz.sha256
cd ~/SATURN-unsupervised
tar --no-same-owner -xzf ~/saturn-gpu-exact.tar.gz
source .venv-saturn-gpu/bin/activate
python scripts/package_saturn_source.py --verify-tree source_manifest.json
python -m pip install 'nbclient==0.10.4' psutil
python -c 'import jax; assert jax.__version__ == "0.6.2"; print(jax.devices("gpu"))'
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR=/tmp/macrogenes-numba-cache
export MPLCONFIGDIR=/tmp/macrogenes-matplotlib
```

Reuse the retained JAX/CUDA/Flax/Optax/Orbax stack. Do not rebuild or broadly
upgrade it just to deploy source. Save the environment and current input hashes:

```bash
mkdir -p out/human_monkey_mouse_frog_fish_jax_gpu_exact
python -m pip freeze > out/human_monkey_mouse_frog_fish_jax_gpu_exact/environment.txt
python scripts/record_saturn_inputs.py \
  --manifest data/human_monkey_mouse_frog_fish.csv \
  --output out/human_monkey_mouse_frog_fish_jax_gpu_exact/input_record.json
cp jax_saturn/EXPERIMENT_RECORD_TEMPLATE.md \
  out/human_monkey_mouse_frog_fish_jax_gpu_exact/EXPERIMENT_RECORD.md
python scripts/run_hmmff_gpu.py preflight
```

Preflight must pass before launching any training. Missing/inconsistent foundation
files fail; the runner never substitutes fresh pretraining or baseline training.
Hashing current raw inputs documents current contents; the historical foundation
may lack original fit-time hashes, and that limitation remains in the record.

## 6. VM: start five-minute backups, then the single production command

Create a separate backup session:

```bash
tmux new-session -d -s saturn-backup \
  'cd ~/SATURN-unsupervised && source .venv-saturn-gpu/bin/activate && python scripts/backup_saturn_run.py --run-dir out/human_monkey_mouse_frog_fish_jax_gpu_exact --bucket-prefix gs://macrogenes-saturn-artifacts/runs/human_monkey_mouse_frog_fish_jax_gpu_exact --loop >> out/human_monkey_mouse_frog_fish_jax_gpu_exact/backup.log 2>&1'
tmux new-session -s saturn-production
```

Inside the production tmux shell:

```bash
cd ~/SATURN-unsupervised
source .venv-saturn-gpu/bin/activate
export NUMBA_CACHE_DIR=/tmp/macrogenes-numba-cache
export MPLCONFIGDIR=/tmp/macrogenes-matplotlib
export NUMBA_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
set -o pipefail
python -u scripts/run_hmmff_gpu.py run \
  --vm-start-time "$(cat ~/saturn-vm-start.txt)" \
  2>&1 | tee -a out/human_monkey_mouse_frog_fish_jax_gpu_exact/production.log
```

This is the **only training launch command**. It runs InfoNCE epochs 1–2, then
MMD 1–2, then OT 1–2, saving normal production checkpoints. It measures the
remaining work, gates, then resumes each objective at epoch 3 through 30 and
executes evaluation/report cells. A later launch starts above the latest
committed epoch rather than repeating epochs. No separate benchmark is needed.

**There is no manual launch between epoch 2 and epoch 30.** Step 6 contains the
whole production run, with these outcomes:

1. After two epochs of each objective, the runner writes the timing gate.
2. If the gate passes, it automatically resumes InfoNCE, MMD and OT through
   epoch 30, then runs evaluation, figures, the executed notebook and final
   checks. Leave the same command running throughout.
3. When it prints `execution_success` and returns to the shell, **skip step 7
   and continue at step 8** for final backup/download and VM shutdown.
4. If the gate fails, the deadline is reached, or execution is interrupted,
   follow step 7 instead. Preserve the saved production epochs for a later launch.

Epoch-2 `run_summary.json` and embedding exports are interim artifacts. Completing
training replaces those exports/summaries with the full 30-epoch result and
extends the history through epoch 30. This is why step 8 checks 30 epochs.
The selected model may come from an earlier epoch even when all 30 were trained.

Detach with `Ctrl-b`, then `d`. Reattach using `tmux attach -t saturn-production`.
A detached session keeps running. Terminal streams/errors, 25-second heartbeats,
phase timings and a partial executed notebook make progress observable.

The first-launch gate is:

```text
1.3 × (remaining setup and compilation
       + 28 × measured InfoNCE epoch time
       + 28 × measured MMD epoch time
       + 28 × measured OT epoch time
       + final evaluation, figures and backup)
≤ actual time before automatic shutdown
```

The recorded defaults reserve 600 seconds for setup and 3,600 for final work,
plus measured compilation. These estimates do not guarantee finishing; the
six-hour deadline remains authoritative. On failure the runner exits with code 3,
preserves useful checkpoints and prints the stop command. It never extends the
VM runtime or changes epochs/species/dimensions.

## 7. Recovery only — skip this step when step 6 succeeds

Use this branch if the timing gate fails or the run errors/is interrupted before
`execution_success`. A successful timing gate alone is not final completion;
training and reporting continue automatically in step 6.

Back up and stop from the local terminal:

```bash
gcloud compute instances stop "$SATURN_VM" \
  --project="$SATURN_PROJECT" --zone="$SATURN_ZONE"
```

For a later attempt, repeat start/SSH/timestamp capture. Deploy source only if it
changed, activate the same environment, preflight, restart backups and run the
same production command with the **new** actual start timestamp. Keep the output
and cache directories. A committed Orbax epoch contains optimizer, RNG, best
parameters, adaptive weight and history. `selected_orbax` is not a resume source.

If a write was interrupted before its `metadata.json` commit marker, the runner
preserves it under `interrupted/` and resumes from the latest committed epoch. A corrupt cache reports its exact entry; remove that derived entry to
rebuild it. Do not remove completed checkpoint epochs.

## 8. Verify outputs, final backup/download, complete record, stop

Start here after step 6 prints `execution_success` and exits successfully.
The runner has already continued beyond its two-epoch timing phase through the
full target and reporting. Do not start another training command.

Execution success requires all three label-free histories to contain exactly
1–30 with matching epoch-30 summaries/checkpoints; unchanged original foundation
hashes; `executed.ipynb`; PNG and PDF report figures; four evaluated `.h5ad`
representations, trial `.npz` embeddings, checkpoints, comparison metrics and
acceptance results. Scientific acceptance is reported separately in
`acceptance.json` and `execution_result.json`.

After production and periodic backup have finished, stop the backup loop so it
cannot race the final snapshot; then run the final one-off backup **on the VM**:

```bash
tmux kill-session -t saturn-backup
python scripts/backup_saturn_run.py \
  --run-dir out/human_monkey_mouse_frog_fish_jax_gpu_exact \
  --bucket-prefix gs://macrogenes-saturn-artifacts/runs/human_monkey_mouse_frog_fish_jax_gpu_exact
```

Finish the experiment record and repeat that final backup if you edited it.
**LOCAL terminal:**

```bash
mkdir -p out/downloaded_human_monkey_mouse_frog_fish_jax_gpu_exact
gcloud storage rsync --recursive \
  gs://macrogenes-saturn-artifacts/runs/human_monkey_mouse_frog_fish_jax_gpu_exact \
  out/downloaded_human_monkey_mouse_frog_fish_jax_gpu_exact
python scripts/backup_saturn_run.py \
  --run-dir out/downloaded_human_monkey_mouse_frog_fish_jax_gpu_exact --verify-download
gcloud compute instances stop "$SATURN_VM" \
  --project="$SATURN_PROJECT" --zone="$SATURN_ZONE"
gcloud compute instances describe "$SATURN_VM" \
  --project="$SATURN_PROJECT" --zone="$SATURN_ZONE" --format='value(status)'
```

Require `TERMINATED`. Keep the verified bucket/local copies. For a completed run,
continue to step 9 to delete the VM and disks after local verification. Keep the
disk temporarily only while verification is pending or an authorized resume needs it.
Notebook figures display in the executed notebook and exist separately as
PNG/PDF; representations and `.npz` files are separate artifacts too.

## 9. Permanent cloud cleanup after preserving the local results

`TERMINATED` stops compute but retains the protected boot disk and its charges.
When the experiment is complete and the local backup verifies, use the
[permanent retirement steps](../../GCP_RUNBOOK.md#13-local--retire-the-cloud-resources-permanently)
to delete the VM with its disks. Preserve the dedicated artifacts bucket unless
its deletion is separately requested. Keep the verified local downloaded run. No VM
restart is needed to perform this cleanup.

# Bounded single-host TPU pilot

The runnable pilot is `scripts/smoke_saturn_jax.py`. It needs synthetic inputs
only. Initial phase performs one training step, saves an Orbax checkpoint and
computes one expected following update. A separate resume process restores the
checkpoint and performs one update, checking parameters, optimizer, RNG and
loss against that reference. Both phases export validated unpadded embeddings.
There are three updates total, with no epoch-length training loop.

CPU validation is available; actual TPU execution remains unverified. Before
provisioning, fill in the project, account, billing cap, quota, storage destination
and maximum VM lifetime. No resources were created while preparing this runbook.

## Capacity and cost

Facts checked 2026-10-02: Google's [zone list](https://docs.cloud.google.com/tpu/docs/regions-zones)
lists v6e in Tokyo `asia-northeast1-b` and no Australian TPU zone. Its
[v6e hardware guide](https://docs.cloud.google.com/tpu/docs/v6e) identifies
`v6e-1` / `ct6e-standard-1t` as a one-chip testing VM. Use that for the first
hardware smoke; a later single-host four-chip pilot can exercise collectives.
Do not infer project quota or available capacity from this public listing.

The [pricing table](https://cloud.google.com/tpu/pricing) lists Tokyo v6e at
USD 3.24 per chip-hour on demand and USD 1.35 per chip-hour for Flex-start.
For planning, one chip for 30 minutes is USD 1.62 on demand, or USD 6.48 for
four chips for 30 minutes, before additional costs. These are arithmetic
estimates, not project-specific quotes or budget guarantees. Setup and idle
VM time count toward the rental lifetime. A command timeout stops its process;
it does not delete or stop the VM. Record creation/deletion times and verify
teardown. Budget notifications alone are not a hard spending limit.

Google now recommends [Compute Engine](https://docs.cloud.google.com/tpu/docs/tpus-in-compute-engine)
for direct VM management. The older Cloud TPU API is maintained for fixes.
Use the [current Compute Engine quickstart](https://docs.cloud.google.com/tpu/docs/quickstart-create-tpu-instance)
for provisioning, and confirm project permissions/quota first. This runbook
uses existing VMs for execution; resource creation needs a concrete approved
project and spending cap.

## Account and existing VM checks

Run read-only checks from a machine with the Google Cloud CLI:

```bash
export SATURN_PROJECT='project-id'
export SATURN_ZONE='asia-northeast1-b'
export SATURN_VM='existing-tpu-vm'
gcloud auth list --filter=status:ACTIVE
gcloud projects describe "$SATURN_PROJECT"
gcloud compute instances describe "$SATURN_VM" \
  --project="$SATURN_PROJECT" --zone="$SATURN_ZONE" --format=json
```

Check the chosen TPU consumption mode and quota in the project, following
the [quota guide](https://docs.cloud.google.com/tpu/docs/quota). Record account,
zone, machine type/chip count, consumption mode, permission to use the existing
VM, and the approved lifetime/cost cap. A legacy TPU VM must use its own API
for lifecycle operations; do not mix Compute Engine and legacy resource names.

## Isolated runtime

On the approved VM, stage this worktree and create a dedicated Python 3.10+
environment. From the repository root:

```bash
python3 -m venv .venv-saturn-tpu
.venv-saturn-tpu/bin/python -m pip install -r jax_saturn/requirements-tpu.txt
.venv-saturn-tpu/bin/python -m pip install 'numpy<2' pandas
.venv-saturn-tpu/bin/python -m pip freeze > tpu-environment.txt
```

[JAX installation documentation](https://docs.jax.dev/en/latest/installation.html)
uses the `jax[tpu]` extra. This repository pins JAX 0.6.2 and its tested
Flax/Optax/Orbax stack. Treat TPU driver/runtime compatibility as a smoke gate;
do not silently upgrade it if installation or initialization fails. Record the
VM image and installed versions. The synthetic command requires no Scanpy,
AnnData, gene embeddings, PyTorch, cloud bucket or biological input downloads.

## Smoke and separate-process resume

The one-chip example expects one JAX device. Confirm that expectation against
the actual platform/topology before the run; a mismatch fails immediately.
Choose a fresh local output directory. Both invocations use identical arguments:

```bash
export JAX_PLATFORMS=tpu
SATURN_PYTHON=.venv-saturn-tpu/bin/python
SATURN_OUTPUT=out/tpu-smoke-bf16
SATURN_ARGS=(--device tpu --expected-local-device-count 1
  --global-batch-size 16 --mixed-precision bf16 --output-dir "$SATURN_OUTPUT"
  --project "$SATURN_PROJECT" --zone "$SATURN_ZONE" --tpu-type v6e-1
  --chip-count 1 --usd-per-chip-hour 3.24 --quota-mode on-demand)
timeout --signal=TERM --kill-after=30s 600s \
  "$SATURN_PYTHON" scripts/smoke_saturn_jax.py --phase initial "${SATURN_ARGS[@]}"
timeout --signal=TERM --kill-after=30s 600s \
  "$SATURN_PYTHON" scripts/smoke_saturn_jax.py --phase resume "${SATURN_ARGS[@]}"
```

Read both receipt JSON files and require `success: true`, `platform: tpu`,
the expected topology and step counts 1 and 2. Keep receipts, environment,
checkpoint and NPZ artifacts together. `compute_chip_hours` covers measured
command compute only; use VM lifetime for cost accounting. Receipts do not
establish full benchmark acceptance. Repeat fp32 in a separate directory only
within the approved time/cost cap if numerical investigation needs it.

## Short biological pipeline after smoke

After hardware smoke/resume succeeds, stage the tiny atlas and portable caches
plus its CPU preprocessing/evaluation dependencies. Run one shared pretraining
epoch and one epoch per objective, not the full HMM reference lengths:

```bash
PYTHON=.venv-saturn-tpu/bin/python DEVICE=tpu \
  IN_DATA=data/saturn_run_tiny.csv OUT_DIR=out/tpu-tiny-pilot \
  PRETRAIN_EPOCHS=1 EPOCHS=1 HV_GENES=64 NUM_MACROGENES=8 \
  HIDDEN_DIM=16 MODEL_DIM=16 PRETRAIN_BATCH_SIZE=256 BATCH_SIZE=256 \
  PRETRAIN_DISTRIBUTED=1 METRIC_DISTRIBUTED=1 LABEL_DISTRIBUTED=1 \
  EXPECTED_LOCAL_DEVICE_COUNT=1 MIXED_PRECISION=bf16 \
  bash scripts/run_label_agnostic_benchmark_jax.sh
```

Include preprocessing, compilation, graph construction and evaluator time in
the approved lifetime. Preserve existing shared checkpoints and immutable trial
outputs. The 25-species pilot is separately gated: five embedding files listed
by `data/datatable.csv` are currently absent, so do not substitute a reduced
species manifest and call it the 25-species pilot.

## Export and teardown

Copy receipts, checkpoints and outputs to the approved durable storage path
before deleting a VM created for the pilot. Do not delete a pre-existing shared
VM without its owner's authorization. For an owned Compute Engine pilot VM,
the lifecycle commands are:

```bash
gcloud compute instances delete "$SATURN_VM" \
  --project="$SATURN_PROJECT" --zone="$SATURN_ZONE"
gcloud compute instances list --project="$SATURN_PROJECT" \
  --filter="name=$SATURN_VM"
```

Confirm the VM is absent and account for retained disks/storage separately.
No paid execution should be inferred from local smoke receipts.

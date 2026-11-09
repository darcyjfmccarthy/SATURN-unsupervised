#!/usr/bin/env bash
# Run inside saturn-production tmux on the fresh, six-hour-bounded VM.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export SATURN_CONFIG="${SATURN_CONFIG:-$(cat ~/saturn-experiment-config.txt 2>/dev/null || echo experiments/hmmff_cell_holdout.env)}"
source "$SATURN_CONFIG"
export SATURN_VM_START="$(cat ~/saturn-vm-start.txt)"
export SATURN_VM_DEADLINE="$(cat ~/saturn-vm-deadline.txt)"
mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/setup.log") 2>&1
rebooting=0
backup_pid=''
finish() {
  local status=$?
  trap - EXIT
  set +e
  if [[ "$rebooting" == 1 ]]; then return; fi
  if [[ -n "$backup_pid" ]]; then kill -- "-$backup_pid"; wait "$backup_pid"; fi
  printf '{"bootstrap_exit_code":%s,"automatic_restart_authorized":false}\n' "$status" > "$RUN_ROOT/cloud_exit_status.json"
  if [[ -f .venv-saturn-gpu/holdout-installed ]]; then
    source .venv-saturn-gpu/bin/activate
    python scripts/backup_saturn_run.py --run-dir "$RUN_ROOT" --bucket-prefix "$BUCKET_PREFIX" || \
      python scripts/backup_saturn_run.py --run-dir "$RUN_ROOT" --bucket-prefix "$BUCKET_PREFIX"
  elif command -v gcloud >/dev/null; then
    gcloud storage cp "$RUN_ROOT/setup.log" "$RUN_ROOT/cloud_exit_status.json" "$BUCKET_PREFIX/"
  fi
  sync
  sudo shutdown -h now
  exit "$status"
}
trap finish EXIT
python3 - <<'PY'
import datetime, os
start = datetime.datetime.fromisoformat(os.environ['SATURN_VM_START'].replace('Z','+00:00'))
remaining = start.timestamp() + 21600 - datetime.datetime.now(datetime.timezone.utc).timestamp()
assert remaining > 900, 'Insufficient time remains in the original six-hour window'
print('Original six-hour deadline:', (start + datetime.timedelta(hours=6)).isoformat(), flush=True)
PY
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv python3-pip curl tmux ca-certificates gnupg
if ! command -v gcloud >/dev/null; then
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor --yes -o /usr/share/keyrings/cloud.google.gpg
  echo 'deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main' | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list
  sudo apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y google-cloud-cli
fi
if ! nvidia-smi; then
  if [[ -f "$RUN_ROOT/driver_reboot_requested" ]]; then
    echo 'Driver still unavailable after the setup reboot.' >&2
    exit 1
  fi
  # Pin the system driver as well as the Python CUDA packages. The floating
  # production installer selected R615, which hung in A100 UVM initialization.
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential dkms "linux-headers-$(uname -r)"
  curl -fL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb --output /tmp/saturn-cuda-keyring.deb
  sudo dpkg -i /tmp/saturn-cuda-keyring.deb
  sudo apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-driver-pinning-580.178.04
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-drivers
  touch "$RUN_ROOT/driver_reboot_requested"
  rebooting=1
  sync
  sudo reboot
  exit 0
fi
nvidia-smi --query-gpu=name,driver_version,pci.bus_id --format=csv > "$RUN_ROOT/gpu_driver.txt"
cat /proc/driver/nvidia/version >> "$RUN_ROOT/gpu_driver.txt"
uname -a >> "$RUN_ROOT/gpu_driver.txt"
if [[ ! -f .venv-saturn-gpu/holdout-installed ]]; then
  python3 -m venv .venv-saturn-gpu
  source .venv-saturn-gpu/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -r jax_saturn/requirements-holdout-gpu.txt
  python -m pip install 'torch==2.14.1+cpu' --index-url https://download.pytorch.org/whl/cpu
  python -m pip check
  touch .venv-saturn-gpu/holdout-installed
fi
source .venv-saturn-gpu/bin/activate
python scripts/package_saturn_source.py --verify-tree source_manifest.json
python -m pip freeze > "$RUN_ROOT/environment.txt"
cp source_manifest.json "$RUN_ROOT/source_manifest.json"
cp "$SATURN_CONFIG" "$RUN_ROOT/experiment.env"
setsid python -u scripts/backup_saturn_run.py --run-dir "$RUN_ROOT" --bucket-prefix "$BUCKET_PREFIX" --loop > /tmp/saturn-holdout-backup.log 2>&1 &
backup_pid=$!
mkdir -p "$(dirname "$IN_DATA")"
gcloud storage cp "gs://$SATURN_BUCKET/inputs/${IN_DATA#data/}" "$IN_DATA"
extra_args=()
if [[ -n "${DISEASE_INPUT_BUNDLE:-}" ]]; then
  mkdir -p "$(dirname "$DISEASE_INPUT_BUNDLE")"
  gcloud storage cp "gs://$SATURN_BUCKET/inputs/${DISEASE_INPUT_BUNDLE#data/}" "$DISEASE_INPUT_BUNDLE"
  extra_args+=(--bundle-record "$DISEASE_INPUT_BUNDLE")
fi
python scripts/stage_saturn_inputs.py --manifest "$IN_DATA" --bucket "$SATURN_BUCKET" --direction download --execute "${extra_args[@]}"
if [[ -n "${DISEASE_INPUT_BUNDLE:-}" ]]; then
  python scripts/prepare_disease_cloud_inputs.py --verify-bundle "$DISEASE_INPUT_BUNDLE"
  mkdir -p "$RUN_ROOT/input_provenance"
  cp "$(dirname "$DISEASE_INPUT_BUNDLE")"/*.json "$(dirname "$DISEASE_INPUT_BUNDLE")"/*.csv* "$RUN_ROOT/input_provenance/"
fi
python scripts/run_cell_holdout_jax.py prepare
gcloud storage cp "gs://$SATURN_BUCKET/experiment_specs/$EXPERIMENT/input_record.json" "$RUN_ROOT/local_input_record.json"
gcloud storage cp "gs://$SATURN_BUCKET/experiment_specs/$EXPERIMENT/EXPERIMENT_RECORD.md" "$RUN_ROOT/EXPERIMENT_RECORD.md"
gcloud storage rsync --recursive "gs://$SATURN_BUCKET/experiment_specs/$EXPERIMENT/" "$RUN_ROOT/launch_record"
python scripts/record_saturn_inputs.py --manifest "$IN_DATA" --split "$RUN_ROOT/split/split.csv" --output "$RUN_ROOT/input_record.json"
python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['RUN_ROOT'])
local = json.loads((root / 'local_input_record.json').read_text())
vm = json.loads((root / 'input_record.json').read_text())
assert local['manifest_sha256'] == vm['manifest_sha256']
assert local['inputs'] == vm['inputs'], 'Actual VM inputs differ from staged local inputs'
assert local['split']['sha256'] == vm['split']['sha256'], 'VM train/test split differs'
PY
python scripts/run_cell_holdout_jax.py preflight
python -u scripts/run_cell_holdout_jax.py run --vm-start-time "$SATURN_VM_START"

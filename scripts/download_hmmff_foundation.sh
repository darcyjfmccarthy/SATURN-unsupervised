#!/usr/bin/env bash
# Run on your local machine. This never starts or changes a Compute Engine VM.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$ROOT/out/human_monkey_mouse_frog_fish_jax_benchmark_walkthrough}"
SOURCE='gs://macrogenes-saturn-artifacts/runs/human_monkey_mouse_frog_fish_jax_benchmark_walkthrough'
mkdir -p "$DEST/shared" "$DEST/baseline/saturn_results"
for checkpoint in \
  shared/pretrain_orbax/checkpoints/epoch_0020 \
  baseline/metric_orbax/checkpoints/epoch_0030; do
  mkdir -p "$DEST/$checkpoint"
  gcloud storage rsync --recursive "$SOURCE/$checkpoint" "$DEST/$checkpoint"
done
for name in label_free_artifact.npz evaluation_triplets.npz artifact_metadata.json; do
  gcloud storage cp "$SOURCE/shared/$name" "$DEST/shared/$name"
done
for name in adata_pretrain.h5ad final_adata.h5ad metric_history.csv metric_summary.json pretrain_losses.csv pretrain_summary.json; do
  gcloud storage cp "$SOURCE/baseline/saturn_results/$name" "$DEST/baseline/saturn_results/$name"
done
for name in gpu-environment.txt source.ipynb notebook-execution.log; do
  gcloud storage cp "$SOURCE/$name" "$DEST/$name"
done
python3 - "$DEST" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve()
records = []
for path in sorted(root.rglob('*')):
    if path.is_file() and path.name != 'preservation_manifest.json':
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        records.append({'path': path.relative_to(root).as_posix(), 'size': path.stat().st_size, 'sha256': digest.hexdigest()})
(root / 'preservation_manifest.json').write_text(json.dumps({'schema_version': 1, 'files': records}, indent=2) + '\n')
print(f'Foundation downloaded to {root}; recorded {len(records)} files. The VM remains stopped.')
PY

# Australia-Preferred TPU Capacity Spec

## Status
Runbook and bounded smoke/resume CLI prepared. Actual cloud project, budget,
quota and TPU execution remain pending; no cloud resources were created.

## Notebook Scope
This spec covers the future TPU rental and execution environment for notebook cells 7, 16, 18, 20, and 22. It does not alter notebook analysis or report cells.

## Current PyTorch Contract
The current notebook assumes local CUDA:
- `DEVICE="cuda"`
- `DEVICE_NUM=0`
- Fresh rebuild raises an error if `torch.cuda.is_available()` is false.

No current repo script provisions cloud resources.

## JAX Design Target
The future JAX workflow should prefer TPU capacity near Australia while remaining realistic about current public TPU availability.

Current public facts rechecked on 2026-10-02:
- Google Cloud TPU regions and zones page lists no Australia TPU zone.
- The same page lists APAC v6e in `asia-northeast1-b`.
- The same page lists APAC v2 in `asia-east1-c`.
- TPU pricing lists Trillium/v6e Tokyo `asia-northeast1` at USD 3.24 per chip-hour on demand and USD 1.35 per chip-hour for DWS Flex-start.
- TPU pricing lists v5e Singapore `asia-southeast1` at USD 1.56 per chip-hour on demand and USD 0.60 per chip-hour for DWS Flex-start.
- TPU regions and zones page does not list v5e in `asia-southeast1`, so Singapore v5e must be verified with `gcloud` before planning a run there.

Official sources:
- TPU regions and zones: https://docs.cloud.google.com/tpu/docs/regions-zones?hl=en
- TPU pricing: https://cloud.google.com/tpu/pricing?e=13802955
- TPU quotas: https://docs.cloud.google.com/tpu/docs/quota?authuser=00

Capacity preference order:
1. `asia-northeast1-b` v6e, because it is the nearest current public modern TPU zone listed for APAC.
2. `asia-southeast1` v5e only if `gcloud` confirms TPU VM availability for the project and desired accelerator type.
3. `us-west4-a` v5e Flex-start as cheapest modern fallback with public region support.
4. `us-central1-a` v5e Flex-start as broad-capacity fallback.

First TPU target:
- v6e-1 in Tokyo for the bounded hardware smoke; v6e-4 for a later
  single-host collective pilot if needed and approved.
- Use only if quota, price, and availability are acceptable.

Google's current docs recommend Compute Engine for direct TPU VM management;
the legacy Cloud TPU API is maintained for fixes. See
`TPU_RUNBOOK.md` for current official sources, setup, three-update smoke,
separate-process resume, cost accounting and teardown. The candidate runtime
is pinned in `../requirements-tpu.txt` and is not claimed TPU-verified.

Cheap fallback:
- v5e Flex-start in a verified supported zone.
- The future code must be resumable before using Spot or Flex-start for long jobs.

Required cloud storage layout:

```text
gs://{bucket}/saturn-jax/
  data/
    manifests/
    anndata/
    gene-embedding-cache/
    centroid-cache/
  runs/
    {run_id}/
      config/
      checkpoints/
      artifacts/
      logs/
      profiler/
```

Required runtime checks:
- Confirm project ID.
- Confirm active account.
- Confirm region and zone.
- Confirm TPU quota for the chosen accelerator type.
- Confirm GCS bucket read/write.
- Confirm `jax.device_count()` equals expected device count.
- Confirm a one-step smoke run completes before full HMM training.

Quota posture:
- Quota is regional/zonal and accelerator-type specific.
- Future runbook must check both on-demand and Spot/Flex-start quota.
- Quota request should be made before relying on any high-dimensional 25-species run.

Cost posture:
- Pilot jobs should use Flex-start or Spot only after checkpoint/resume tests pass.
- On-demand is acceptable for first tiny smoke if it avoids queue complexity.
- Record actual chip-hours and artifact path in run metadata.

## Decisions
- Prefer APAC over US because the user is in Australia.
- Treat Tokyo v6e as the first credible APAC modern TPU target.
- Treat Singapore v5e as price-visible but availability-unconfirmed.
- Do not design around Australia TPU capacity until Google lists an Australia TPU zone or project-specific capacity is confirmed.
- Keep provisioning commands out of this repo until specs are accepted; future implementation can add a runbook or scripts.

## Non-Goals
- Creating Google Cloud resources during this SDD step.
- Choosing a GCP project or billing account.
- Long-term reservations before code passes tiny and HMM smoke tests.
- Running training from Australia on local hardware as a substitute for TPU planning.

## Acceptance Criteria
- Future cloud runbook can choose a zone without revisiting architecture.
- Future specs and scripts fail fast if the requested TPU zone/type is unavailable.
- Future run metadata records region, zone, TPU type, device count, pricing assumption, and quota mode.
- The first paid TPU run is a smoke test, not the full high-dimensional experiment.

## Open External Facts
- GCP project ID.
- Billing account and hard budget.
- Actual TPU quota for `asia-northeast1-b` v6e.
- Whether `asia-southeast1` v5e TPU VMs are available to this project despite pricing visibility.
- Preferred bucket location, likely `asia` multi-region or nearest acceptable region.
- Whether university/cloud program credits or TPU Research Cloud access are available.

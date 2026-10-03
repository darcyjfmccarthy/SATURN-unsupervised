# Bounded real-HMM baseline receipts

These receipts were captured on 2026-10-02 with JAX 0.6.2 and PyTorch 2.12.0.
Their source hashes identify the same reference artifact, frozen triplets and
pretraining checkpoint. They cover sampled updates, with dropout disabled, at
batch size 512 and macrogene/hidden/model dimensions 200/256/256.

| Receipt | Reference/JAX | XLA flags | Result |
| --- | --- | --- | --- |
| `hmm_baseline_cpu.json` | CPU/CPU | none | Three updates pass |
| `hmm_baseline_cuda_fusion_failure.json` | CUDA/CUDA | `--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0` | One entry fails at update one |
| `hmm_baseline_cuda_no_fusion.json` | CUDA/CUDA | `--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0 --xla_disable_hlo_passes=priority-fusion` | Three updates pass |

CUDA commands also used `CUBLAS_WORKSPACE_CONFIG=:4096:8` and a wall-clock
timeout. The passing no-fusion run had a two-minute cap. The receipts predate
the verifier's addition of an `xla_flags` field, so the invocation flags are
recorded here. See `../RESULTS.md` for coverage and limits.

Two pretraining failure receipts are also preserved:
`hmm_pretrain_fresh_cpu_failure.json` checks fresh reference initialization;
`hmm_pretrain_trained_cuda_failure.json` checks trained weights with reset Adam
moments. Both use the corrected saturated softplus rule and unchanged numeric
tolerances. The latter records the same no-priority-fusion flags as the passing
baseline above. These are unresolved gates, not passing pretraining evidence.

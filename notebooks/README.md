# JAX SATURN notebooks

The current benchmark notebooks use the JAX backend:

- `human_monkey_mouse.ipynb`: shared pretraining, labeled baseline, InfoNCE,
  MMD, OT, evaluation and report figures. Defaults to 20 pretraining epochs and
  30 metric epochs, with fp32 training on CUDA.
- `human_monkey_mouse_smoke.ipynb`: the same pipeline with 20 pretraining epochs
  and 5 metric epochs, plus a repeatability manifest of logical Orbax checkpoint
  contents and public artifacts. Compare with another JAX smoke run.
- `label_agnostic_benchmark.ipynb`: plots a completed JAX shell benchmark.
- `human_monkey_mouse_venous_static.ipynb`: plots a completed JAX walkthrough.

Use a kernel with the dependencies in `jax_saturn/requirements.txt` and the
appropriate CUDA or TPU requirements file. Trainers use the kernel's Python
interpreter. The walkthrough configuration supports `DEVICE` and `DEVICE_NUM`;
its default is CUDA device 0. Training seeds are passed to the JAX trainers.

Checkpoints are native Orbax directories, and centroid caches are portable NPZ
files. Public AnnData, evaluation artifacts and plotting contracts retain their
existing formats. Legacy `.pt` protein inputs are converted by the backend;
these inputs still require PyTorch for conversion.

The walkthrough writes to `out/human_monkey_mouse_jax_benchmark_walkthrough`.
Smoke runs write to `out/human_monkey_mouse_jax_benchmark_smoke_runs/<run-id>`.
The comparison viewer defaults to `out/human_monkey_mouse_jax_benchmark`, matching
`scripts/run_label_agnostic_benchmark_jax.sh`. Set `HMM_REPORT_SOURCE_DIR` or
`LABEL_AGNOSTIC_OUT` to inspect another completed run.

The port was checked statically without executing notebook cells. Saved outputs
from the previous backend were cleared. Matching artifact contracts do not
establish identical training trajectories or numerical results.

Older paper vignettes in `Vignettes/` and the protein embedding generation
notebook are outside the current HMM migration plan. The protein models retain
their upstream PyTorch implementation.

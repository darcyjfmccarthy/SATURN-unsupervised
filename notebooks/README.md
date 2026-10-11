# SATURN notebooks

Notebooks are separated by training backend:

- `pytorch/`: original notebooks restored from Git commit `27ed84d`, before the
  JAX migration. These use `train-saturn.py`, `train_label_agnostic.py`, `.pt`
  model checkpoints, and pickle centroid caches.
- `jax/`: notebooks from the JAX migration, using JAX trainers, native Orbax
  checkpoints, and portable NPZ centroid caches.

Each directory contains the full `human_monkey_mouse.ipynb` walkthrough, the
smoke notebook, the static venous plotting notebook, and the completed benchmark
viewer `label_agnostic_benchmark.ipynb`. The four original PyTorch notebooks were
restored without modifying their historical contents, including saved outputs.

## Human, monkey, mouse, frog and fish

Open [the PyTorch five-species walkthrough](pytorch/human_monkey_mouse_frog_fish.ipynb)
for the requested experiment. It adds the local frog (`x_laevis`) and zebrafish
(`d_rerio`) atlases to the three mammalian inputs using
`data/human_monkey_mouse_frog_fish.csv`. All four trials (baseline, InfoNCE, MMD,
and OT) fine-tune for **5 epochs**, after **20 pretraining epochs**.

The PyTorch copy writes to
`out/human_monkey_mouse_frog_fish_pytorch_benchmark_walkthrough`.
The [JAX copy](jax/human_monkey_mouse_frog_fish.ipynb) is retained separately and
writes to `out/human_monkey_mouse_frog_fish_jax_benchmark_walkthrough`.
Set `HMMFF_REPORT_SOURCE_DIR` to regenerate either copy's report from a completed
run using its backend's artifacts.

Broad-category ontologies are available for the three mammals. Frog and fish
appear as `unknown` in broad-category plots, and unknown anchor cells are
excluded from broad-category preservation scores. Fine-label evaluation includes
all five species.

## Running the notebooks

Use a kernel with the backend's dependencies: the repository `requirements.txt`
for PyTorch, or `jax_saturn/requirements.txt` and the appropriate CUDA or TPU
requirements for JAX. Run the PyTorch notebooks from the SATURN Python environment
so the shell baseline's `python` resolves to that environment. The walkthroughs
locate the repository by searching parent directories, so either notebook
subdirectory can be the working directory.

The original full walkthrough defaults to 30 fine-tuning epochs; smoke runs use
5. CUDA device 0 is the default. Use `HMM_REPORT_SOURCE_DIR` for the original
three-species report notebooks and `LABEL_AGNOSTIC_OUT` for completed benchmark
viewers. PyTorch and JAX checkpoints are not interchangeable.

The new five-species notebooks and the JAX migration have been checked
statically; training has not been executed as part of these changes.
Older paper vignettes in `Vignettes/` and protein embedding generation notebooks
remain in their original locations.

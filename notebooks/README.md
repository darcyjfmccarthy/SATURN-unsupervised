# SATURN notebooks

For local disease-data preparation, open
[the CELLxGENE disease notebook](local/disease_cellxgene_preparation.ipynb) and
[its experiment protocol](local/DISEASE_EXPERIMENT.md). It adds matched-control
mouse disease studies to the five-species inputs, audits expression scale and
batch confounding, and prepares the full augmented population for the existing
cell holdout. Donor-fold files are reserved for optional later validation.
The [RDEB report notebook](jax/human_monkey_mouse_frog_fish_rdeb_skin_holdout.ipynb)
compares fresh pretraining, baseline and InfoNCE; use
`experiments/hmmff_rdeb_skin_cell_holdout.env`.

Start here: [canonical from-zero GCP runbook](../GCP_RUNBOOK.md).

Reusable GPU experiment workflow, resume/holdout policies and records: [experiment guide](../jax_saturn/EXPERIMENT_WORKFLOW.md).

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

## Combined local experiment report

Open [the five-species experiment report](jax/five_species_experiment_report.ipynb)
for the completed 300/384/384 full-data experiment and the strict 200/256/256 cell
holdout, with charts, tables and plain-language conclusions linked to the earlier
COMP9991 report. Saved outputs are included. Run All uses local CSV/JSON records
and saved figures only; it does not load models, run inference or access the cloud.
The original report's 20 fine-tuning epochs are author-confirmed. Epoch counts for
the two new experiments come from saved histories; selected checkpoints are
reported separately, with a source-file audit.
Regenerated PNG/PDF charts and an input-checksum manifest are written to
`out/five_species_experiment_report/`.

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
For the Google Cloud GPU run, use this JAX copy: it runs **20 shared pretraining
epochs and 30 fine-tuning epochs per trial**, with fresh-run defaults restored to
**200 macrogenes**, **256 hidden dimensions**, and **256 embedding dimensions**.
Historical report mode reads saved checkpoint dimensions, including 300/384/384.
The PyTorch copy retains its 5-epoch configuration.
Set `HMMFF_REPORT_SOURCE_DIR` to regenerate either copy's report from a completed
run using its backend's artifacts.

Broad-category ontologies are available for the three mammals. Frog and fish
appear as `unknown` in broad-category plots, and unknown anchor cells are
excluded from broad-category preservation scores. Fine-label evaluation includes
all five species.

## Strict held-out cells on GCP

[The held-out five-species notebook](jax/human_monkey_mouse_frog_fish_holdout.ipynb)
uses a deterministic 90/10 split within each species, with test cells excluded
from preprocessing, training, graphs and selection. It restores 200/256/256,
fits a fresh 20-epoch foundation and runs four 30-epoch fine-tuning methods.
Training-fitted UMAPs project test points with `transform`; PNG/PDF figures and
interactive HTML highlight their placement. Held-out metrics use test queries
against training references, overall and per species. Source
`experiments/hmmff_cell_holdout.env` and use `run_cell_holdout_jax.py` as described
in the [cloud runbook](../GCP_RUNBOOK.md#strict-five-species-cell-holdout).
The authorized cloud run is one six-hour window including fresh-machine setup.
After a completed run's local download verifies, delete the VM and its disks using
the [retirement steps](../GCP_RUNBOOK.md#13-local--retire-the-cloud-resources-permanently).
Keep the results bucket unless its deletion is separately requested.

For completed runs, regenerate the generalization diagnostics without training:
`DEVICE=cpu NEIGHBOR_BACKEND=jax python scripts/run_cell_holdout_jax.py diagnostics`.
The `diagnostics/` directory contains train-versus-test comparisons, species and
cell-type generalization gaps, per-test-cell scores and an HTML explorer for
wrong cell-type votes. Training queries exclude themselves; held-out queries
use the same training reference. These measure cell holdout within the input
atlases, not transfer to unseen donors. Existing UMAPs and model selections are
reused. The notebook shows these diagnostics before the supplementary UMAPs.

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

Completed cloud runs retain their executed notebooks, environment records and
training histories in their experiment output directories.
Older paper vignettes in `Vignettes/` and protein embedding generation notebooks
remain in their original locations.

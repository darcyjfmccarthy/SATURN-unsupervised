# Local disease-data preparation

Open [disease_cellxgene_preparation.ipynb](disease_cellxgene_preparation.ipynb)
and use **Run All** in a local Python kernel. Start with the default mouse RDEB
skin preset; read [the experiment protocol](DISEASE_EXPERIMENT.md) alongside it.

The notebook downloads a version-pinned public CELLxGENE dataset, retains its
raw counts, audits gene/protein coverage and confounding, and writes five-species
normalized-expression inputs plus separate donor-held-out mouse inputs. All five
existing atlas `.X` matrices are normalized expression; raw study counts must not
be concatenated directly with them. The outputs document their scale explicitly.

Use the existing local `saturn` environment, or install
[requirements.txt](requirements.txt) in a local environment and select its kernel.
The notebook locates the repository from its working directory or a parent.
Network access is needed for the initial public download and live metadata check.

| Setting in the setup cell | Default | Alternative |
| --- | --- | --- |
| `PRESET` / environment `DISEASE_PRESET` | `rdeb_skin` | `post_flu_lung` |
| `LIVE_API` / `DISEASE_LIVE_API` | `1` | `0` for reviewed snapshot and cached file offline |
| `MATERIALIZE_FOLDS` | `(0,)` | `(0, 1)` to write both donor folds |
| `DEST` | `out/disease_pilot/PRESET/prep_seed0` | New directory when changing preparation settings |

Raw downloads total about 109 MB for skin and 1.613 GB for lung. Derived
reference copies and fold files need several additional GB; preparation reads
one reference species at a time and retains sparse matrices. The lung file has
much larger working-memory requirements. Preparation uses CPU and does not
start model training or cloud resources.

For the first disease experiment, use `DEST/exploration.csv`, containing the
original mouse atlas plus every retained WT and knockout study cell. Apply the
existing 90/10 cell split before fitting. `DEST/fold_0/train.csv` and the other
donor-fold outputs are reserved for the optional later donor-holdout experiment.
`DEST/external_prepared.h5ad` retains raw study counts separately.

For GCP, `scripts/prepare_disease_cloud_inputs.py` verifies the existing preparation
and copies the five normalized inputs and report metadata under
`data/disease_pilot/rdeb_skin/cell_holdout_seed0/`. It writes the portable manifest
`data/manifests/hmmff_rdeb_skin_cell_holdout.csv`. Source
`experiments/hmmff_rdeb_skin_cell_holdout.env` for baseline and InfoNCE only.
See the canonical [cloud runbook](../../GCP_RUNBOOK.md#rdeb-skin-disease-experiment).

Both presets were executed locally against the live API and audited across all
saved matrix rows, donor assignments, covariates and checksums. The
[validation receipt](validation_receipt.json) records the verified populations
and source hashes. Seven focused tests also pass. This validates preparation;
the disease-model experiment has not been trained.

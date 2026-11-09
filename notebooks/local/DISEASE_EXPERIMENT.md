# Add disease cells to the five-species SATURN experiment

## First experiment: add the cells and rerun the same pipeline

The first experiment is deliberately simple: **add healthy and diseased mouse
cells from one study to the existing mouse atlas, then rerun the same
five-species pipeline with fresh training**. Keep the existing methods,
settings, cell-holdout procedure and evaluation, and add plots showing where
the new cells land.

The initial question is whether diseased cells show an interesting shift within
a relevant cell type while healthy study cells align sensibly with the atlas.
This is an exploratory experiment. Whole-animal holdouts, disease classifiers,
additional training arms and ablations can follow if the initial result is
worth investigating; they are not required to get started.

## Connection to the current work

The most recent completed experiment is the **strict held-out-cell experiment
across five species**, documented in the
[held-out notebook](../jax/human_monkey_mouse_frog_fish_holdout.ipynb).
It excludes 10% of each species' cells before fitting and evaluates them with
frozen preprocessing and models. Use this workflow as the starting point,
retaining its cell split for the initial disease experiment.

The inputs shared by that experiment and the earlier full-data five-species
run are listed in `data/human_monkey_mouse_frog_fish.csv`:

| Species | Local atlas | Cells |
| --- | --- | ---: |
| Human, `h_sapiens` | `data/cell_atlases/h_sapiens.h5ad` | 51,881 |
| Mouse, `m_musculus` | `data/cell_atlases/m_musculus.h5ad` | 26,947 |
| Mouse lemur, `m_murinus` | `data/cell_atlases/m_murinus.h5ad` | 38,276 |
| Frog, `x_laevis` | `data/cell_atlases/x_laevis.h5ad` | 29,437 |
| Zebrafish, `d_rerio` | `data/cell_atlases/d_rerio.h5ad` | 2,186 |

Total: 148,727 cells. “Monkey” in the notebook names means mouse lemur.
The new cells remain `m_musculus` and use the existing mouse ESMC600 protein
embeddings. The manifest still has five biological species.

The [combined report](../jax/five_species_experiment_report.ipynb) compares the
earlier 300/384/384 full-data run with the most recent 200/256/256 cell holdout.
It favors InfoNCE among the label-free methods for integration, while the
supervised baseline retains higher absolute held-out label accuracy. This new
experiment extends that comparison to inputs containing disease-associated
states.

For execution, use the [canonical GCP runbook](../../GCP_RUNBOOK.md), including
its [strict five-species cell-holdout protocol](../../GCP_RUNBOOK.md#strict-five-species-cell-holdout),
and the [shared experiment workflow guide](../../jax_saturn/EXPERIMENT_WORKFLOW.md).
Data preparation remains local. These links describe the existing cloud
operations if training is later run on GCP.

## Dataset selection, checked 3 November 2026

The checked-in [metadata snapshot](cellxgene_candidates.json) records the public
catalog results and pinned version IDs. The notebook verifies version metadata
through the [official Discover API](https://api.cellxgene.cziscience.com/curation/ui/)
and streams the H5AD asset. No authentication is needed for these public files.

| Candidate | Raw cells; download | Replication and decision |
| --- | --- | --- |
| [Col7a1-knockout RDEB mouse paw skin](https://cellxgene.cziscience.com/collections/f5af7a2f-ab4c-4728-829e-48efb9562105), `rdeb_skin` | 9,816; 109 MB | **Default pilot.** WT1/WT2 and KO1/KO2; 4 animals, one 10x v3 assay. Same-study controls and existing mouse skin overlap. Very limited biological replication. |
| [Mouse Post-Flu Time Series](https://cellxgene.cziscience.com/collections/606dc65c-ebd8-4f93-ac86-d0789c04863f), `post_flu_lung` | 123,154; 1.613 GB | **Second exploratory experiment.** 23 exposed donors and only 2 healthy donors, both female (43 and 102 days old). Existing mouse lung/AT2 reference; age, sex, time, tracing and tamoxifen groups limit causal interpretation. More cells do not fix the shortage of healthy animals. |
| [Spinal cord injury microglia](https://cellxgene.cziscience.com/collections/3aff0465-9ec0-4a90-9673-71b6b32de532), `sci_microglia` | 5,178; 51 MB | Deferred. 15 listed donors, but nuclei, a new tissue, and microglia absent from the mammalian reference atlases make a separate cluster hard to interpret. Do not add the overlapping all-cell and microglia exports together. |
| [Mouse MASLD/HCC liver](https://cellxgene.cziscience.com/collections/9a85c2f4-ba6b-4b7a-afc4-b3e2ec99c17c), `masld_liver` | 21,715; 336 MB | Deferred. Four donor entries are **pooled experimental groups**, with diet, knockout and tumour context. Diet, genotype and tumour context complicate interpretation; defer this option. |

The [RDEB publication](https://doi.org/10.3389/fimmu.2023.1211505) provides
predeclared skin-state programmes. Its scRNA-seq experiment uses two knockout and
two wild-type animals. The [influenza publication](https://doi.org/10.1016/j.stem.2024.12.002)
studies recovery after viral injury, so a binary exposure label should not be
interpreted as identical pathology at every time point. Marker concordance with
the same source publication is a sanity check, not independent replication.

For a confirmatory study, obtain a balanced independent cohort with multiple
healthy and affected animals in each technical processing block. A power
calculation must use between-animal variation from the pilot, not the number of
cells. These two public studies alone cannot establish general disease effects
or conservation of the same disease across species.

## Prepare the combined inputs locally

Run [disease_cellxgene_preparation.ipynb](disease_cellxgene_preparation.ipynb)
with the local `saturn` kernel. Start with `PRESET='rdeb_skin'`. It downloads
pinned public data, extracts raw study counts, applies the declared QC, maps
genes to the existing mouse atlas and retains condition, donor, source and
technical covariates. Preparation does not launch model training or cloud jobs.

**Include both study WT and knockout cells.** These same-study healthy controls
are the main extra control needed for interpreting the initial run. With all
9,816 retained skin-study cells, the augmented input has 158,543 cells across
the same five species.

One necessary preparation detail is expression scale. All five existing atlas
`.X` matrices contain fractional normalized expression; only the lemur file
retains a raw-count layer. The notebook preserves study raw counts separately
and creates derived copies of all five inputs with a common per-cell sum of
10,000 over retained genes. Atlas and study mouse cells use the same shared
gene set. Original files remain untouched. This avoids directly mixing raw
study counts with normalized atlas values.

This is a comparison using normalized expression, consistent with the legacy
atlas inputs after the explicit rescaling. SATURN's inherited count likelihood
is therefore a compatibility surrogate, rather than a calibrated raw-count
model. Keep that limitation in the report. No new batch-correction model is
required for this first experiment.

The default output directory is `out/disease_pilot/rdeb_skin/prep_seed0/`:

| Output | Use |
| --- | --- |
| **`exploration.csv`** | **Full augmented five-species source manifest for the first experiment.** Apply the existing cell split to this input before fitted preprocessing. |
| `m_musculus_augmented.h5ad` | Mouse atlas plus healthy and diseased study cells, on the shared expression scale. |
| `reference_atlases/*.h5ad` | Derived reference inputs on the same per-cell scale. |
| `external_prepared.h5ad` | Separate raw study counts for later expression analysis. |
| `evaluation_metadata.csv.gz` | Mouse source, condition, donor and other covariates to join to embeddings by observation ID. |
| QC, gene-mapping, protein-coverage and sample-audit files | Input checks and possible confounding. |
| `preparation_record.json`, `notebook_execution.json` | Settings, versions and checksums. |
| `donor_folds.csv`, `fold_summary.csv`, `fold_0/*` | Prepared inputs for the **optional later donor-holdout experiment**. These are not the source for the first run. |

The filename `exploration.csv` means it contains the complete augmented
population; it does not require fitting on every cell. For the latest workflow,
create its normal 90/10 cell split from this manifest. A full-data rerun is also
possible, but should be described as transductive exploration.

Run the influenza preset separately if desired. Do not merge these unrelated
diseases into one binary disease label. A new preparation configuration needs
a new destination, as described in the [local setup guide](README.md).

## Run the existing pipeline on the augmented data

1. **Create the same 90/10 cell split within species**, now using the augmented
   input. Keep test cells out of fitted preprocessing, pretraining, graphs and
   checkpoint selection, exactly as in the most recent experiment. Cells from
   the same animal may occur on both sides; this remains a cell holdout.
2. **Fit a fresh shared SATURN foundation.** Refit the input-dependent
   preprocessing and caches for the changed mouse population and gene set.
   Use new output directories so the earlier experiment remains intact.
3. **Run two trials:** the supervised cell-type baseline and InfoNCE,
   both starting from that shared foundation. MMD and partial OT are excluded
   from this disease experiment following the earlier comparison.
4. **Run the existing evaluation and report**, then join the retained mouse
   metadata by observation ID and add the disease/source/donor plots below.

Use `experiments/hmmff_rdeb_skin_cell_holdout.env` with
`SATURN_TRIALS=baseline,infonce`. The cloud-ready source manifest is
`data/manifests/hmmff_rdeb_skin_cell_holdout.csv`; it is a byte-preserving portable
copy of the full augmented population, not a donor-fold subset.

Keep the latest run's settings: **200 macrogenes, 256 hidden dimensions,
256 embedding dimensions, 2,000 HVGs, batch 512, FP32, seed 0, 20 pretraining
epochs and 30 fine-tuning epochs**. Keep its checkpoint-selection rule and
existing metrics. Extra seeds and selection-rule comparisons can wait.

Disease condition is evaluation metadata. The supervised baseline uses the
same coarse cell-type label for healthy and diseased cells of that type.
Label-free fine-tuning continues to receive only `embeddings`, `macrogenes`,
`species` and `obs_ids`.

The method and workflow stay the same. Operationally, the runner needs a new
experiment configuration, the augmented source manifest, fresh output paths,
and validation against the new population counts and hashes. The existing
cell-holdout runner's completion checks refer to the earlier saved experiment;
review those assumptions before launching an augmented run. The preparation
notebook does not make an unchanged cloud command a validated launcher for the
new inputs. A donor-aware training driver is not needed for this first design.

## What to look for, including batch effects

Keep the existing integration and cell-type metrics. Add plots of the same
embedding coordinates colored by condition, source (atlas or external study),
donor, cell type, tissue, library size and mitochondrial fraction. Where the
existing workflow projects held-out cells, keep its training-fitted UMAP and
`transform` procedure.

For skin, first inspect **keratinocytes within skin**, where the mouse atlas
and new study overlap. Compare three groups: atlas mouse cells, study WT cells
and study knockout cells. Fibroblasts are also interesting within the study,
but the existing mouse skin subset has no fibroblast reference, so their
separation from atlas cells needs a different interpretation.

| Pattern | Initial interpretation |
| --- | --- |
| Study WT cells align with comparable atlas cells, while knockout cells shift | A promising disease-associated pattern to investigate. Check whether it occurs in both knockout animals. |
| Study WT and knockout cells move together away from the atlas | Consistent with a study, assay, age or tissue-site effect; disease alone does not explain the separation. |
| A distinct group contains mainly one animal or library | An unresolved donor or technical effect. |
| The shift changes between pretraining, baseline and label-free methods | Evidence about how each representation changes the pattern; investigate whether integration preserves useful biology. |
| No distinct disease cluster appears | A valid exploratory result. Disease may produce a continuous shift or change cell-type proportions instead. |

Compare like cell types and tissues. Separating keratinocytes from immune cells,
or all new study cells from the atlas, is not by itself disease-state discovery.
A cluster plot is a starting observation, not proof of biological utility.

Atlases can also contain batch effects. These local reference files retain
`organ` and `cellType` but lack donor/assay metadata, so their construction does
not eliminate that concern. The same-study WT controls, compatible expression
scale and donor/QC plots are sufficient controls to **start this exploration**;
they do not guarantee that every remaining difference is biological. If the
result is ambiguous, investigate it before adding a correction method. The
initial experiment does not require Harmony, scVI, a disease classifier or an
extra PCA benchmark.

A 90/10 cell split evaluates new cells from the exposed population. It does not
establish transfer to previously unseen animals. The small number of healthy
animals and the influenza study's age/sex structure also limit biological
claims. Retain these limitations when reporting an interesting pattern.

## Optional follow-up if the first result is interesting

The next question would be whether the pattern reproduces across animals and
agrees with relevant biology. At that point, consider whole-animal holdouts,
a within-study PCA comparison, disease probes, marker programmes, additional
input arms, extra seeds and targeted ablations. Those analyses answer stronger
questions than the initial augmented rerun.

The earlier detailed plan is retained in
[optional disease-state validation](DISEASE_VALIDATION_FOLLOWUP.md).
The notebook's donor-fold outputs support that later stage; they do not replace
the existing cell split in the first experiment.

## Reproducibility and checks

The notebook records versions, pinned data, QC, gene/protein coverage, metadata
and input/output checksums. Both presets have been executed and audited locally;
see the [validation receipt](validation_receipt.json). This verifies data
preparation, not a completed disease-model experiment.

Focused preparation checks can be run with:

```bash
/home/darcy/miniconda3/envs/saturn/bin/python -m unittest disease_pilot.test_preparation -v
```

Raw-count and feature-name conventions follow the
[CELLxGENE schema](https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/7.0.0/schema.md).

# Optional validation after the augmented five-species experiment

Start with [the simple experiment protocol](DISEASE_EXPERIMENT.md): add healthy
and diseased study cells to the mouse input and rerun the existing pipeline.
The analyses below are optional next steps if that first run reveals an
interesting pattern. Donor holdouts, disease probes, extra input arms,
correction methods and ablations are not prerequisites for the initial run.

This follow-up asks the stronger question of whether a disease-associated
pattern reproduces in previously unseen animals. Its donor-based fitting and
selection rules apply to this follow-up, rather than replacing the existing
cell-holdout design of the first experiment. The preparation notebook already
writes donor-fold inputs that can be used here later.

For cloud operations, use the [canonical GCP runbook](../../GCP_RUNBOOK.md)
and the [experiment workflow guide](../../jax_saturn/EXPERIMENT_WORKFLOW.md).

## Follow-up arms and split discipline

Use a staged experiment rather than demanding a full model comparison before
the data controls have been inspected.

**Stage A: establish the within-study signal.** On the skin study alone, fit a
simple count-normalize/log1p, training-variable-gene and PCA baseline using
training donors, then transform test donors. Evaluate fibroblasts as the
predeclared primary cell type and keratinocytes as secondary. Normalize each
cell by its own library size; fit any gene selection, centering, PCA and probe
parameters on training cells only. Assess library size, mitochondrial fraction,
age/site and sample composition alongside the signal. This establishes whether
there is an animal-reproducible state signal before cross-species alignment.

**Stage B: measure what SATURN preserves or erases.** For each fold and seed,
fit a **new** shared five-species foundation on `fold_N/train.csv`. Compare the
foundation representation with InfoNCE and the supervised cell-type baseline;
extend to MMD and partial OT as in the recent report. Keep the same foundation,
teacher features and evaluation population within each fold/seed. Never reuse
full-data centroids, HVGs, teacher graphs, bandwidth fits, optimizer states or
pretrained checkpoints from the earlier atlas runs for an inductive claim.

Use three input arms on the same measured feature inventory and explicit scale:
the five atlas references alone, references plus training-donor WT cells, and
references plus training-donor WT and disease cells. Fit a fresh foundation per
arm. The first arm controls for the new scale/feature preparation, and the second
controls for adding an independent study. The third is the main integration
experiment. Fit the disease probe on external training donors' **projected**
embeddings in every arm, including atlas-only; this probe does not update the
foundation. Thus a representation can be compared fairly whether or not it
saw external training cells during unsupervised fitting. Only training donors
enter the WT-only arm; do not accidentally include held-out WT animals.

**Stage C: sensitivity and negative controls.** Add an external-healthy-only
foundation arm with disease cells projected afterward, disclosing that study
animals' healthy cells contributed to fitting where applicable. If a strictly
healthy-reference experiment is claimed, its study training animals and all
fitted components must be specified afresh; disease training donors cannot
remain in pretraining or graph construction. Compare unadjusted and separately
adjusted embedding analyses, equal-donor/cell-type sampling, QC cutoffs, and
the two complementary donor folds. Keep these analyses distinct from primary
predeclared results. A fixed trained embedding with disease labels permuted
between whole animals is a useful null check; never shuffle individual cells
as if they were independent biological replicates.

For the RDEB pilot, add a feature ablation excluding the directly perturbed
`Col7a1` gene from every mouse input, rebuilding all fitted components with the
same donor assignments. Persistence of broader inflammatory programmes is more
informative than detecting the knockout transcript alone. Failure of this
ablation should be reported as a genotype-feature dependence, not hidden by
selecting a different cluster or marker panel.

Match the recent strict holdout's **200 macrogenes / 256 hidden / 256 embedding
dimensions**, 2,000 HVGs, batch 512, FP32, 20 pretraining epochs and 30
fine-tuning epochs. Use seeds 0, 1, 2; seeds quantify training variability and
are not biological replicates. Keep the prepared donor split at seed 0 and
vary model initialization/training seeds only. Hold preprocessing constant
across methods.
Retain the repository's current normalization/HVG policy for the primary
comparison and document any alternative count-based HVG policy as a separate
ablation; do not change global training behavior in this preparation task.

Only training animals may contribute to preprocessing, centroid fitting,
pretraining, teachers, positive/local/preservation graphs, bandwidths, epoch
selection or batch parameters. Per-cell QC with fixed thresholds and the
fixed measured-feature inventory do not fit test expression distributions.
Apply the saved gene order and frozen checkpoint to test expression. Fit UMAP on
training embeddings and use `transform` for test plots. Cell-type and disease
annotations return only in the evaluator for label-free arms; the strict
training artifact still has exactly `embeddings`, `macrogenes`, `species`,
`obs_ids`. The supervised baseline gets coarse cell-type labels only, with
identical labels in healthy and disease states, never condition/genotype.

No genuine inner validation cohort exists in this pilot. Freeze hyperparameters
and either use a fixed epoch (30 as primary) or a predeclared training-only
selection rule. Report the repository's mixing/teacher-recall-selected
checkpoint as a separate compatibility analysis if it differs. Do not optimize
selection or choose the best disease cell type using test AUROC, plots, or
marker agreement. Global mixing selection can otherwise reward overcorrection.

### Existing code and what still needs implementation to train

The preparation notebook and its generated manifests are runnable now. Training
the proposed experiment is a subsequent experiment, not an outcome claimed by
this deliverable. `scripts/train_saturn_jax.py --in_data FOLD/train.csv
--preprocessing-out RUN/fitted_preprocessing.json` accepts the generated
five-species training manifest. Set the architecture and epochs above and use
a fresh work directory for every fold/seed. Full training is substantial and
can be run on a local workstation GPU; the preparation notebook is CPU-only.

The existing `scripts/run_cell_holdout_jax.py` creates a **cell** split and its
population/completion checks assume the old experiment. Do not point it at
these inputs and call its outputs donor validation. A future donor experiment
driver should reuse `jax_saturn.data.holdout.save_fitted_preprocessing`,
`frozen_counts`, the frozen JAX inference methods in the existing cell holdout
workflow, and the same four-field artifact exporter, while consuming these
explicit donor assignments and separately joining evaluation metadata. The
current preprocessing/output drops most original `.obs` fields; the stable-ID
metadata join is necessary. The preparation does not silently change this
training contract or introduce an untested training command.

## Batch effects: identifiable controls first

Study, condition, donor, technical library, age, anatomic site, assay and cell
identity are different variables. Here the RDEB library identifier equals the
animal identifier, and each animal has only one condition. Donor-specific
technical effects cannot be separated from animal biology within that library.
The influenza data also have only two normal donors and additional time,
age and lineage-tracing structure. Inspect `sample_covariates.csv` and the
condition-by-library table before interpreting either result.

The primary comparison is **disease versus healthy within the external study**,
using whole-animal holdout and matching cell type and tissue. The original
atlas provides cross-species context, not the only healthy control. Plot study
WT and disease cells together with the atlas. If both study conditions move
together away from the atlas, the simplest supported explanation is a
study/site/age effect; label this unresolved rather than disease discovery.

The current autoencoder conditions on species, so it does not automatically
model study/library effects *within* mouse. Do not change species IDs to donor,
study or condition in an attempt to make it do so. Similarly, maximizing global
species mixing is not a sufficient correction criterion when species, tissue,
developmental stage and disease distributions differ.

As a transparent sensitivity analysis, estimate a study-level offset in frozen
embedding space using **training healthy controls and comparable atlas mouse
skin keratinocytes**. Apply the same fixed offset to healthy and disease study
cells, including held-out cells; never fit an offset from test animals or
regress out condition. Use predeclared healthy anchors only. This is an
annotation-assisted post-hoc sensitivity analysis, not a modification of the
label-free trainer. Report its anchor coverage and assumptions. It cannot
remove cell-type-specific/nonlinear batch effects or repair missing overlap;
there is no mouse-atlas skin fibroblast anchor in these inputs. A study-wide
translation leaves within-study geometry unchanged and mainly tests atlas
placement. Do not use it to claim identification of a disease effect.

More flexible corrections (for example Harmony, or scVI after raw reference
counts have been recovered) are optional future
comparators with training-only fitting and a valid held-out transform. They
must preserve condition and raw counts, and need the same controls. No method
can rescue a perfectly confounded condition/library design by assumption alone.
Benchmark literature explicitly measures both integration and preservation of
biology: [Luecken et al., Nature Methods](https://doi.org/10.1038/s41592-021-01336-8).

## Follow-up outcomes and interpretation

**Primary outcome for this optional follow-up:** train a fixed L2 logistic disease probe on frozen
**training-donor fibroblast embeddings** (`C=1`, standardized with training
statistics, inverse donor-cell-count weights), then score held-out animals.
Do not train this probe as part of the representation. Keep probe settings
fixed across representations. Report AUROC and balanced accuracy with each
animal weighted equally, donor mean predicted probability, and per-fold scores.
The within-study PCA baseline and unfine-tuned SATURN are essential comparisons.
If any held-out animal has fewer than 30 QC-passing fibroblasts, report that
fold/type as insufficient rather than silently switching the primary type.
Keratinocytes are the predeclared secondary outcome; other types are exploratory.

For the separate influenza experiment, predeclare alveolar type 2 epithelial
cells as the primary type and fibroblasts as secondary, using the published
`cell_type` and `author_cell_type` mapping. Keep the same 30-cells-per-animal
eligibility rule and report missing strata. Analyse each recorded
`sacrifice_day` separately and show age, sex, lineage-tracing and tamoxifen
groups; do not select the time point with the best separation. A pooled binary
exposure result is secondary and cannot establish a time-specific disease
effect. Both normal donors are female and their ages are 43 and 102 days;
exposed animals include both sexes and ages up to 446 days. Include a
female-only, overlapping-age sensitivity analysis and show its sample counts,
without treating that restriction as full matching. Long-term injury cannot
be separated reliably from ageing with these controls. With only two healthy
donors, this remains an exploratory replication
of the workflow in a different disease/tissue, not confirmation of the RDEB
biology or a conserved disease programme.

**State preservation and biological evidence:** compare disease-versus-WT
expression programmes within each cell type using raw-count donor pseudobulks
and separately normalized per-cell scores. Predeclare skin keratinocyte stress
genes (Krt6a, Krt16, Krt17), fibroblast inflammatory genes (C3, Ccl2, Cxcl1), and
matrix-remodelling genes (Col1a1, Acta2) before inspecting new embeddings. Report
which markers survive assay/protein/HVG selection. Do not require every gene
to change in the same direction or use the same data to choose a favourable
programme. Pseudobulk DE uses the full prepared raw counts with a disease term
and estimable covariates, not batch-corrected expression or SATURN embeddings.
With two animals per condition, emphasize effect sizes and individual-animal
consistency, and avoid a long list of definitive DE discoveries.

**Healthy alignment and identity:** use held-out healthy mouse-study queries
against training references; report atlas-study neighbor mixing within shared
mouse skin keratinocytes, healthy-only cross-species cell-type agreement where
homologous populations exist, and cell-type purity/silhouette. Match source
sizes or compare against the expected source proportions. Do not demand 50/50
mixing from unequal pools or treat mixed tissues as equivalent controls.

**Donor and batch sensitivity:** report each animal's state score, QC and sample
covariates. A donor classifier within condition is descriptive (this pilot
lacks enough same-condition training donors for strong generalization claims).
Examine whether clusters are dominated by one animal or library. If clustering
is used, choose its resolution from training data in advance and keep it fixed;
then assign/evaluate test cells without reclustering all observations together.
Show embeddings colored by condition, donor, source, tissue, type, library
size and mitochondrial fraction, using the same coordinates. UMAP separation
is a visualization, not the quantitative outcome.

**Composition:** separately report donor-level cell-type proportions; inflammation
may recruit a population without changing its intrinsic state. Stratify state
metrics by type and balance donors so this does not masquerade as a disease
embedding. Dataset construction and dissociation can also affect proportions.

For uncertainty, show animal-level values and the variation across folds/seeds.
Use donor-block bootstrap only as a descriptive sensitivity with its severe
small-sample limitation. A whole-animal exact permutation test in the skin
study has only `choose(4, 2) = 6` condition assignments, so it cannot yield
fine-grained p-values. If the label-dependent split/probe is part of the tested
procedure, recompute that procedure for each donor-label assignment; do not
permute test labels while pretending a selected supervised probe was fixed
independently. There are four animals, not 9,816 independent replicates.

A useful pilot result requires directionally consistent held-out animal
effects, comparison to the simple within-study baseline, retained cell identity
and relevant programmes, plus sensible healthy-control atlas placement. A
method that erases a reproducible within-study shift has failed state
preservation even if its species mixing increases. A cluster containing both
healthy and disease study cells supports a study effect; a single-donor cluster
supports an unresolved donor/library effect. If no reproducible disease signal
exists in the count/PCA baseline, do not demand that SATURN manufacture one.
These decision rules and all negative or ambiguous outcomes belong in the report.

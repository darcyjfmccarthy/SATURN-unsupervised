# JAX SATURN Spec

## Status
Implementation started with user authorization. Phases 1–6 host-side contracts,
Flax models, losses, CPU pretraining, labeled baseline and label-free objectives
are implemented with parity/integration tests. Full HMM scientific parity and
TPU execution remain pending.

## Notebook Scope
This spec covers the full execution path of `notebooks/human_monkey_mouse.ipynb`, not the whole repository.

In scope notebook cells:
- Cell 2: repository discovery, rebuild mode, GPU/device settings, canonical output paths, and `run_stage`.
- Cell 5: `data/human_monkey_mouse.csv` inspection and AnnData shape reporting.
- Cell 7: `scripts/human_monkey_mouse.sh`, which calls `train-saturn.py` for shared SATURN pretraining and labeled baseline metric learning.
- Cells 8, 10, 11: pretrain and baseline AnnData loading, UMAP, and benchmark metrics.
- Cells 13, 14: `scripts/prepare_label_agnostic_artifacts.py` strict label-free artifact and frozen evaluation triplets.
- Cells 16, 18, 20, 22: `scripts/train_label_agnostic.py` for `infonce`, `mmd`, and `ot`.
- Cell 24: `scripts/evaluate_label_agnostic_benchmark.py`.
- Cells 26 through 38: report helpers and figures that consume the stable output contract.

In scope Python modules:
- `train-saturn.py`
- `scripts/human_monkey_mouse.sh`
- `scripts/prepare_label_agnostic_artifacts.py`
- `scripts/train_label_agnostic.py`
- `scripts/evaluate_label_agnostic_benchmark.py`
- `model/saturn_model.py`
- `data/gene_embeddings.py`
- `data/multi_species_data.py`
- `label_agnostic/artifacts.py`
- `label_agnostic/objectives.py`
- `label_agnostic/metrics.py`
- `miners/triplet_margin_miner.py`
- `losses/triplet_margin_loss.py`
- `distances/cosine_similarity.py`
- `utils/loss_and_miner_utils.py`

## Current PyTorch Contract
The defended notebook runs a four-trial benchmark:

1. Labeled SATURN baseline.
2. Label-free InfoNCE fine-tuning.
3. Label-free MMD fine-tuning.
4. Label-free partial OT fine-tuning.

The shared HMM data manifest is `data/human_monkey_mouse.csv` with columns:
- `species`
- `path`
- `embedding_path`
- `in_label_col`

The defended HMM run uses 117,104 cells across:
- `h_sapiens`: 51,881 cells, 19,504 genes before filtering.
- `m_musculus`: 26,947 cells, 15,437 genes before filtering.
- `m_murinus`: 38,276 cells, 13,495 genes before filtering.

Notebook default hyperparameters:
- `SEED=0`
- `DEVICE=cuda`
- `DEVICE_NUM=0`
- `METRIC_EPOCHS=30`
- `BATCH_SIZE=512`
- `HV_GENES=2000`
- `NUM_MACROGENES=200`
- `MODEL_DIM=256`
- `HIDDEN_DIM=256`
- `PRETRAIN_EPOCHS=20`
- `PRETRAIN_BATCH_SIZE=512`
- `PRETRAIN_LR=0.0005`
- `METRIC_LR=0.001`
- `PE_SIM_PENALTY=0.2`
- `L1_PENALTY=0.0`
- `CENTROID_SCORE_FUNC=default`

The PyTorch implementation writes the following notebook-visible artifacts:
- `baseline/saturn_results/adata_pretrain.h5ad`
- `baseline/saturn_results/final_adata.h5ad`
- `baseline/saturn_results/metric_history.csv`
- `shared/pretrain_model.pt`
- `shared/centroids_seed0.pkl`
- `shared/label_free_artifact.npz`
- `shared/evaluation_triplets.npz`
- `{infonce,mmd,ot}/final_model.pt`
- `{infonce,mmd,ot}/final_embeddings.npz`
- `{infonce,mmd,ot}/metric_history.csv`
- `{infonce,mmd,ot}/run_summary.json`
- `{baseline,infonce,mmd,ot}/evaluated_adata.h5ad`
- `comparison.csv`
- `acceptance.json`

The PyTorch code remains the reference implementation for behavior, scientific outputs, and file contracts until JAX parity is explicitly accepted.

## JAX Design Target
The JAX port is parallel to the existing PyTorch implementation. The original
specification-only SDD pass is complete; implementation now follows the phases
in `migration_plan.SPEC.md`.

Future implementation areas:
- `jax_saturn/contracts`: CLI and artifact contracts.
- `jax_saturn/data`: manifest normalization, AnnData preparation, embedding cache conversion, centroid cache specification, and batching.
- `jax_saturn/models`: Flax modules for SATURN pretraining and metric models.
- `jax_saturn/losses`: JAX losses and objective functions.
- `jax_saturn/training`: pretraining, labeled baseline metric learning, and label-agnostic fine-tuning loops.
- `jax_saturn/distributed`: TPU execution, pmap, Orbax checkpointing, dtype policy, and multi-host rules.
- `jax_saturn/cloud`: Australia-preferred TPU rental plan and cloud facts.
- `jax_saturn/verification`: future parity and smoke tests.

The JAX implementation will use:
- JAX for array programming and compilation.
- Flax Linen for model definitions.
- Optax for optimizer definitions and gradient transformations.
- Orbax for JAX-native checkpointing.

The external notebook-facing outputs must stay compatible with the existing evaluator and report cells.

## Decisions
- Use spec-driven development. Implement the reviewed migration phases with contract and parity tests.
- Restrict v1 scope to code touched by `human_monkey_mouse.ipynb`.
- Keep CPU/scikit-learn/Scanpy preprocessing and evaluation in v1 unless a spec explicitly says otherwise.
- Treat PyTorch outputs as the scientific reference.
- Default local implementation validation to two epochs with shared
  pretraining. Matched short-run numerical checks and artifact/resume contracts
  allow implementation progress; full scientific runs are explicitly opt-in.
- Optimize for metric parity, not bitwise parity. Exact random trajectories may differ.
- Keep the current pretrain model's `spec_idx = 0` species one-hot behavior for first parity. A corrected species one-hot may be added later behind an explicit experiment flag.
- Add Orbax checkpoints for future JAX training and keep notebook-facing artifacts stable.
- Prefer APAC TPU capacity because the user is in Australia, while recording that current public TPU region docs list no Australia TPU zone.

## Non-Goals
- Porting unused SATURN losses, miners, transfer models, testers, or old vignettes.
- Rewriting plotting/report cells.
- Replacing AnnData, Scanpy, sklearn KNN, or UMAP in v1.
- Building a production training service.
- Running full training or provisioning cloud capacity during Phase 1.

## Acceptance Criteria
- Phase 1 adds host-side manifest/cache implementations and schema tests under `jax_saturn/`.
- Each spec uses the shared SDD template:
  - `Status`
  - `Notebook Scope`
  - `Current PyTorch Contract`
  - `JAX Design Target`
  - `Decisions`
  - `Non-Goals`
  - `Acceptance Criteria`
  - `Open External Facts`
- A future engineer can implement the JAX port without choosing architecture, output schemas, training contracts, or TPU posture.
- Existing PyTorch training and notebook behavior is unchanged.

## Open External Facts
- Google Cloud project ID.
- Billing account and budget.
- TPU quota per region and per accelerator type.
- Whether `asia-southeast1` actually has v5e TPU VM capacity for this project.
- Preferred GCS bucket names and retention policy.
- Whether a short reservation, Flex-start, Spot, or on-demand capacity is administratively acceptable.

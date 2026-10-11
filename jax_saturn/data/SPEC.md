# Data Spec

## Status
Manifest normalization, portable caches, CPU preprocessing and fixed-shape
pretraining batches implemented through Phase 4. Large-run staging and
distributed device transfer remain pending.

## Notebook Scope
This spec covers notebook cells 5, 7, 13, 14, and the data consumed by cells 24 through 38.

In scope source files:
- `data/human_monkey_mouse.csv`
- `data/datatable.csv`
- `data/saturn_run_tiny.csv`
- `data/gene_embeddings.py`
- `data/multi_species_data.py`
- `train-saturn.py`
- `label_agnostic/artifacts.py`

## Current PyTorch Contract
`data/human_monkey_mouse.csv` uses this schema:
- `species`
- `path`
- `embedding_path`
- `in_label_col`

`data/datatable.csv` uses this schema:
- `species`
- `atlas`
- `gene_embeddings`

`data/saturn_run_tiny.csv` uses the HMM-style schema:
- `species`
- `path`
- `embedding_path`
- `in_label_col`

`train-saturn.py` currently:
1. Reads the manifest with `species` as index.
2. Loads AnnData with `scanpy.read`.
3. Adds `obs["species"]`.
4. Builds species-prefixed labels in `obs["species_type_label"]`.
5. Factorizes labels into `obs["truth_labels"]`.
6. Factorizes reference labels into `obs["ref_labels"]`.
7. Loads `.pt` gene embedding dictionaries with `torch.load`.
8. Filters genes to those with embeddings for the active species.
9. Temporarily normalizes expression for HVG selection.
10. Runs `scanpy.pp.highly_variable_genes(..., flavor="seurat_v3")`.
11. Restores raw counts to `adata.X`.
12. Stacks selected gene embeddings in sorted species order.
13. Creates or loads centroid scores from a pickle file.
14. Builds `ExperimentDatasetMultiEqual` for pretraining and `ExperimentDatasetMulti` for output ordering.

Important ordering contract:
- Training species are sorted alphabetically with `sorted(species_to_adata.keys())`.
- Output observation names are appended in that sorted species order.
- `get_all_embeddings` and `get_all_embeddings_metric` emit embeddings in the same sorted species order.

Current data tensors:
- Gene expression per species: float32 dense matrix `[n_cells_species, n_hv_genes_species]`.
- Gene embeddings per species: float32 tensor `[n_hv_genes_species, protein_embedding_dim]`.
- Centroid weights: float32 tensor `[sum_hv_genes, num_macrogenes]`, then transposed/logged inside the model.
- Macrogenes: float32 tensor `[n_cells, num_macrogenes]`.
- Species codes in label-free training: int64 from `np.unique(species, return_inverse=True)`.

Current label-free artifact loader requires:
- No label arrays.
- Unique `obs_ids`.
- Matching row counts across `embeddings`, `macrogenes`, `species`, and `obs_ids`.

## JAX Design Target
Future data implementation must have a manifest normalization layer:

Input columns:
- `species`: required.
- `path` or `atlas`: required, normalized to `path`.
- `embedding_path` or `gene_embeddings`: required for this repo's local data, normalized to `embedding_path`.
- `in_label_col`: optional for `datatable.csv`; default to `cellType`.

Normalized manifest rows:
- `species: str`
- `path: Path`
- `embedding_path: Path`
- `in_label_col: str`
- `source_row_index: int`

AnnData handling:
- Continue to use AnnData and Scanpy on CPU.
- Preserve raw count matrix for training.
- Use `adata.layers["raw"]` only as a temporary CPU preprocessing detail.
- Store final dense arrays as float32 NumPy arrays before device transfer.
- If sparse AnnData input is encountered, densify only after HVG subsetting when possible.

Gene embedding cache conversion:
- Convert `.pt` dictionaries to JAX-safe cache before TPU runs.
- Cache format may be `.npz` for HMM/tiny and Zarr for large 25-species runs.
- Required arrays:
  - `gene_symbols`: string, shape `[n_genes]`, lower-case lookup key.
  - `embeddings`: float32, shape `[n_genes, protein_embedding_dim]`.
- Required metadata:
  - `schema_version`
  - `species`
  - `source_path`
  - `source_sha256`
  - `created_by`
  - `gene_symbol_case: "lower"`

Centroid cache format:
- Prefer a JAX-safe `.npz` or Zarr cache over pickle for future JAX runs.
- Required arrays:
  - `scores`: float32, shape `[total_hv_genes, num_macrogenes]`.
  - `centroids`: float32, shape `[num_macrogenes, protein_embedding_dim]`.
  - `all_gene_names`: string, shape `[total_hv_genes]`, formatted as `{species}_{gene}`.
  - `species_names`: string, shape `[n_species]`, sorted.
  - `species_gene_starts`: int64, shape `[n_species]`.
  - `species_gene_ends`: int64, shape `[n_species]`.
- Required metadata:
  - `schema_version`
  - `seed`
  - `score_func`
  - `hv_genes`
  - `num_macrogenes`
  - `embedding_cache_sha256`
  - `source_manifest_sha256`
- Future code must be able to import the current pickle cache for parity, then write the JAX-safe cache.

Batching:
- Pretraining batches are species-grouped batches equivalent to `multi_species_collate_fn`.
- Metric batches over macrogenes follow the same global observation order.
- TPU batches must be fixed-shape and padded with masks.
- Batch records must include:
  - `values`: float32 or bf16 device array.
  - `labels`: int32 for labeled baseline only.
  - `ref_labels`: int32 for labeled baseline only.
  - `species_code`: int32.
  - `global_index`: int64 or int32 if `n_cells < 2^31`.
  - `valid_mask`: bool.
- Padding rows must not affect losses, metrics, or output artifacts.

Data movement:
- CPU preprocessing writes arrays to host memory or cache.
- Device transfer happens per fixed batch or via preloaded arrays for small HMM runs.
- Large 25-species runs may memory-map host arrays and prefetch to device.

## Decisions
- Keep Scanpy HVG selection in v1 for scientific parity.
- Keep sklearn KMeans or existing centroid initialization in v1; do not port KMeans to TPU first.
- Normalize manifest schemas so future code can use both HMM and 25-species manifests.
- Use sorted species order as the canonical training/output order.
- Store species strings in artifacts, not only integer codes.
- Use NumPy/Pandas/AnnData for public artifacts; use JAX arrays only inside training.

## Non-Goals
- Implementing a new HVG algorithm.
- Streaming raw `.h5ad` directly from GCS into TPU device memory in v1.
- Supporting arbitrary manifest schemas beyond the two local patterns.
- Porting `ExperimentDatasetMultiEqualCT`.

## Acceptance Criteria
- Normalized HMM manifest reproduces the current sorted species order: `h_sapiens`, `m_murinus`, `m_musculus`.
- Normalized 25-species manifest loads all rows from `data/datatable.csv` and defaults `in_label_col` to `cellType`.
- Future JAX strict label-free artifact has exactly the same keys and dtypes as the PyTorch artifact.
- Future JAX pretrain and final AnnData preserve `obs_names` and `obs["species"]` order expected by the evaluator.
- Cache conversion is deterministic and records source hashes.

## Open External Facts
- GCS bucket for large AnnData and embedding caches.
- Whether GCS FUSE, local SSD copy, or explicit `gsutil` staging is preferred on TPU VMs.
- Whether the 25-species high-dimensional run should use `.npz` or Zarr for embedding and centroid caches.

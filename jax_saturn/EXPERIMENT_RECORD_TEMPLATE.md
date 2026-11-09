# SATURN experiment record

Copy this into the new run directory before execution. Fill actual values and
mark missing historical evidence explicitly. Link `input_record.json`,
`foundation.json`, `execution_options.json`, `execution.jsonl`, `timing_gate.json`
and the source manifest; never replace content hashes with a manifest filename.

## Question and comparison

- Experiment ID, owner, date:
- Scientific question and preregistered acceptance criteria:
- Parent experiment/foundation and immutable bucket prefix:
- Changed variables; variables held fixed:
- Trial IDs and controls:
- Execution success (separate from scientific acceptance):

## Inputs and splits

- Species manifest path **and content hash**:
- For every atlas and protein embedding: actual file path, byte size, SHA256:
- Ordered source observation identities and gene identities, per species:
- Ordered exported observation identities and species codes:
- Split CSV path/hash, counts by species/donor/cell type, grouping rule:
- Held-out observation and donor identities; overlap checks:
- Were these hashes captured before the foundation was fitted? If unavailable,
  state that a current hash cannot retrospectively establish fit-time identity.

## Preprocessing and exposure

- Gene filtering, shared feature selection/HVG settings and fitted gene lists:
- Counts layer, normalization, scale/log transforms; what was fitted on which split:
- Protein embedding model/version and gene matching/case rules:
- Centroid algorithm, seed, scores/cache hashes and fitting population:
- Exposure table (list the actual observations/species, not just “train”):

| Stage | Cells/species available | Labels available/used | Fitted/selected artifacts | Held-out access |
|---|---|---|---|---|
| Preprocessing / HVGs | | | | |
| Centroids | | | | |
| Shared pretraining | | | | |
| Labelled baseline / evaluation triplet construction | | | | |
| Frozen teacher graph / bandwidth | | | | |
| Fine-tuning objective and local/preservation losses | | | | |
| Epoch selection | | | | |
| Final evaluation and visualization | | | | |

- Holdout claim: evaluation subset / fine-tuning-only exclusion / strict inductive:
- Unsupported inference/transform steps or remaining implementation gaps:

## Scientific configuration

- Macrogene / hidden / embedding dimensions, dropout, precision:
- Full-pipeline seed, pretraining seed, fine-tuning seed, batching/shuffling:
- Epochs per stage, batch size, optimizer/lr/epsilon:
- Objective, temperature, local graph weight and temperature, positive mining:
- Preservation neighbors/negatives/temperature, gradient ratio and adaptive rule:
- MMD bandwidth and seed/sample policy, or OT epsilon/mass/iterations:
- Epoch-selection rule and whether labels were excluded:

## Code, environment and execution

- Git parent, **working-source manifest/hash**, archive SHA256:
- Python/JAX/jaxlib/CUDA/Flax/Optax/Orbax/sklearn/Scanpy/UMAP versions:
- `pip freeze`/environment record and device/driver identity:
- Exact commands and environment variables, including JAX/XLA settings:
- Backend, numerical policy, query/reference blocks, cache keys/checksums:
- VM type, start timestamp, six-hour deadline and termination action:
- Graph/compilation/update/evaluation/save timings; useful epoch timings:
- Runtime estimate, reserves, safety factor, gate decision, interruptions:

## Artifacts, recovery and results

- Foundation checkpoint, original hash inventory, pre/post hash comparison:
- Epoch checkpoint directories, latest **committed** epoch, selected epoch:
- History coverage (must be 1–30 for this experiment):
- Executed/partial notebook, PNG/PDF figures, `.npz`, evaluated `.h5ad`:
- Comparison metrics and scientific acceptance, including negative results:
- Backup prefix, checksum verification output, local download and verification:
- Boot disk retained, VM STOP confirmed, stop timestamp:
- Interpretation, limits, follow-up experiments:

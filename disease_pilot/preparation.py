"""Version-pinned downloads and sparse, scale-compatible atlas augmentation.

Raw study counts are retained separately. The legacy atlases contain normalized
expression; model inputs use an explicit common per-cell scale, not fake counts.
These functions do not fit HVGs, batch correction or models.
Donor splits are materialized before the existing SATURN preprocessor is used.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

API = "https://api.cellxgene.cziscience.com/curation/v1"
SPECIES = {"h_sapiens", "m_murinus", "m_musculus", "x_laevis", "d_rerio"}
NORMAL = "PATO:0000461"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def api_json(path):
    """Public metadata API; no account or API key required."""
    for attempt in range(3):
        try:
            request = urllib.request.Request(f"{API}/{path}", headers={"Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.load(response)
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def resolve_dataset(snapshot, preset, *, live=True):
    """Use a pinned version, never silently switch to a revised dataset."""
    expected = snapshot["datasets"][preset]
    actual = api_json("dataset_versions/" + expected["dataset_version_id"]) if live else expected.copy()
    for key in ("dataset_id", "dataset_version_id", "collection_id", "cell_count"):
        if actual[key] != expected[key]:
            raise ValueError(f"Pinned metadata changed: {key}; review a new snapshot explicitly")
    expected_asset = next(a for a in expected["assets"] if a["filetype"].upper() == "H5AD")
    actual_asset = next(a for a in actual["assets"] if a["filetype"].upper() == "H5AD")
    if actual_asset != expected_asset:
        raise ValueError("Pinned H5AD asset changed; review before downloading")
    return {**expected, **actual}


def download_dataset(metadata, directory):
    """Stream to an atomic file and retain a local content checksum on first use.

    The API supplies file size, not a content checksum. The receipt verifies
    subsequent local reuse; it is not an independent server-side digest.
    """
    import h5py

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    asset = next(a for a in metadata["assets"] if a["filetype"].upper() == "H5AD")
    path = directory / (metadata["dataset_version_id"] + ".h5ad")
    receipt_path = path.with_suffix(".download.json")
    if not path.exists():
        temporary = path.with_suffix(".part")
        print(f"Downloading {metadata['title']}: {asset['filesize'] / 1e6:.1f} MB", flush=True)
        with urllib.request.urlopen(asset["url"], timeout=120) as response, temporary.open("wb") as output:
            while block := response.read(4 * 1024**2):
                output.write(block)
        if temporary.stat().st_size != asset["filesize"]:
            raise ValueError(f"Incomplete download: {temporary}; rerun to retry")
        temporary.replace(path)
    if path.stat().st_size != asset["filesize"] or not h5py.is_hdf5(path):
        raise ValueError(f"Invalid cached H5AD: {path}")
    receipt = {"dataset_id": metadata["dataset_id"], "dataset_version_id": metadata["dataset_version_id"],
               "url": asset["url"], "bytes": path.stat().st_size, "sha256": sha256(path)}
    if receipt_path.exists() and json.loads(receipt_path.read_text()) != receipt:
        raise ValueError(f"Cached download changed: {path}")
    write_json(receipt_path, receipt)
    return path


def assert_counts(matrix):
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix).ravel()
    # Bound temporary allocations for the 1.6 GB lung H5AD on a local machine.
    for start in range(0, len(values), 1_000_000):
        block = values[start:start + 1_000_000]
        if not np.isfinite(block).all() or np.any(block < 0) or np.any(np.abs(block - np.rint(block)) > 1e-5):
            raise ValueError("Expected finite, nonnegative integer raw counts, not normalized/corrected expression")


def normalized_expression(matrix, *, target_sum=1e4):
    """A per-cell transform, no fitted statistics; never claim it restores counts."""
    result = sparse.csr_matrix(matrix, dtype=np.float32).copy()
    if not np.isfinite(result.data).all() or np.any(result.data < 0):
        raise ValueError("Expression must be finite and nonnegative")
    totals = np.asarray(result.sum(axis=1)).ravel()
    if np.any(totals <= 0):
        raise ValueError("Zero-library observations after measured-feature restriction")
    result = sparse.diags(target_sum / totals) @ result
    return result.tocsr().astype(np.float32)


def count_source(atlas, source="auto"):
    """Return matrix and its OWN feature table (raw.var can differ from var)."""
    if source == "auto":
        source = "raw" if atlas.raw is not None else "counts" if "counts" in atlas.layers else "X"
    if source == "raw":
        if atlas.raw is None:
            raise ValueError("Requested raw.X is absent")
        return atlas.raw.X, atlas.raw.var.copy(), "raw.X"
    if source == "X":
        return atlas.X, atlas.var.copy(), "X"
    if source not in atlas.layers:
        raise ValueError(f"Missing count layer {source}")
    return atlas.layers[source], atlas.var.copy(), f"layers/{source}"


def map_counts(matrix, var, reference_genes):
    """Map feature_name to existing atlas symbols, summing duplicate symbols.

    Return only measured shared genes. Missing features are never zero-filled.
    Case folding follows SATURN's protein lookup; ambiguous atlas aliases fail.
    """
    reference = pd.Index(reference_genes).astype(str)
    folded = reference.str.lower()
    if not folded.is_unique:
        raise ValueError("Reference atlas has ambiguous case-folded gene symbols")
    symbols = var["feature_name"].astype(str) if "feature_name" in var else pd.Series(var.index.astype(str), index=var.index)
    lookup = dict(zip(folded, reference))
    mapped = symbols.str.lower().map(lookup)
    keep = mapped.notna().to_numpy()
    shared = reference[reference.isin(mapped.dropna())]
    if len(shared) < 2:
        raise ValueError("No useful symbol overlap; check raw.var.feature_name / identifier mapping")
    destinations = shared.get_indexer(mapped[keep])
    mapper = sparse.csr_matrix((np.ones(keep.sum(), dtype=np.float32),
                                (np.arange(keep.sum()), destinations)), shape=(keep.sum(), len(shared)))
    selected = sparse.csr_matrix(matrix)[:, keep]
    counts = (selected @ mapper).tocsr()
    assert_counts(counts)
    audit = pd.DataFrame({"feature_id": var.index.astype(str), "feature_name": symbols.to_numpy(),
                          "atlas_symbol": mapped.fillna("").to_numpy()})
    return counts, shared, audit


def read_study_counts(path, metadata, count_slot):
    """Read ONLY the chosen sparse matrix; normalized X stays on disk."""
    source = ad.read_h5ad(path, backed="r")
    try:
        required = {"donor_id", "disease_ontology_term_id", "cell_type", "assay", "suspension_type", "is_primary_data"}
        if missing := required - set(source.obs):
            raise ValueError(f"Missing required study metadata: {sorted(missing)}")
        if source.n_obs != metadata["cell_count"] or not source.obs_names.is_unique:
            raise ValueError("Downloaded cell count/identities disagree with pinned metadata")
        primary = source.obs.is_primary_data
        if not primary.isin([True, False]).all():
            raise ValueError("is_primary_data must be boolean")
        if not primary.all():
            source = source[primary.astype(bool).to_numpy()]
        raw, var, slot = count_source(source, count_slot)
        if hasattr(raw, "to_memory"):
            raw = raw.to_memory()
        return sparse.csr_matrix(raw), var, source.obs.copy(), slot
    finally:
        source.file.close()


def prepare_external(path, metadata, reference_genes, *, organ, count_slot="auto", min_genes=200,
                     max_mito_fraction=.25, max_cells_per_donor_type=None, seed=0):
    """Curated primary RNA cells, raw counts, auditable QC, no fitted transform."""
    raw, var, obs, slot = read_study_counts(path, metadata, count_slot)
    assert_counts(raw)
    symbols = var["feature_name"].astype(str) if "feature_name" in var else pd.Series(var.index.astype(str), index=var.index)
    totals = np.asarray(raw.sum(axis=1)).ravel()
    genes = np.diff(raw.indptr) if not np.any(raw.data == 0) else np.asarray((raw > 0).sum(axis=1)).ravel()
    mito = symbols.str.lower().str.startswith("mt-").to_numpy()
    mito_fraction = np.asarray(raw[:, mito].sum(axis=1)).ravel() / np.maximum(totals, 1)
    for name in ["donor_id", "disease_ontology_term_id", "cell_type", "assay", "suspension_type"]:
        if obs[name].isna().any() or obs[name].astype(str).isin(["", "unknown", "na"]).any():
            raise ValueError(f"Unusable {name}; curate sample metadata before proceeding")
    disease = obs.disease_ontology_term_id.astype(str)
    allowed = {d["ontology_term_id"] for d in metadata["disease"]}
    if not set(disease).issubset(allowed) or NORMAL not in set(disease) or len(set(disease)) < 2:
        raise ValueError("Study must contain per-cell normal AND disease annotations")
    obs["source_obs_id"] = obs.index.astype(str)
    obs["source_dataset_id"] = metadata["dataset_id"]
    obs["source_dataset_version_id"] = metadata["dataset_version_id"]
    obs["study_id"] = metadata["collection_id"]
    obs["source"] = "external_study"
    obs["species"] = "m_musculus"
    obs["condition"] = np.where(disease == NORMAL, "healthy", "disease")
    obs["donor_group"] = metadata["collection_id"] + ":" + obs.donor_id.astype(str)
    donors_before_qc = set(obs.donor_group)
    obs["organ"] = organ
    # Keep ontology/published labels separate; aliases only harmonize coarse evaluation names.
    aliases = {"endothelial cell": "capillary", "Schwann cell": "schwann", "T cell": "T",
               "B cell": "B", "natural killer cell": "NK", "mast cell": "mast"}
    obs["cellType"] = obs.cell_type.astype(str).map(lambda value: aliases.get(value, value))
    technical_column = next((c for c in ("batch", "sample_id") if c in obs), None)
    obs["technical_batch"] = obs[technical_column].astype(str) if technical_column else "unknown"
    obs["batch_id"] = (obs.study_id.astype(str) + ":" + obs.technical_batch.astype(str)
                       + ":" + obs.assay.astype(str) + ":" + obs.suspension_type.astype(str))
    obs["qc_total_counts"] = totals
    obs["qc_n_genes"] = genes
    obs["qc_mito_fraction"] = mito_fraction
    keep = (totals > 0) & (genes >= min_genes) & (mito_fraction <= max_mito_fraction)
    obs["qc_pass"] = keep
    qc = obs[["source_obs_id", "donor_group", "condition", "cellType", "qc_total_counts", "qc_n_genes", "qc_mito_fraction", "qc_pass"]].copy()
    selected = np.flatnonzero(keep)
    if max_cells_per_donor_type is not None:
        if max_cells_per_donor_type < 1:
            raise ValueError("Cell cap must be positive")
        # Optional computational cap; not condition balancing and never donor splitting.
        rng = np.random.default_rng(seed)
        frame = obs.iloc[selected].copy()
        frame["position"] = selected
        selected = np.sort(np.concatenate([
            rng.choice(g.position.to_numpy(), min(len(g), max_cells_per_donor_type), replace=False)
            for _, g in frame.groupby(["donor_group", "cellType"], observed=True, sort=True)
        ]))
    counts, shared, mapping = map_counts(raw[selected], var, reference_genes)
    obs = obs.iloc[selected].copy()
    obs.index = "cellxgene:" + metadata["dataset_id"] + ":" + obs.source_obs_id.astype(str)
    obs.index.name = "obs_id"
    result = ad.AnnData(counts.astype(np.float32), obs=obs, var=pd.DataFrame(index=shared))
    # Refuse to lose a condition/donor during QC without an explicit review.
    if set(result.obs.donor_group) != donors_before_qc:
        raise ValueError("QC removed an entire donor; review thresholds before creating a split")
    result.uns["disease_preparation"] = {"counts_from": slot, "min_genes": min_genes,
        "max_mito_fraction": max_mito_fraction, "mitochondrial_features_found": int(mito.sum()),
        "measured_shared_genes": len(shared), "atlas_genes": len(reference_genes),
        "no_normalization_or_batch_correction": True}
    return result, qc, mapping


def donor_folds(obs, *, seed=0, max_folds=5):
    """Stratified outer folds hold out whole donors, preserving all their cells.

    With two donors per condition this produces two complementary pilot folds.
    Hyperparameters are fixed in advance; no inner validation set is fabricated.
    A donor must have one condition. Donor IDs must already group all replicates.
    """
    groups = obs[["donor_group", "condition"]].drop_duplicates()
    if groups.donor_group.duplicated().any():
        raise ValueError("A donor spans conditions; curate longitudinal/group-aware folds explicitly")
    if set(groups.condition) != {"healthy", "disease"}:
        raise ValueError("Both study conditions are required")
    by_condition = {}
    for index, condition in enumerate(["healthy", "disease"]):
        donors = np.sort(groups.loc[groups.condition == condition, "donor_group"].to_numpy())
        if len(donors) < 2:
            raise ValueError("At least two independent donor groups per condition are required")
        rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
        by_condition[condition] = donors[rng.permutation(len(donors))]
    n_folds = min(max_folds, min(map(len, by_condition.values())))
    if n_folds < 2:
        raise ValueError("At least two folds are required")
    parts = {condition: np.array_split(donors, n_folds) for condition, donors in by_condition.items()}
    folds = []
    for fold in range(n_folds):
        test = {donor for chunks in parts.values() for donor in chunks[fold]}
        frame = obs[["donor_group", "condition"]].copy()
        frame.insert(0, "obs_id", obs.index.astype(str))
        frame["fold"] = fold
        frame["split"] = np.where(frame.donor_group.isin(test), "test", "train")
        folds.append(frame)
    return pd.concat(folds, ignore_index=True)


def combine_mouse(reference_path, external):
    """One matrix per biological species; disease is NOT a sixth species."""
    reference = ad.read_h5ad(reference_path)
    reference = reference[:, external.var_names].copy()
    reference.X = normalized_expression(reference.X)
    reference.layers.clear()
    reference.obsm.clear()
    reference.uns.clear()
    obs = reference.obs
    obs["source_obs_id"] = obs.index.astype(str)
    obs["source"] = "atlas"
    obs["study_id"] = "original_atlas:m_musculus"
    obs["source_dataset_id"] = "original_atlas:m_musculus"
    obs["source_dataset_version_id"] = "local_input_sha256"
    obs["species"] = "m_musculus"
    obs["condition"] = "atlas_reference"
    # Original inputs have no donor/assay metadata. Do not invent independent donors.
    obs["donor_id"] = "unknown"
    obs["donor_group"] = "original_atlas:m_musculus:unknown"
    obs["assay"] = "unknown"
    obs["suspension_type"] = "unknown"
    obs["batch_id"] = "original_atlas:m_musculus:unknown"
    obs["technical_batch"] = "unknown"
    obs["cell_type"] = obs.cellType.astype(str)
    reference.obs_names = "atlas:" + reference.obs_names.astype(str)
    study_expression = external.copy()
    study_expression.X = normalized_expression(external.X)
    study_expression.uns.clear()
    # Feature sets already match exactly. Outer preserves study-only observation
    # covariates (inner concat would silently drop age, disease ontology, etc.).
    combined = ad.concat([reference, study_expression], join="outer", merge="same")
    if not combined.obs_names.is_unique or combined.n_obs != reference.n_obs + external.n_obs:
        raise ValueError("Observation collision during augmentation")
    if list(combined.var_names) != list(external.var_names):
        raise ValueError("Combined feature order changed")
    combined.uns["expression_scale"] = "per-cell sum 10000 over shared measured mouse genes; NOT raw counts"
    return combined


def materialize_experiment(manifest_path, external, destination, *, root, seed=0, config=None, materialize_folds=(0,)):
    """Write five-species exploration + training-only fold manifests and receipts.

    All five species receive derived copies at the same per-cell expression
    scale. Raw external counts remain separate. No fit/test cells are mixed in
    fold training manifests. This is a normalized-expression compatibility arm.
    """
    root, destination = Path(root).resolve(), Path(destination).resolve()
    rows = pd.read_csv(manifest_path)
    if len(rows) != 5 or set(rows.species) != SPECIES or rows.species.duplicated().any():
        raise ValueError("Expected the canonical five-species manifest")
    for column in ("path", "embedding_path"):
        rows[column] = rows[column].map(lambda p: str((root / p).resolve()))
        for path in rows[column]:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
    mouse_path = Path(rows.loc[rows.species == "m_musculus", "path"].iloc[0])
    destination.mkdir(parents=True, exist_ok=True)
    inputs = []
    for row in rows.itertuples():
        inputs.append({"species": row.species, "atlas_path": row.path, "atlas_sha256": sha256(row.path),
                       "embedding_path": row.embedding_path, "embedding_sha256": sha256(row.embedding_path)})
    # Commit to the exact prepared population/counts and settings before overwriting outputs.
    x = sparse.csr_matrix(external.X)
    fingerprint = hashlib.sha256()
    for value in (x.data, x.indices, x.indptr):
        fingerprint.update(value.tobytes())
    fingerprint.update(external.obs.to_csv().encode())
    fingerprint.update(json.dumps(list(external.var_names)).encode())
    identity = {"schema_version": 2, "manifest_sha256": sha256(manifest_path), "seed": seed,
                "preparation_source_sha256": sha256(__file__),
                "expression_policy": "all species per-cell sum 10000; raw study counts retained separately",
                "materialize_folds": list(materialize_folds),
                "inputs": inputs, "external_fingerprint": fingerprint.hexdigest(), "config": config or {}}
    record_path = destination / "preparation_record.json"
    if record_path.exists() and json.loads(record_path.read_text())["identity"] != identity:
        raise ValueError("Inputs/configuration changed; choose a new destination")
    combined = combine_mouse(mouse_path, external)
    # Original reference files are never overwritten. Normalize one species at
    # a time to bound local working memory. Only mouse features are intersected.
    reference_dir = destination / "reference_atlases"
    reference_dir.mkdir(exist_ok=True)
    for index, row in rows.iterrows():
        reference = ad.read_h5ad(row.path)
        if row.species == "m_musculus":
            reference = reference[:, external.var_names].copy()
        reference.layers.clear()
        reference.obsm.clear()
        reference.uns.clear()
        reference.X = normalized_expression(reference.X)
        reference.uns["expression_scale"] = "per-cell sum 10000 over retained genes; NOT raw counts"
        path = reference_dir / f"{row.species}.h5ad"
        reference.write_h5ad(path, compression="gzip")
        rows.loc[index, "path"] = str(path)
        del reference
    combined.write_h5ad(destination / "m_musculus_augmented.h5ad", compression="gzip")
    external.write_h5ad(destination / "external_prepared.h5ad", compression="gzip")
    def save_manifest(path, mouse):
        updated = rows.copy()
        updated.loc[updated.species == "m_musculus", "path"] = str(mouse)
        updated.to_csv(path, index=False)
    save_manifest(destination / "exploration.csv", destination / "m_musculus_augmented.h5ad")
    folds = donor_folds(external.obs, seed=seed)
    if set(materialize_folds) - set(folds.fold.unique()):
        raise ValueError("Requested fold does not exist")
    folds.to_csv(destination / "donor_folds.csv", index=False)
    external_mask = combined.obs.source.eq("external_study").to_numpy()
    for fold, frame in folds.groupby("fold", sort=True):
        if fold not in materialize_folds:
            continue
        fold_dir = destination / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True)
        train_ids = frame.loc[frame.split == "train", "obs_id"]
        test_ids = frame.loc[frame.split == "test", "obs_id"]
        train = combined[~external_mask | combined.obs_names.isin(train_ids)].copy()
        test = combined[test_ids].copy()
        if set(train.obs_names).intersection(test.obs_names):
            raise ValueError("Train/test overlap")
        train.write_h5ad(fold_dir / "m_musculus_train.h5ad", compression="gzip")
        test.write_h5ad(fold_dir / "m_musculus_test.h5ad", compression="gzip")
        save_manifest(fold_dir / "train.csv", fold_dir / "m_musculus_train.h5ad")
        frame.to_csv(fold_dir / "split.csv", index=False)
    combined.obs.to_csv(destination / "evaluation_metadata.csv.gz")
    composition = combined.obs.groupby(["species", "source", "condition", "organ", "cellType"], observed=True).size().rename("cells").reset_index()
    composition.to_csv(destination / "composition.csv", index=False)
    products = [destination / name for name in ("m_musculus_augmented.h5ad", "external_prepared.h5ad",
                "exploration.csv", "donor_folds.csv", "evaluation_metadata.csv.gz", "composition.csv")]
    products.extend(reference_dir / f"{species}.h5ad" for species in sorted(SPECIES))
    for fold in materialize_folds:
        products.extend(destination / f"fold_{fold}" / name for name in
                        ("m_musculus_train.h5ad", "m_musculus_test.h5ad", "train.csv", "split.csv"))
    outputs = {str(p.relative_to(destination)): sha256(p) for p in sorted(products)}
    write_json(record_path, {"identity": identity, "outputs_sha256": outputs,
        "claim": "donor-held-out external study; original atlases remain training references with unknown donors",
        "feature_policy": "fixed assay/reference intersection, no expression-dependent feature fitting; downstream HVGs train-only",
        "expression_scale": "normalized expression, not raw counts; inherited SATURN count likelihood is a compatibility surrogate"})
    return combined.obs.copy(), folds

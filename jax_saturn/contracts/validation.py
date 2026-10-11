"""Fail early on schema drift before a trainer or evaluator consumes artifacts."""

import json
from pathlib import Path

import numpy as np


NPZ_KEYS = {
    "label_free": {"embeddings", "macrogenes", "species", "obs_ids"},
    "final_embeddings": {"embeddings", "species", "obs_ids"},
    "evaluation_triplets": {"anchor", "positive", "negative", "obs_ids"},
}
CHECKPOINT_KEYS = {
    "schema_version", "implementation", "model_kind", "epoch", "step", "seed",
    "hyperparameters", "species_names", "input_shapes", "source_manifest_sha256",
}
SUMMARY_KEYS = {
    "objective", "label_free", "artifact_keys_seen_by_trainer", "selected_epoch",
    "selected_species_mixing_fraction", "selected_teacher_top15_recall_at_50",
    "selection_uses_labels", "epochs", "seed",
}
SUMMARY_DIAGNOSTICS = {
    "schema_version", "trainer_version", "pretrain_checkpoint_sha256",
    "label_free_artifact_sha256", "batch_size", "learning_rate",
    "initial_species_mixing_fraction", "initial_teacher_top15_recall_at_50",
    "alignment_gradient_norm", "preservation_gradient_norm", "initial_preservation_weight",
    "selection_rule", "mmd_bandwidth", "infonce_positive_cell_coverage", "configuration",
    "implementation", "orbax_checkpoint_path", "jax_devices", "global_batch_size",
}


def exact_keys(values, required, *, optional=()):
    missing = set(required) - set(values)
    extra = set(values) - set(required) - set(optional)
    if missing or extra:
        raise ValueError(f"Schema keys: missing={sorted(missing)}, unexpected={sorted(extra)}")


def array(values, key, *, ndim, dtype=None, strings=False):
    result = np.asarray(values[key])
    if result.ndim != ndim:
        raise ValueError(f"{key} must have {ndim} dimensions")
    if strings:
        if result.dtype.kind != "U" or np.any(result == ""):
            raise ValueError(f"{key} must contain nonempty Unicode strings")
    else:
        if dtype is not None and result.dtype != np.dtype(dtype):
            raise ValueError(f"{key} must have dtype {np.dtype(dtype)}")
        if result.dtype.kind not in "fiu" or not np.isfinite(result).all():
            raise ValueError(f"{key} must contain finite numeric values")
    return result


def validate_npz(values, kind):
    """Validate exact keys, dtypes, dimensions, identity and triplet bounds."""
    if kind not in NPZ_KEYS:
        raise ValueError(f"Unknown NPZ contract: {kind}")
    exact_keys(values, NPZ_KEYS[kind])
    obs = array(values, "obs_ids", ndim=1, strings=True)
    n = len(obs)
    if not n or len(np.unique(obs)) != n:
        raise ValueError("Observation identifiers must be nonempty and unique")
    if kind == "evaluation_triplets":
        indices = [array(values, key, ndim=1, dtype=np.int64)
                   for key in ("anchor", "positive", "negative")]
        if len({len(item) for item in indices}) != 1:
            raise ValueError("Triplet arrays must have equal lengths")
        if any(np.any((item < 0) | (item >= n)) for item in indices):
            raise ValueError("Triplet indices must refer to obs_ids")
    else:
        species = array(values, "species", ndim=1, strings=True)
        if len(species) != n:
            raise ValueError("Species and observation row counts differ")
        for key in ("embeddings", "macrogenes"):
            if key in values:
                matrix = array(values, key, ndim=2, dtype=np.float32)
                if matrix.shape[0] != n or matrix.shape[1] == 0:
                    raise ValueError(f"{key} must have n_cells rows and a positive width")


def load_npz(path, kind):
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    validate_npz(values, kind)
    return values


def validate_anndata(adata, *, evaluated=False, expected_obs_ids=None,
                     expected_species=None):
    """Check a pretrain/final AnnData in memory, optionally against source order."""
    import pandas as pd

    n = adata.n_obs
    if not n or not adata.obs_names.is_unique or any(not str(x) for x in adata.obs_names):
        raise ValueError("AnnData observation identifiers must be nonempty and unique")
    matrices = {"X": adata.X, "macrogenes": adata.obsm.get("macrogenes")}
    if evaluated:
        matrices["X_umap"] = adata.obsm.get("X_umap")
    for key, matrix in matrices.items():
        if matrix is None or len(matrix.shape) != 2 or matrix.shape[0] != n or matrix.shape[1] == 0:
            raise ValueError(f"AnnData {key} has invalid shape")
        # Validate sparse inputs without allocating a dense copy.
        data = matrix.data if hasattr(matrix, "tocsr") else np.asarray(matrix)
        if data.dtype.kind != "f" or not np.isfinite(data).all():
            raise ValueError(f"AnnData {key} must contain finite floats")
    if evaluated and matrices["X_umap"].shape[1] != 2:
        raise ValueError("X_umap must have width 2")
    for key in ("labels", "labels2", "ref_labels", "species"):
        if key not in adata.obs or not isinstance(adata.obs[key].dtype, pd.CategoricalDtype):
            raise ValueError(f"AnnData obs[{key!r}] must be categorical")
        if adata.obs[key].isna().any():
            raise ValueError(f"AnnData obs[{key!r}] contains missing values")
    for name, actual, expected in (
        ("obs_ids", np.asarray(adata.obs_names).astype(str), expected_obs_ids),
        ("species", np.asarray(adata.obs["species"]).astype(str), expected_species),
    ):
        if expected is not None and not np.array_equal(actual, np.asarray(expected).astype(str)):
            raise ValueError(f"AnnData {name} order differs from source")


def validate_metric_history(frame, *, epochs=None):
    if not {"epoch", "metric_loss"}.issubset(frame.columns):
        raise ValueError("Metric history requires epoch and metric_loss")
    if frame.columns.has_duplicates:
        raise ValueError("Metric history contains duplicate columns")
    epoch = frame["epoch"].to_numpy()
    count = len(frame) if epochs is None else epochs
    if epoch.dtype.kind not in "iu" or not np.array_equal(epoch, np.arange(1, count + 1)):
        raise ValueError("Metric history epochs must be exactly 1..epochs")
    losses = frame["metric_loss"].to_numpy()
    if losses.dtype.kind not in "fiu" or not np.isfinite(losses).all():
        raise ValueError("Metric history losses must be finite numeric values")


def integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def sha256(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")


def validate_checkpoint_metadata(metadata):
    exact_keys(metadata, CHECKPOINT_KEYS, optional={"git_commit"})
    integer(metadata["schema_version"], "schema_version", minimum=1)
    if metadata["schema_version"] != 1 or metadata["implementation"] != "jax":
        raise ValueError("Unsupported checkpoint schema or implementation")
    if metadata["model_kind"] not in {"pretrain", "baseline_metric", "label_agnostic"}:
        raise ValueError("Unknown checkpoint model_kind")
    for name in ("epoch", "step", "seed"):
        integer(metadata[name], name)
    species = metadata["species_names"]
    if not isinstance(species, list) or not species or any(not isinstance(s, str) or not s for s in species):
        raise ValueError("species_names must contain nonempty strings")
    if species != sorted(set(species)):
        raise ValueError("species_names must be sorted and unique")
    if not isinstance(metadata["hyperparameters"], dict):
        raise ValueError("hyperparameters must be an object")
    shapes = metadata["input_shapes"]
    if not isinstance(shapes, dict) or not shapes:
        raise ValueError("input_shapes must be a nonempty object")
    for shape in shapes.values():
        if not isinstance(shape, list) or not shape:
            raise ValueError("input_shapes values must be nonempty dimension lists")
        for size in shape:
            integer(size, "input dimension", minimum=1)
    sha256(metadata["source_manifest_sha256"], "source_manifest_sha256")
    if "git_commit" in metadata and (not isinstance(metadata["git_commit"], str) or not metadata["git_commit"]):
        raise ValueError("git_commit must be a nonempty string")


def validate_run_summary(summary):
    exact_keys(summary, SUMMARY_KEYS, optional=SUMMARY_DIAGNOSTICS)
    if "implementation" in summary:
        if summary["implementation"] != "jax":
            raise ValueError("Unknown run-summary implementation")
        for key in ("orbax_checkpoint_path", "jax_devices", "global_batch_size"):
            if key not in summary:
                raise ValueError(f"JAX run summary requires {key}")
        if not isinstance(summary["orbax_checkpoint_path"], str) or not summary["orbax_checkpoint_path"]:
            raise ValueError("orbax_checkpoint_path must be a nonempty string")
        devices = summary["jax_devices"]
        if not isinstance(devices, list) or not devices or any(not isinstance(device, str) or not device for device in devices):
            raise ValueError("jax_devices must list device strings")
        integer(summary["global_batch_size"], "global_batch_size", minimum=1)
    if summary["objective"] not in {"infonce", "mmd", "ot"}:
        raise ValueError("Unknown label-free objective")
    if summary["label_free"] is not True or summary["selection_uses_labels"] is not False:
        raise ValueError("Label-free summary must declare label-free training and selection")
    keys = summary["artifact_keys_seen_by_trainer"]
    if not isinstance(keys, list) or len(keys) != len(NPZ_KEYS["label_free"]) or set(keys) != NPZ_KEYS["label_free"]:
        raise ValueError("Trainer must see exactly the strict label-free artifact keys")
    integer(summary["epochs"], "epochs", minimum=1)
    # The reference selects epoch 0 when no fine-tuned model beats the teacher.
    integer(summary["selected_epoch"], "selected_epoch")
    integer(summary["seed"], "seed")
    if summary["selected_epoch"] > summary["epochs"]:
        raise ValueError("selected_epoch exceeds epochs")
    for key in ("selected_species_mixing_fraction", "selected_teacher_top15_recall_at_50"):
        value = summary[key]
        if type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{key} must be a finite fraction")


def load_json(path, kind):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    with Path(path).open(encoding="utf-8") as handle:
        values = json.load(handle, object_pairs_hook=unique_object)
    validators = {"checkpoint": validate_checkpoint_metadata, "run_summary": validate_run_summary}
    if kind not in validators:
        raise ValueError(f"Unknown JSON contract: {kind}")
    if not isinstance(values, dict):
        raise ValueError("JSON artifact must be an object")
    validators[kind](values)
    return values

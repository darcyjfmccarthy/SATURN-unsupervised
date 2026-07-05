"""Strict data contracts for label-free fine-tuning."""

from pathlib import Path

import numpy as np


ALLOWED_ARTIFACT_KEYS = frozenset(
    {"embeddings", "macrogenes", "species", "obs_ids"}
)


def save_label_free_artifact(
    path,
    embeddings,
    macrogenes,
    species,
    obs_ids,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        macrogenes=np.asarray(macrogenes, dtype=np.float32),
        species=np.asarray(species).astype(str),
        obs_ids=np.asarray(obs_ids).astype(str),
    )


def load_label_free_artifact(path):
    artifact = np.load(path, allow_pickle=False)
    keys = frozenset(artifact.files)
    if keys != ALLOWED_ARTIFACT_KEYS:
        raise ValueError(
            f"Label-free artifact keys must be {sorted(ALLOWED_ARTIFACT_KEYS)}, "
            f"received {sorted(keys)}"
        )
    values = {key: artifact[key] for key in artifact.files}
    n_cells = len(values["embeddings"])
    if values["macrogenes"].shape[0] != n_cells:
        raise ValueError("Macrogene and embedding row counts differ")
    if len(values["species"]) != n_cells or len(values["obs_ids"]) != n_cells:
        raise ValueError("Metadata and embedding row counts differ")
    if len(np.unique(values["obs_ids"])) != n_cells:
        raise ValueError("Observation identifiers must be unique")
    return values


def assert_label_free_batch(batch):
    if not isinstance(batch, (tuple, list)) or len(batch) != 3:
        raise AssertionError(
            "Label-free batches must contain only macrogenes, species codes, "
            "and global indices"
        )


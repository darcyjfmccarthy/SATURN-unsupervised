"""Portable NPZ caches with JSON provenance; legacy readers are conversion-only."""

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from jax_saturn.contracts.validation import array, exact_keys, integer, sha256


EMBEDDING_KEYS = {"gene_symbols", "embeddings"}
EMBEDDING_METADATA = {
    "schema_version", "species", "source_path", "source_sha256", "created_by", "gene_symbol_case",
}
CENTROID_KEYS = {
    "scores", "centroids", "all_gene_names", "species_names",
    "species_gene_starts", "species_gene_ends",
}
CENTROID_METADATA = {
    "schema_version", "seed", "score_func", "hv_genes", "num_macrogenes",
    "embedding_cache_sha256", "source_manifest_sha256",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_path(path):
    return Path(path).with_suffix(".json")


def _schema(metadata, keys):
    exact_keys(metadata, keys)
    integer(metadata["schema_version"], "schema_version", minimum=1)
    if metadata["schema_version"] != 1:
        raise ValueError("Unsupported cache schema_version")


def validate_embedding_cache(values, metadata):
    exact_keys(values, EMBEDDING_KEYS)
    _schema(metadata, EMBEDDING_METADATA)
    genes = array(values, "gene_symbols", ndim=1, strings=True)
    embeddings = array(values, "embeddings", ndim=2, dtype=np.float32)
    if not len(genes) or embeddings.shape[0] != len(genes) or embeddings.shape[1] == 0:
        raise ValueError("Gene symbols and embedding shapes differ or are empty")
    if len(np.unique(genes)) != len(genes) or np.any(genes != np.char.lower(genes)):
        raise ValueError("Gene symbols must be unique lowercase lookup keys")
    if metadata["gene_symbol_case"] != "lower":
        raise ValueError("gene_symbol_case must be lower")
    for key in ("species", "source_path", "created_by"):
        if not isinstance(metadata[key], str) or not metadata[key]:
            raise ValueError(f"{key} must be a nonempty string")
    sha256(metadata["source_sha256"], "source_sha256")


def validate_centroid_cache(values, metadata):
    exact_keys(values, CENTROID_KEYS)
    _schema(metadata, CENTROID_METADATA)
    scores = array(values, "scores", ndim=2, dtype=np.float32)
    centroids = array(values, "centroids", ndim=2, dtype=np.float32)
    genes = array(values, "all_gene_names", ndim=1, strings=True)
    species = array(values, "species_names", ndim=1, strings=True)
    starts = array(values, "species_gene_starts", ndim=1, dtype=np.int64)
    ends = array(values, "species_gene_ends", ndim=1, dtype=np.int64)
    if not len(species) or list(species) != sorted(set(species)):
        raise ValueError("Centroid species_names must be sorted and unique")
    if not len(genes) or len(np.unique(genes)) != len(genes):
        raise ValueError("Centroid gene names must be nonempty and unique")
    integer(metadata["num_macrogenes"], "num_macrogenes", minimum=1)
    integer(metadata["hv_genes"], "hv_genes", minimum=1)
    integer(metadata["seed"], "seed")
    if metadata["score_func"] not in {"default", "one_hot", "smoothed"}:
        raise ValueError("Unknown centroid score_func")
    k = metadata["num_macrogenes"]
    if scores.shape != (len(genes), k) or centroids.shape[0] != k or centroids.shape[1] == 0:
        raise ValueError("Centroid scores/coordinates do not match declared dimensions")
    if np.any(scores < 0):
        raise ValueError("Centroid scores must be nonnegative")
    if len(starts) != len(species) or len(ends) != len(species):
        raise ValueError("Species offset counts differ")
    if starts[0] != 0 or ends[-1] != len(genes) or np.any(ends <= starts) or not np.array_equal(starts[1:], ends[:-1]):
        raise ValueError("Species offsets must partition all genes contiguously")
    for name, start, end in zip(species, starts, ends):
        if not all(gene.startswith(name + "_") and len(gene) > len(name) + 1 for gene in genes[start:end]):
            raise ValueError(f"Gene names do not match species block {name}")
    for key in ("embedding_cache_sha256", "source_manifest_sha256"):
        sha256(metadata[key], key)


def _validator(kind):
    if kind not in {"embedding", "centroid"}:
        raise ValueError(f"Unknown cache kind: {kind}")
    return validate_embedding_cache if kind == "embedding" else validate_centroid_cache


def save_cache(path, values, metadata, *, kind):
    """Validate before writing; sidecars contain only JSON, NPZ never uses pickle."""
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Portable caches require an .npz path")
    _validator(kind)(values, metadata)
    # Serialize before touching either output to catch non-JSON metadata early.
    encoded = json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(handle, **{key: values[key] for key in sorted(values)})
    metadata_path(path).write_text(encoded, encoding="utf-8")


def load_cache(path, *, kind):
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate cache metadata key: {key}")
            result[key] = value
        return result
    metadata = json.loads(metadata_path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    if not isinstance(metadata, dict):
        raise ValueError("Cache metadata must be an object")
    _validator(kind)(values, metadata)
    return values, metadata


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def convert_gene_embeddings(source, destination, *, species):
    """Convert a torch tensor dictionary; runtime cache readers do not import torch.

    Lowercase collisions follow the reference dict comprehension: the last
    source entry wins. Sorting the resulting keys makes conversion reproducible.
    """
    import torch

    source = Path(source).resolve()
    raw = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Gene embedding source must be a nonempty dictionary")
    normalized = {}
    for gene, embedding in raw.items():
        if not isinstance(gene, str) or not gene:
            raise ValueError("Gene embedding keys must be nonempty strings")
        vector = _numpy(embedding)
        if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
            raise ValueError(f"Invalid gene embedding vector: {gene}")
        normalized[gene.lower()] = vector
    genes = sorted(normalized)
    if len({normalized[gene].shape for gene in genes}) != 1:
        raise ValueError("Gene embedding dimensions differ")
    values = {"gene_symbols": np.asarray(genes, dtype=str),
              "embeddings": np.stack([normalized[gene] for gene in genes])}
    metadata = {
        "schema_version": 1, "species": species, "source_path": str(source),
        "source_sha256": file_sha256(source), "created_by": "jax_saturn.data.cache",
        "gene_symbol_case": "lower",
    }
    save_cache(destination, values, metadata, kind="embedding")
    return values, metadata


def import_centroid_pickle(source, destination, *, seed, hv_genes,
                           embedding_cache_sha256, source_manifest_sha256):
    """Import a trusted reference pickle in canonical species/gene order.

    Legacy pickles do not record seed, HVG limit or provenance hashes; callers
    must supply those from the run that created the pickle.
    """
    with Path(source).open("rb") as handle:
        raw = pickle.load(handle)
    required = {"scores", "centroids", "score_func", "sorted_species_names",
                "species_to_gene_idx_hv", "all_gene_names"}
    if not isinstance(raw, dict) or not required.issubset(raw):
        raise ValueError("Legacy centroid pickle is missing reference cache fields")
    species = raw["sorted_species_names"]
    genes = raw["all_gene_names"]
    offsets = raw["species_to_gene_idx_hv"]
    if not isinstance(raw["scores"], dict) or set(raw["scores"]) != set(genes):
        raise ValueError("Legacy centroid score keys do not match all_gene_names")
    if not isinstance(offsets, dict) or set(offsets) != set(species):
        raise ValueError("Legacy centroid offsets do not match species")
    values = {
        "scores": np.stack([_numpy(raw["scores"][gene]) for gene in genes]),
        "centroids": _numpy(raw["centroids"]),
        "all_gene_names": np.asarray(genes, dtype=str),
        "species_names": np.asarray(species, dtype=str),
        "species_gene_starts": np.asarray([offsets[name][0] for name in species], dtype=np.int64),
        "species_gene_ends": np.asarray([offsets[name][1] for name in species], dtype=np.int64),
    }
    metadata = {
        "schema_version": 1, "seed": seed, "score_func": raw["score_func"],
        "hv_genes": hv_genes, "num_macrogenes": values["centroids"].shape[0],
        "embedding_cache_sha256": embedding_cache_sha256,
        "source_manifest_sha256": source_manifest_sha256,
    }
    save_cache(destination, values, metadata, kind="centroid")
    return values, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    embedding = commands.add_parser("embeddings", help="Convert a .pt gene dictionary")
    centroid = commands.add_parser("centroids", help="Import a trusted centroid pickle")
    for command in (embedding, centroid):
        command.add_argument("source", type=Path)
        command.add_argument("destination", type=Path)
    embedding.add_argument("--species", required=True)
    centroid.add_argument("--seed", type=int, required=True)
    centroid.add_argument("--hv-genes", type=int, required=True)
    centroid.add_argument("--embedding-cache-sha256", required=True)
    centroid.add_argument("--source-manifest-sha256", required=True)
    options = vars(parser.parse_args())
    command = options.pop("command")
    if command == "embeddings":
        convert_gene_embeddings(**options)
    else:
        import_centroid_pickle(**options)


if __name__ == "__main__":
    main()

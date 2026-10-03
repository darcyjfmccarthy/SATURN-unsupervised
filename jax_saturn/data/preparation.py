"""Reference CPU preprocessing and canonical output metadata for JAX training."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import rankdata
from sklearn.cluster import KMeans

from .cache import convert_gene_embeddings, file_sha256, import_centroid_pickle, load_cache, save_cache
from .manifest import load_manifest


@dataclass
class PreparedData:
    species_names: tuple
    values: tuple
    gene_embeddings: np.ndarray
    gene_names: tuple
    obs: pd.DataFrame
    embedding_cache_sha256: str
    source_manifest_sha256: str

    @property
    def gene_counts(self):
        return tuple(matrix.shape[1] for matrix in self.values)


def prepare_data(manifest, *, cache_dir, hv_genes=2000, ref_label_col="cellType", base_dir=None):
    if type(hv_genes) is not int or hv_genes < 1:
        raise ValueError("hv_genes must be a positive integer")
    rows = load_manifest(manifest, base_dir=base_dir, check_paths=True)
    cache_dir = Path(cache_dir)
    values, embeddings, names, observations, cache_hashes = [], [], [], [], []
    for row in rows:
        if row.embedding_path.suffix == ".npz":
            cache_path = row.embedding_path
            cache, metadata = load_cache(cache_path, kind="embedding")
        elif row.embedding_path.suffix == ".pt":
            cache_path = cache_dir / f"{row.species}.npz"
            source_hash = file_sha256(row.embedding_path)
            if cache_path.exists():
                cache, metadata = load_cache(cache_path, kind="embedding")
                if metadata["source_sha256"] != source_hash:
                    raise ValueError(f"Stale embedding cache for {row.species}; remove it or use another cache directory")
            else:
                cache, metadata = convert_gene_embeddings(row.embedding_path, cache_path, species=row.species)
        else:
            raise ValueError("Embedding sources must be .pt or portable .npz caches")
        if metadata["species"] != row.species:
            raise ValueError(f"Embedding cache species differs from manifest: {row.species}")
        cache_hashes.append({"species": row.species, "sha256": file_sha256(cache_path)})
        atlas = ad.read_h5ad(row.path)
        if not atlas.n_obs or not atlas.obs_names.is_unique or not atlas.var_names.is_unique:
            raise ValueError(f"{row.species}: atlas must have cells and unique cell/gene identifiers")
        for column in (row.in_label_col, ref_label_col):
            if column not in atlas.obs or atlas.obs[column].isna().any():
                raise ValueError(f"{row.species}: missing or null label column {column}")
        lookup = {gene: index for index, gene in enumerate(cache["gene_symbols"])}
        selected = np.asarray([gene.lower() in lookup for gene in atlas.var_names])
        atlas = atlas[:, selected].copy()
        if not atlas.n_vars:
            raise ValueError(f"{row.species}: no genes have protein embeddings")
        raw_values = atlas.X.data if hasattr(atlas.X, "tocsr") else np.asarray(atlas.X)
        if not np.isfinite(raw_values).all() or np.any(raw_values < 0):
            raise ValueError(f"{row.species}: expression must be finite nonnegative counts")
        atlas.layers["raw"] = atlas.X.copy()
        sc.pp.normalize_total(atlas, target_sum=1e4)
        sc.pp.highly_variable_genes(atlas, flavor="seurat_v3", n_top_genes=hv_genes)
        atlas.X = atlas.layers.pop("raw")
        atlas = atlas[:, atlas.var["highly_variable"].to_numpy()].copy()
        if not atlas.n_vars:
            raise ValueError(f"{row.species}: HVG selection returned no genes")
        matrix = atlas.X.toarray() if hasattr(atlas.X, "toarray") else np.asarray(atlas.X)
        values.append(np.asarray(matrix, dtype=np.float32))
        embeddings.append(cache["embeddings"][[lookup[gene.lower()] for gene in atlas.var_names]])
        names.append(tuple(f"{row.species}_{gene}" for gene in atlas.var_names))
        labels = atlas.obs[row.in_label_col].astype(str)
        prefixed = row.species + "_" + labels
        observations.append(pd.DataFrame({
            "labels": prefixed, "labels2": prefixed.str.split("_").str[-1],
            "ref_labels": atlas.obs[ref_label_col].astype(str), "species": row.species,
        }, index=atlas.obs_names))
    obs = pd.concat(observations)
    if not obs.index.is_unique:
        raise ValueError("Observation IDs must be globally unique; preserve or repair source atlases before training")
    for column in obs:
        obs[column] = pd.Categorical(obs[column])
    if len({embedding.shape[1] for embedding in embeddings}) != 1:
        raise ValueError("Protein embedding dimensions differ across species")
    # A canonical manifest of sorted cache identities defines multi-cache provenance.
    cache_digest = hashlib.sha256(json.dumps(cache_hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return PreparedData(tuple(row.species for row in rows), tuple(values), np.concatenate(embeddings),
                        tuple(names), obs, cache_digest, file_sha256(manifest))


def prepare_centroids(data, destination, *, seed=0, hv_genes=2000, num_macrogenes=200, score_func="default", legacy_path=None):
    starts = np.cumsum((0,) + data.gene_counts[:-1], dtype=np.int64)
    ends = np.cumsum(data.gene_counts, dtype=np.int64)
    genes = np.asarray([name for block in data.gene_names for name in block], dtype=str)
    metadata = {"schema_version": 1, "seed": seed, "hv_genes": hv_genes,
                "num_macrogenes": num_macrogenes, "score_func": score_func,
                "embedding_cache_sha256": data.embedding_cache_sha256,
                "source_manifest_sha256": data.source_manifest_sha256}
    destination = Path(destination)
    if destination.exists():
        cache, actual_metadata = load_cache(destination, kind="centroid")
    elif legacy_path is not None:
        cache, actual_metadata = import_centroid_pickle(
            legacy_path, destination, seed=seed, hv_genes=hv_genes,
            embedding_cache_sha256=data.embedding_cache_sha256,
            source_manifest_sha256=data.source_manifest_sha256)
    else:
        kmeans = KMeans(n_clusters=num_macrogenes, random_state=seed).fit(data.gene_embeddings)
        ranks = rankdata(kmeans.transform(data.gene_embeddings), axis=1)
        if score_func == "default":
            scores = 2 * np.log1p(1 / ranks) ** 2
        elif score_func == "one_hot":
            scores = (ranks == 1).astype(np.float32)
        elif score_func == "smoothed":
            scores = 1 / ranks
        else:
            raise ValueError("Unknown centroid score_func")
        cache = {"scores": scores.astype(np.float32), "centroids": kmeans.cluster_centers_.astype(np.float32),
                 "all_gene_names": genes, "species_names": np.asarray(data.species_names, dtype=str),
                 "species_gene_starts": starts, "species_gene_ends": ends}
        save_cache(destination, cache, metadata, kind="centroid")
        actual_metadata = metadata
    if actual_metadata != metadata or not np.array_equal(cache["all_gene_names"], genes):
        raise ValueError("Centroid cache provenance, hyperparameters or selected gene order differs from run")
    if not np.array_equal(cache["species_gene_starts"], starts) or not np.array_equal(cache["species_gene_ends"], ends):
        raise ValueError("Centroid cache species offsets differ from run")
    return cache

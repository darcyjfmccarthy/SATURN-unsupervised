#!/usr/bin/env python3
"""Create strict training artifacts and frozen label-aware evaluation triplets."""

import argparse
import json
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import distances  # noqa: E402
import miners  # noqa: E402

from label_agnostic.artifacts import save_label_free_artifact  # noqa: E402


def build_fixed_triplets(
    embeddings,
    labels,
    species,
    seed=0,
    batch_size=512,
):
    embeddings = torch.as_tensor(
        np.asarray(embeddings), dtype=torch.float32
    )
    labels = torch.as_tensor(
        pd.factorize(np.asarray(labels).astype(str))[0],
        dtype=torch.long,
    )
    species = torch.as_tensor(
        pd.factorize(np.asarray(species).astype(str))[0],
        dtype=torch.long,
    )
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(embeddings), generator=generator)
    torch.manual_seed(seed)
    miner = miners.TripletMarginMiner(
        margin=0.2,
        distance=distances.CosineSimilarity(),
        type_of_triplets="semihard",
        miner_type="cross_species",
    )
    output = [[], [], []]
    for start in range(0, len(permutation), batch_size):
        batch_indices = permutation[start : start + batch_size]
        if len(torch.unique(species[batch_indices])) < 2:
            continue
        local_triplets = miner(
            F.normalize(embeddings[batch_indices], dim=1),
            labels[batch_indices],
            species[batch_indices],
            mnn=True,
        )
        for position in range(3):
            if len(local_triplets[position]):
                output[position].append(
                    batch_indices[local_triplets[position]]
                )
    if not output[0]:
        raise RuntimeError("SATURN's labeled miner produced no triplets")
    return tuple(
        torch.cat(values).numpy().astype(np.int64)
        for values in output
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain-adata", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--triplets", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    adata = ad.read_h5ad(args.pretrain_adata)
    required_obs = {"species", "labels"}
    missing = required_obs - set(adata.obs.columns)
    if missing:
        raise ValueError(f"Pretrain AnnData lacks columns: {sorted(missing)}")
    if "macrogenes" not in adata.obsm:
        raise ValueError("Pretrain AnnData lacks obsm['macrogenes']")

    save_label_free_artifact(
        args.artifact,
        embeddings=np.asarray(adata.X),
        macrogenes=np.asarray(adata.obsm["macrogenes"]),
        species=adata.obs["species"].astype(str).to_numpy(),
        obs_ids=adata.obs_names.astype(str).to_numpy(),
    )

    triplets = build_fixed_triplets(
        adata.X,
        adata.obs["labels"].astype(str).to_numpy(),
        adata.obs["species"].astype(str).to_numpy(),
        seed=args.seed,
        batch_size=args.batch_size,
    )
    triplet_path = Path(args.triplets)
    triplet_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        triplet_path,
        anchor=triplets[0],
        positive=triplets[1],
        negative=triplets[2],
        obs_ids=np.asarray(adata.obs_names, dtype=str),
    )

    metadata = {
        "schema_version": 1,
        "label_free_artifact_keys": [
            "embeddings",
            "macrogenes",
            "species",
            "obs_ids",
        ],
        "cell_count": int(adata.n_obs),
        "embedding_dimension": int(adata.n_vars),
        "macrogene_dimension": int(adata.obsm["macrogenes"].shape[1]),
        "species": sorted(
            adata.obs["species"].astype(str).unique().tolist()
        ),
        "evaluation_triplet_count": int(len(triplets[0])),
        "evaluation_labels_used_only_here": True,
        "seed": args.seed,
        "batch_size": args.batch_size,
    }
    metadata_path = Path(args.metadata)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

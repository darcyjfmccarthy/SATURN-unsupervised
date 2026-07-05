#!/usr/bin/env python3
"""Evaluate all trials with one deterministic, post-hoc label-aware pipeline."""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/macrogenes-numba-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/macrogenes-matplotlib")

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import umap
import umap.umap_ as umap_module

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from label_agnostic.metrics import (  # noqa: E402
    embedding_benchmark_metrics,
    fixed_triplet_margin_loss,
)


TRIALS = ("baseline", "infonce", "mmd", "ot")


def align_truth(truth, obs_ids):
    obs_ids = pd.Index(np.asarray(obs_ids).astype(str))
    if not obs_ids.is_unique:
        raise ValueError("Trial observation identifiers are not unique")
    missing = obs_ids.difference(truth.obs_names)
    if len(missing):
        raise ValueError(
            f"Trial contains {len(missing)} unknown observation identifiers"
        )
    return truth[obs_ids].copy()


def load_trial(root, trial, truth):
    trial_dir = root / trial
    if trial == "baseline":
        source_path = (
            trial_dir / "saturn_results" / "final_adata.h5ad"
        )
        source = sc.read_h5ad(source_path)
        align_truth(truth, source.obs_names)
        source = source[truth.obs_names].copy()
        aligned_truth = truth.copy()
        embeddings = np.asarray(source.X, dtype=np.float32)
        species = source.obs["species"].astype(str).to_numpy()
        if not np.array_equal(
            species,
            aligned_truth.obs["species"].astype(str).to_numpy(),
        ):
            raise ValueError("Baseline species metadata does not match truth")
        history_source = (
            trial_dir / "saturn_results" / "metric_history.csv"
        )
        history = pd.read_csv(history_source)
        native_loss = float(history.iloc[-1]["metric_loss"])
        selected_epoch = int(history.iloc[-1]["epoch"])
        label_free = False
    else:
        values = np.load(
            trial_dir / "final_embeddings.npz", allow_pickle=False
        )
        if frozenset(values.files) != frozenset(
            {"embeddings", "species", "obs_ids"}
        ):
            raise ValueError(
                f"{trial} final embedding artifact contains unexpected keys"
            )
        align_truth(truth, values["obs_ids"])
        order = pd.Index(values["obs_ids"].astype(str)).get_indexer(
            truth.obs_names
        )
        aligned_truth = truth.copy()
        embeddings = values["embeddings"][order].astype(np.float32)
        species = values["species"][order].astype(str)
        if not np.array_equal(
            species,
            aligned_truth.obs["species"].astype(str).to_numpy(),
        ):
            raise ValueError(f"{trial} species metadata does not match truth")
        history = pd.read_csv(trial_dir / "metric_history.csv")
        run_summary = json.loads(
            (trial_dir / "run_summary.json").read_text()
        )
        selected_epoch = int(run_summary["selected_epoch"])
        if selected_epoch == 0:
            native_loss = float("nan")
        else:
            native_loss = float(
                history.loc[
                    history["epoch"] == selected_epoch, "metric_loss"
                ].iloc[0]
            )
        label_free = True

    standardized_history = history.copy()
    standardized_history["trial"] = trial
    standardized_history.to_csv(
        trial_dir / "history.csv", index=False
    )
    return {
        "embeddings": embeddings,
        "species": species,
        "truth": aligned_truth,
        "native_loss": native_loss,
        "selected_epoch": selected_epoch,
        "label_free": label_free,
    }


def deterministic_umap(embeddings, seed):
    original_check_array = umap_module.check_array

    def compatible_check_array(*values, **kwargs):
        if "ensure_all_finite" in kwargs:
            kwargs["force_all_finite"] = kwargs.pop(
                "ensure_all_finite"
            )
        return original_check_array(*values, **kwargs)

    umap_module.check_array = compatible_check_array
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.3,
        n_components=2,
        metric="cosine",
        random_state=seed,
        transform_seed=seed,
        n_jobs=1,
    )
    return reducer.fit_transform(embeddings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--truth-adata", required=True)
    parser.add_argument("--triplets", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.root)
    truth = sc.read_h5ad(args.truth_adata)
    if "labels2" not in truth.obs or "species" not in truth.obs:
        raise ValueError("Truth AnnData must contain labels2 and species")

    try:
        triplet_data = np.load(args.triplets, allow_pickle=False)
        triplet_obs_ids = triplet_data["obs_ids"].astype(str)
    except ValueError:
        # Compatibility for manifests produced before obs_ids were forced to
        # a NumPy unicode dtype. These files are generated locally by the
        # benchmark and contain only arrays listed below.
        triplet_data = np.load(args.triplets, allow_pickle=True)
        if frozenset(triplet_data.files) != frozenset(
            {"anchor", "positive", "negative", "obs_ids"}
        ):
            raise ValueError("Unexpected evaluation-triplet manifest keys")
        triplet_obs_ids = triplet_data["obs_ids"].astype(str)
    if not np.array_equal(
        triplet_obs_ids, truth.obs_names.astype(str).to_numpy()
    ):
        raise ValueError(
            "Evaluation triplets do not match truth observation order"
        )
    triplets = (
        triplet_data["anchor"],
        triplet_data["positive"],
        triplet_data["negative"],
    )

    rows = []
    for trial in TRIALS:
        trial_dir = root / trial
        loaded = load_trial(root, trial, truth)
        embeddings = loaded["embeddings"]
        trial_truth = loaded["truth"]
        labels = trial_truth.obs["labels2"].astype(str).to_numpy()
        metrics = embedding_benchmark_metrics(
            embeddings,
            loaded["species"],
            labels,
            k=15,
        )
        metrics.update(
            fixed_triplet_margin_loss(embeddings, triplets, margin=0.2)
        )
        coordinates = deterministic_umap(embeddings, args.seed)
        evaluated = ad.AnnData(
            X=embeddings,
            obs=trial_truth.obs[
                ["species", "labels", "labels2", "ref_labels"]
            ].copy(),
        )
        evaluated.obs_names = trial_truth.obs_names.copy()
        evaluated.obsm["X_umap"] = coordinates
        if "macrogenes" in trial_truth.obsm:
            evaluated.obsm["macrogenes"] = np.asarray(
                trial_truth.obsm["macrogenes"]
            )
        evaluated.write_h5ad(trial_dir / "evaluated_adata.h5ad")
        np.savez_compressed(
            trial_dir / "umap.npz",
            coordinates=coordinates.astype(np.float32),
            species=loaded["species"],
            labels=labels,
            obs_ids=trial_truth.obs_names.astype(str).to_numpy(),
        )
        result = {
            "trial": trial,
            "label_free": loaded["label_free"],
            "selected_epoch": loaded["selected_epoch"],
            "native_objective_loss": loaded["native_loss"],
            **metrics,
        }
        (trial_dir / "metrics.json").write_text(
            json.dumps(result, indent=2, sort_keys=True)
        )
        rows.append(result)

    summary = pd.DataFrame(rows)
    baseline = summary.loc[summary["trial"] == "baseline"].iloc[0]
    summary["passes_loss"] = (
        summary["fixed_triplet_margin_loss"]
        <= baseline["fixed_triplet_margin_loss"] + 0.02
    )
    summary["passes_label_same_neighbor"] = (
        summary["label_same_neighbor_fraction"]
        >= baseline["label_same_neighbor_fraction"] - 0.05
    )
    summary["passes_species_mixing"] = (
        summary["species_mixing_fraction"]
        >= baseline["species_mixing_fraction"] - 0.03
    )
    summary["passes_all"] = (
        summary["passes_loss"]
        & summary["passes_label_same_neighbor"]
        & summary["passes_species_mixing"]
    )
    summary.to_csv(root / "comparison.csv", index=False)

    label_free_rows = summary[summary["label_free"]]
    acceptance = {
        "baseline": {
            "fixed_triplet_margin_loss": float(
                baseline["fixed_triplet_margin_loss"]
            ),
            "label_same_neighbor_fraction": float(
                baseline["label_same_neighbor_fraction"]
            ),
            "species_mixing_fraction": float(
                baseline["species_mixing_fraction"]
            ),
        },
        "thresholds": {
            "fixed_triplet_margin_loss_max": float(
                baseline["fixed_triplet_margin_loss"] + 0.02
            ),
            "label_same_neighbor_fraction_min": float(
                baseline["label_same_neighbor_fraction"] - 0.05
            ),
            "species_mixing_fraction_min": float(
                baseline["species_mixing_fraction"] - 0.03
            ),
        },
        "passing_label_free_trials": label_free_rows.loc[
            label_free_rows["passes_all"], "trial"
        ].tolist(),
        "success": bool(label_free_rows["passes_all"].any()),
    }
    (root / "acceptance.json").write_text(
        json.dumps(acceptance, indent=2, sort_keys=True)
    )
    print(summary.to_string(index=False))
    print(json.dumps(acceptance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

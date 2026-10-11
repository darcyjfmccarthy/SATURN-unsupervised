#!/usr/bin/env python3
"""Evaluate JAX with reference truth/triplets and audit HMM migration parity."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    args = parser.parse_args()
    reference, candidate = args.reference_root.resolve(), args.candidate_root.resolve()
    if reference == candidate:
        parser.error("Reference and candidate outputs must be separate")

    import anndata as ad
    import numpy as np
    import pandas as pd
    from jax_saturn.contracts.validation import load_json, load_npz, validate_anndata, validate_metric_history
    from jax_saturn.data.cache import file_sha256
    from jax_saturn.verification.parity import compare_metrics, METRICS, TRIALS

    results = candidate / "baseline/saturn_results"
    reference_config = json.loads((reference / "baseline/saturn_results/config.json").read_text())
    truth_path = reference / "baseline/saturn_results/adata_pretrain.h5ad"
    triplets_path = reference / "shared/evaluation_triplets.npz"
    truth, pretrain = ad.read_h5ad(truth_path), ad.read_h5ad(results / "adata_pretrain.h5ad")
    validate_anndata(pretrain, expected_obs_ids=truth.obs_names, expected_species=truth.obs["species"])
    for column in ("labels", "labels2", "ref_labels"):
        if not np.array_equal(pretrain.obs[column].astype(str), truth.obs[column].astype(str)):
            raise ValueError(f"JAX {column} differs from reference evaluation truth")
    triplets = load_npz(triplets_path, "evaluation_triplets")
    if not np.array_equal(triplets["obs_ids"], truth.obs_names):
        raise ValueError("Frozen reference triplets differ from truth order")
    if pretrain.shape[1] != reference_config["model_dim"] or pretrain.obsm["macrogenes"].shape[1] != reference_config["num_macrogenes"]:
        raise ValueError("JAX output dimensions differ from the defended reference")

    checkpoint_metadata = {}
    for stage, expected_kind, epochs_key, batch_key, lr_key in (
        ("pretrain", "pretrain", "pretrain_epochs", "pretrain_batch_size", "pretrain_lr"),
        ("metric", "baseline_metric", "epochs", "batch_size", "metric_lr"),
    ):
        summary = json.loads((results / f"{stage}_summary.json").read_text())
        metadata = load_json(Path(summary["orbax_checkpoint_path"]) / "metadata.json", "checkpoint")
        config = metadata["hyperparameters"]
        if metadata["model_kind"] != expected_kind or metadata["epoch"] != reference_config[epochs_key]:
            raise ValueError(f"JAX {stage} checkpoint kind/epochs differs from reference")
        for actual_key, expected_key in (("batch_size", batch_key), ("learning_rate", lr_key),
                                        ("hidden_dim", "hidden_dim"), ("embed_dim", "model_dim"), ("seed", "seed")):
            if config[actual_key] != reference_config[expected_key]:
                raise ValueError(f"JAX {stage} {actual_key} differs from reference")
        checkpoint_metadata[stage] = metadata
    pretrain_config = checkpoint_metadata["pretrain"]["hyperparameters"]
    manifest = Path(reference_config["in_data"])
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    if checkpoint_metadata["pretrain"]["source_manifest_sha256"] != file_sha256(manifest):
        raise ValueError("JAX source manifest differs from the defended reference")
    if checkpoint_metadata["metric"]["source_manifest_sha256"] != checkpoint_metadata["pretrain"]["source_manifest_sha256"]:
        raise ValueError("JAX metric/pretrain source manifests differ")
    for key in ("l1_penalty", "pe_sim_penalty", "num_macrogenes"):
        if pretrain_config[key] != reference_config[key]:
            raise ValueError(f"JAX pretrain {key} differs from reference")
    for path, epochs in ((results / "pretrain_losses.csv", reference_config["pretrain_epochs"]),
                         (results / "metric_history.csv", reference_config["epochs"])):
        frame = pd.read_csv(path)
        if frame["epoch"].tolist() != list(range(1, epochs + 1)):
            raise ValueError(f"Incomplete training history: {path}")
    for trial in TRIALS[1:]:
        reference_summary = load_json(reference / trial / "run_summary.json", "run_summary")
        summary = load_json(candidate / trial / "run_summary.json", "run_summary")
        if summary["implementation"] != "jax" or summary["objective"] != trial:
            raise ValueError(f"{trial} is not a JAX trial")
        for key in ("epochs", "seed", "batch_size", "learning_rate"):
            if summary[key] != reference_summary[key]:
                raise ValueError(f"{trial} {key} differs from reference")
        for key, value in reference_summary["configuration"].items():
            if key not in {"artifact", "pretrain_checkpoint", "output_dir", "device", "device_num"}:
                if key not in summary["configuration"] or summary["configuration"][key] != value:
                    raise ValueError(f"{trial} configuration {key} differs from reference")
        validate_metric_history(pd.read_csv(candidate / trial / "metric_history.csv"), epochs=summary["epochs"])
        arrays = load_npz(candidate / trial / "final_embeddings.npz", "final_embeddings")
        if not np.array_equal(arrays["obs_ids"], truth.obs_names):
            raise ValueError(f"{trial} observation order differs from reference")

    # Both metric comparisons use the reference's frozen evaluation triples.
    subprocess.run([sys.executable, str(ROOT / "scripts/evaluate_label_agnostic_benchmark.py"),
                    "--root", str(candidate), "--truth-adata", str(truth_path),
                    "--triplets", str(triplets_path), "--seed", str(reference_config["seed"])], check=True, cwd=ROOT)
    table = compare_metrics(pd.read_csv(reference / "comparison.csv"), pd.read_csv(candidate / "comparison.csv"))
    table.to_csv(candidate / "reference_parity.csv", index=False)
    acceptance = json.loads((candidate / "acceptance.json").read_text())
    report = {"schema_version": 1, "reference_root": str(reference), "candidate_root": str(candidate),
              "reference_comparison_sha256": file_sha256(reference / "comparison.csv"),
              "candidate_comparison_sha256": file_sha256(candidate / "comparison.csv"),
              "truth_sha256": file_sha256(truth_path), "evaluation_triplets_sha256": file_sha256(triplets_path),
              "triplet_count": len(triplets["anchor"]), "cell_count": truth.n_obs,
              "metric_degradation_tolerances": {key: value[0] for key, value in METRICS.items()},
              "reference_agreement": bool(table["passes"].all()),
              "benchmark_acceptance": bool(acceptance["success"])}
    report["success"] = report["reference_agreement"] and report["benchmark_acceptance"]
    (candidate / "reference_parity.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(table.to_string(index=False))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

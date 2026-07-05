"""Training loop for the logged phase-2 label-free study."""

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from label_free_utils import (
    embed_label_free_dataset,
    frozen_neighbor_attraction_loss,
    multi_species_mmd,
)
from phase2_utils import (
    PHASE2_OBJECTIVES,
    curriculum_anchor_loss,
    curriculum_fraction,
    embedding_shape_metrics,
    fused_gw_loss_with_diagnostics,
    gradient_diagnostics,
    graph_preservation_metrics,
    multiscale_preservation_loss,
    partial_ot_loss_with_diagnostics,
    update_preservation_multiplier,
)


EXPERIMENTS = {
    "bridge_frozen": {
        "parent": "frozen_neighbors",
        "changed_factors": ["preservation subsystem"],
        "causal_contrast": True,
    },
    "bridge_mmd": {
        "parent": "mmd",
        "changed_factors": ["preservation subsystem"],
        "causal_contrast": True,
    },
    "low_mass_ot": {
        "parent": "partial_ot",
        "changed_factors": ["preservation subsystem", "transported mass 0.8 to 0.3"],
        "causal_contrast": False,
    },
    "fused_gw": {
        "parent": None,
        "changed_factors": ["new fused Gromov-Wasserstein alignment"],
        "causal_contrast": False,
    },
    "anchor_curriculum": {
        "parent": "frozen_neighbors",
        "changed_factors": ["cycle-consistent anchors", "confidence curriculum", "preservation subsystem"],
        "causal_contrast": False,
    },
}

TRANSPORT_COLUMNS = [
    "run", "epoch", "batch", "species_a", "species_b", "requested_mass",
    "realized_mass", "row_marginal_violation", "column_marginal_violation",
    "plan_entropy", "transport_cost_mean", "transport_cost_p05",
    "transport_cost_p50", "transport_cost_p95", "structural_cost_mean",
    "active_edges", "coverage",
]
ANCHOR_COLUMNS = [
    "run", "epoch", "batch", "record_type", "species_a", "species_b", "coverage",
    "active_edges", "mean_confidence", "curriculum_fraction",
    "confidence_threshold", "confidence_p05", "confidence_p50",
    "confidence_p95", "cycle_support_mean", "candidate_edges",
    "rejected_candidates", "reciprocal_rate", "cycle_support_rate",
]


def _git_value(args):
    try:
        return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unavailable"


def _package_versions():
    versions = {}
    for name in ("torch", "numpy", "pandas", "scanpy", "scikit-learn", "umap-learn"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _sha256(path):
    if path is None or not Path(path).exists():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_event(handle, event_type, **values):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        **values,
    }
    handle.write(json.dumps(record, sort_keys=True) + "\n")
    handle.flush()


def _batch_from_global_indices(model, dataset, indices, device, species_to_code):
    embedding_parts, species_parts, index_parts = [], [], []
    for species_name in dataset.species:
        offset = dataset.offsets[species_name]
        upper = offset + dataset.num_cells[species_name]
        selected = indices[(indices >= offset) & (indices < upper)]
        if len(selected) == 0:
            continue
        local = selected - offset
        values = dataset.xs[species_name][local].to(device)
        embedding_parts.append(F.normalize(model(values, species_name), dim=1))
        species_parts.append(
            torch.full(
                (len(values),), species_to_code[species_name], dtype=torch.long, device=device
            )
        )
        index_parts.append(selected.to(device))
    return torch.cat(embedding_parts), torch.cat(species_parts), torch.cat(index_parts)


def _alignment_loss(
    objective,
    args,
    embeddings,
    species_codes,
    global_indices,
    embedding_bank,
    phase1,
    phase2,
    epoch,
):
    diagnostics = {"coverage": len(embeddings), "mean_confidence": np.nan}
    transport = []
    if objective == "bridge_frozen":
        loss, coverage = frozen_neighbor_attraction_loss(
            embeddings,
            global_indices,
            embedding_bank,
            torch.from_numpy(phase1["frozen_positives"]).long(),
            torch.from_numpy(phase1["frozen_positive_weights"]).float(),
        )
        diagnostics["coverage"] = coverage
    elif objective == "bridge_mmd":
        loss = multi_species_mmd(
            embeddings, species_codes, float(phase1["mmd_bandwidth"])
        )
    elif objective == "low_mass_ot":
        loss, transport = partial_ot_loss_with_diagnostics(
            embeddings,
            torch.from_numpy(phase1["embeddings"]).float().to(embeddings.device)[global_indices],
            torch.from_numpy(phase1["macrogenes"]).float().to(embeddings.device)[global_indices],
            species_codes,
            epsilon=args.ot_epsilon,
            transported_mass=0.3,
            iterations=args.ot_iterations,
        )
    elif objective == "fused_gw":
        loss, transport = fused_gw_loss_with_diagnostics(
            embeddings,
            torch.from_numpy(phase1["embeddings"]).float().to(embeddings.device)[global_indices],
            torch.from_numpy(phase1["macrogenes"]).float().to(embeddings.device)[global_indices],
            species_codes,
            alpha=0.5,
            epsilon=args.ot_epsilon,
            transported_mass=0.3,
            outer_iterations=10,
            sinkhorn_iterations=args.ot_iterations,
        )
    elif objective == "anchor_curriculum":
        stage = min((epoch - 1) // 5, 3)
        thresholds = phase2["curriculum_thresholds"]
        loss, anchor_diagnostics = curriculum_anchor_loss(
            embeddings,
            global_indices,
            embedding_bank,
            torch.from_numpy(phase2["cycle_positive_indices"]).long(),
            torch.from_numpy(phase2["cycle_confidence"]).float(),
            float(thresholds[stage]),
        )
        diagnostics.update(anchor_diagnostics)
        diagnostics["curriculum_fraction"] = curriculum_fraction(epoch)
        diagnostics["confidence_threshold"] = float(thresholds[stage])
    else:
        raise ValueError(f"Unknown phase-2 objective: {objective}")
    return loss, diagnostics, transport


def _numeric_summary(frame):
    output = {}
    for column in frame.select_dtypes(include=[np.number]).columns:
        values = frame[column].dropna()
        if len(values) == 0:
            continue
        output[f"{column}_mean"] = float(values.mean())
        output[f"{column}_std"] = float(values.std(ddof=0))
        output[f"{column}_p05"] = float(values.quantile(0.05))
        output[f"{column}_p50"] = float(values.quantile(0.50))
        output[f"{column}_p95"] = float(values.quantile(0.95))
    return output


def _anchor_pair_rows(
    objective,
    epoch,
    batch_index,
    global_indices,
    species_values,
    phase1,
    phase2,
):
    rows = global_indices.detach().cpu().numpy()
    if objective == "bridge_frozen":
        positives = phase1["frozen_positives"][rows]
        confidence = phase1["frozen_positive_weights"][rows]
        support = np.full_like(confidence, np.nan, dtype=np.float32)
        active = positives >= 0
    elif objective == "anchor_curriculum":
        positives = phase2["cycle_positive_indices"][rows]
        confidence = phase2["cycle_confidence"][rows]
        support = phase2["cycle_support"][rows]
        stage = min((epoch - 1) // 5, 3)
        active = (positives >= 0) & (confidence >= phase2["curriculum_thresholds"][stage])
    else:
        return []
    grouped = defaultdict(lambda: {"confidence": [], "support": []})
    for local_anchor, global_anchor in enumerate(rows):
        for position in range(positives.shape[1]):
            if not active[local_anchor, position]:
                continue
            positive = positives[local_anchor, position]
            pair = tuple(sorted((species_values[global_anchor], species_values[positive])))
            grouped[pair]["confidence"].append(confidence[local_anchor, position])
            grouped[pair]["support"].append(support[local_anchor, position])
    output = []
    for pair, values in grouped.items():
        confidence_values = np.asarray(values["confidence"], dtype=float)
        support_values = np.asarray(values["support"], dtype=float)
        output.append({
            "epoch": epoch,
            "batch": batch_index,
            "species_a": pair[0],
            "species_b": pair[1],
            "active_edges": len(confidence_values),
            "confidence_p05": float(np.quantile(confidence_values, 0.05)),
            "confidence_p50": float(np.quantile(confidence_values, 0.50)),
            "confidence_p95": float(np.quantile(confidence_values, 0.95)),
            "cycle_support_mean": float(np.nanmean(support_values)) if np.any(np.isfinite(support_values)) else np.nan,
        })
    return output


def run_phase2_finetuning(
    args,
    metric_model,
    label_free_dataset,
    train_loader,
    device,
    sorted_species_names,
    metric_dir,
    phase1_artifacts,
    save_adata,
):
    if args.finetune_objective not in PHASE2_OBJECTIVES:
        raise ValueError("run_phase2_finetuning called for a non-phase-2 objective")
    if args.phase2_artifacts_path is None:
        raise ValueError("--phase2_artifacts_path is required")
    phase2 = np.load(args.phase2_artifacts_path)
    objective = args.finetune_objective
    experiment = EXPERIMENTS[objective]
    species_to_code = {name: index for index, name in enumerate(sorted_species_names)}
    species_values = phase1_artifacts["species"].astype(str)
    teacher_neighbors = {
        k: phase2[f"neighbors_{k}"] for k in (5, 15, 50)
    }
    preservation_pairs = torch.from_numpy(phase2["preservation_pairs"]).long()
    teacher_pair_similarities = torch.from_numpy(
        phase2["teacher_pair_similarities"]
    ).float()
    teacher_embeddings = torch.from_numpy(phase1_artifacts["embeddings"]).float().to(device)
    diagnostic_indices = torch.from_numpy(phase2["diagnostic_indices"]).long()

    optimizer = torch.optim.Adam(metric_model.parameters(), lr=args.metric_lr)
    events_path = metric_dir / "events.jsonl"
    event_handle = open(events_path, "w")
    start_time = datetime.now(timezone.utc)
    manifest = {
        "schema_version": 1,
        "run": metric_dir.parent.name,
        "objective": objective,
        "phase1_parent": experiment["parent"],
        "changed_factors": experiment["changed_factors"],
        "causal_contrast": experiment["causal_contrast"],
        "argv": sys.argv,
        "command": " ".join(shlex.quote(value) for value in sys.argv),
        "configuration": {key: str(value) for key, value in vars(args).items()},
        "root_git_revision": _git_value(["git", "rev-parse", "HEAD"]),
        "submodule_git_revision": _git_value(["git", "-C", "fabio-saturn", "rev-parse", "HEAD"]),
        "root_git_status": _git_value(["git", "status", "--short"]),
        "submodule_git_status": _git_value(["git", "-C", "fabio-saturn", "status", "--short"]),
        "package_versions": _package_versions(),
        "torch_cuda_version": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(device) if str(device).startswith("cuda") else "cpu",
        "seed": args.seed,
        "pretrain_checkpoint_sha256": _sha256(args.pretrain_model_path),
        "phase1_teacher_artifacts_sha256": _sha256(args.teacher_artifacts_path),
        "phase2_artifacts_sha256": _sha256(args.phase2_artifacts_path),
        "evaluation_triplets_sha256": _sha256(
            Path(args.teacher_artifacts_path).with_name("eval_triplets.npz")
        ),
        "centroids_sha256": _sha256(args.centroids_init_path),
        "species_cell_counts": {
            name: int(label_free_dataset.num_cells[name]) for name in sorted_species_names
        },
        "total_cell_count": len(label_free_dataset),
        "started_at": start_time.isoformat(),
        "constraint_target": 0.70,
    }
    if args.anchor_diagnostics_path and Path(args.anchor_diagnostics_path).exists():
        manifest["anchor_artifact_diagnostics"] = json.loads(
            Path(args.anchor_diagnostics_path).read_text()
        )
        if manifest["anchor_artifact_diagnostics"].get("fallback_used"):
            _write_event(
                event_handle,
                "anchor_filter_fallback",
                **manifest["anchor_artifact_diagnostics"],
            )
    (metric_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )

    embedding_bank = embed_label_free_dataset(
        metric_model, label_free_dataset, device, batch_size=args.batch_size
    ).to(device)
    metric_model.eval()
    diagnostic_embeddings, diagnostic_species, diagnostic_global = _batch_from_global_indices(
        metric_model, label_free_dataset, diagnostic_indices, device, species_to_code
    )
    diagnostic_alignment, _, _ = _alignment_loss(
        objective,
        args,
        diagnostic_embeddings,
        diagnostic_species,
        diagnostic_global,
        embedding_bank,
        phase1_artifacts,
        phase2,
        epoch=1,
    )
    diagnostic_preservation, _, _ = multiscale_preservation_loss(
        diagnostic_embeddings,
        diagnostic_global,
        embedding_bank,
        teacher_embeddings,
        preservation_pairs,
        teacher_pair_similarities,
        temperature=0.1,
    )
    initial_gradients = gradient_diagnostics(
        diagnostic_alignment,
        diagnostic_preservation,
        list(metric_model.parameters()),
        multiplier=1.0,
    )
    multiplier = float(
        np.clip(
            0.25
            * initial_gradients["alignment_gradient_norm"]
            / max(initial_gradients["preservation_gradient_norm"], 1e-12),
            1e-4,
            50.0,
        )
    )
    _write_event(
        event_handle,
        "preservation_calibration",
        multiplier=multiplier,
        calibrated_weighted_ratio=(
            multiplier * initial_gradients["preservation_gradient_norm"]
            / max(initial_gradients["alignment_gradient_norm"], 1e-12)
        ),
        **initial_gradients,
    )

    batch_rows, epoch_rows, gradient_rows, transport_rows, anchor_rows = [], [], [], [], []
    previous_stage = None
    for epoch in range(1, args.epochs + 1):
        metric_model.train()
        epoch_batch_rows = []
        if objective == "anchor_curriculum":
            stage = min((epoch - 1) // 5, 3)
            if stage != previous_stage:
                _write_event(
                    event_handle,
                    "curriculum_expansion",
                    epoch=epoch,
                    stage=stage + 1,
                    active_fraction=curriculum_fraction(epoch),
                    confidence_threshold=float(phase2["curriculum_thresholds"][stage]),
                )
                previous_stage = stage
            for pair_diagnostics in manifest.get("anchor_artifact_diagnostics", {}).get(
                "species_pair_diagnostics", []
            ):
                anchor_rows.append({
                    "run": metric_dir.parent.name,
                    "epoch": epoch,
                    "record_type": "artifact_filter",
                    **pair_diagnostics,
                })

        for batch_index, batch_dict in enumerate(train_loader):
            optimizer.zero_grad()
            embedding_parts, species_parts, index_parts = [], [], []
            for species_name, (values, global_indices) in batch_dict.items():
                values = values.to(device)
                embedding_parts.append(F.normalize(metric_model(values, species_name), dim=1))
                species_parts.append(
                    torch.full(
                        (len(values),), species_to_code[species_name], dtype=torch.long, device=device
                    )
                )
                index_parts.append(global_indices.to(device))
            embeddings = torch.cat(embedding_parts)
            species_codes = torch.cat(species_parts)
            global_indices = torch.cat(index_parts)
            alignment_loss, alignment_diagnostics, batch_transport = _alignment_loss(
                objective,
                args,
                embeddings,
                species_codes,
                global_indices,
                embedding_bank,
                phase1_artifacts,
                phase2,
                epoch,
            )
            if batch_transport:
                for key in (
                    "requested_mass", "realized_mass", "plan_entropy",
                    "transport_cost_mean", "active_edges", "coverage",
                ):
                    alignment_diagnostics[f"transport_{key}"] = float(
                        np.mean([entry[key] for entry in batch_transport])
                    )
            preservation_loss, info_nce_loss, distortion_loss = multiscale_preservation_loss(
                embeddings,
                global_indices,
                embedding_bank,
                teacher_embeddings,
                preservation_pairs,
                teacher_pair_similarities,
                temperature=0.1,
            )
            weighted_preservation = multiplier * preservation_loss
            total_loss = alignment_loss + weighted_preservation
            if not torch.isfinite(total_loss):
                _write_event(event_handle, "non_finite_loss", epoch=epoch, batch=batch_index)
                raise FloatingPointError(f"Non-finite phase-2 loss at epoch {epoch}")
            total_loss.backward()
            optimizer.step()
            shape_metrics = embedding_shape_metrics(embeddings.detach().cpu())
            row = {
                "run": metric_dir.parent.name,
                "objective": objective,
                "epoch": epoch,
                "batch": batch_index,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "total_loss": float(total_loss.detach().cpu()),
                "alignment_loss": float(alignment_loss.detach().cpu()),
                "preservation_loss": float(preservation_loss.detach().cpu()),
                "preservation_info_nce": float(info_nce_loss.detach().cpu()),
                "preservation_distortion": float(distortion_loss.detach().cpu()),
                "preservation_multiplier": multiplier,
                "weighted_preservation_loss": float(weighted_preservation.detach().cpu()),
                **alignment_diagnostics,
                **shape_metrics,
            }
            batch_rows.append(row)
            epoch_batch_rows.append(row)
            if objective in {"bridge_frozen", "anchor_curriculum"}:
                anchor_rows.append({
                    "run": metric_dir.parent.name,
                    "epoch": epoch,
                    "batch": batch_index,
                    "record_type": "batch_aggregate",
                    "coverage": alignment_diagnostics.get("coverage"),
                    "active_edges": alignment_diagnostics.get("active_edges", np.nan),
                    "mean_confidence": alignment_diagnostics.get("mean_confidence", np.nan),
                    "curriculum_fraction": alignment_diagnostics.get("curriculum_fraction", np.nan),
                    "confidence_threshold": alignment_diagnostics.get("confidence_threshold", np.nan),
                })
                for pair_row in _anchor_pair_rows(
                    objective,
                    epoch,
                    batch_index,
                    global_indices,
                    species_values,
                    phase1_artifacts,
                    phase2,
                ):
                    anchor_rows.append({
                        "run": metric_dir.parent.name,
                        "record_type": "batch_species_pair",
                        **pair_row,
                    })
            for transport in batch_transport:
                transport_rows.append({
                    "run": metric_dir.parent.name,
                    "epoch": epoch,
                    "batch": batch_index,
                    **transport,
                })

        embedding_bank = embed_label_free_dataset(
            metric_model, label_free_dataset, device, batch_size=args.batch_size
        ).to(device)
        graph_metrics = graph_preservation_metrics(
            embedding_bank.cpu().numpy(), species_values, teacher_neighbors
        )
        shape_metrics = embedding_shape_metrics(embedding_bank.cpu())
        old_multiplier = multiplier
        multiplier, violation = update_preservation_multiplier(
            multiplier, graph_metrics["teacher_top15_recall_at_50"]
        )
        _write_event(
            event_handle,
            "constraint_observation",
            epoch=epoch,
            target=0.70,
            recall_at_50=graph_metrics["teacher_top15_recall_at_50"],
            violation=violation,
            old_multiplier=old_multiplier,
            new_multiplier=multiplier,
        )
        if violation > 0:
            _write_event(
                event_handle,
                "constraint_violation",
                epoch=epoch,
                recall_at_50=graph_metrics["teacher_top15_recall_at_50"],
                violation=violation,
            )
        if multiplier != old_multiplier:
            _write_event(
                event_handle,
                "constraint_multiplier_update",
                epoch=epoch,
                recall_at_50=graph_metrics["teacher_top15_recall_at_50"],
                violation=violation,
                old_multiplier=old_multiplier,
                new_multiplier=multiplier,
            )

        metric_model.eval()
        diagnostic_embeddings, diagnostic_species, diagnostic_global = _batch_from_global_indices(
            metric_model, label_free_dataset, diagnostic_indices, device, species_to_code
        )
        diagnostic_alignment, _, _ = _alignment_loss(
            objective,
            args,
            diagnostic_embeddings,
            diagnostic_species,
            diagnostic_global,
            embedding_bank,
            phase1_artifacts,
            phase2,
            epoch,
        )
        diagnostic_preservation, _, _ = multiscale_preservation_loss(
            diagnostic_embeddings,
            diagnostic_global,
            embedding_bank,
            teacher_embeddings,
            preservation_pairs,
            teacher_pair_similarities,
            temperature=0.1,
        )
        gradient_row = {
            "run": metric_dir.parent.name,
            "epoch": epoch,
            **gradient_diagnostics(
                diagnostic_alignment,
                diagnostic_preservation,
                list(metric_model.parameters()),
                multiplier,
            ),
        }
        gradient_rows.append(gradient_row)
        epoch_frame = pd.DataFrame(epoch_batch_rows)
        epoch_row = {
            "run": metric_dir.parent.name,
            "objective": objective,
            "epoch": epoch,
            "constraint_target": 0.70,
            "constraint_violation": violation,
            "multiplier_before_update": old_multiplier,
            "multiplier_update": multiplier - old_multiplier,
            "preservation_multiplier": multiplier,
            **graph_metrics,
            **shape_metrics,
            **gradient_row,
            **_numeric_summary(epoch_frame),
        }
        epoch_rows.append(epoch_row)
        # Rewrite the compact histories after every epoch so an interrupted run
        # retains all completed diagnostics rather than only the console log.
        pd.DataFrame(batch_rows).to_csv(
            metric_dir / "batch_history.csv.gz", index=False, compression="gzip"
        )
        pd.DataFrame(epoch_rows).to_csv(metric_dir / "epoch_history.csv", index=False)
        pd.DataFrame(gradient_rows).to_csv(
            metric_dir / "gradient_history.csv", index=False
        )
        pd.DataFrame(transport_rows).reindex(columns=TRANSPORT_COLUMNS).to_csv(
            metric_dir / "transport_history.csv", index=False
        )
        pd.DataFrame(anchor_rows).reindex(columns=ANCHOR_COLUMNS).to_csv(
            metric_dir / "anchor_history.csv", index=False
        )
        print(
            f"Epoch {epoch}: loss={epoch_frame.total_loss.mean():.6f}, "
            f"recall@50={graph_metrics['teacher_top15_recall_at_50']:.4f}, "
            f"lambda={multiplier:.4f}"
        )
        if epoch % args.polling_freq == 0:
            save_adata(metric_dir / f"adata_ep_{epoch}.h5ad")

    pd.DataFrame(batch_rows).to_csv(
        metric_dir / "batch_history.csv.gz", index=False, compression="gzip"
    )
    epoch_frame = pd.DataFrame(epoch_rows)
    epoch_frame.to_csv(metric_dir / "epoch_history.csv", index=False)
    pd.DataFrame(
        {
            "epoch": epoch_frame["epoch"],
            "metric_loss": epoch_frame["total_loss_mean"],
            "alignment_loss": epoch_frame["alignment_loss_mean"],
            "preservation_loss": epoch_frame["preservation_loss_mean"],
            "preservation_multiplier": epoch_frame["preservation_multiplier"],
            "finetune_objective": objective,
        }
    ).to_csv(metric_dir / "metric_history.csv", index=False)
    pd.DataFrame(gradient_rows).to_csv(metric_dir / "gradient_history.csv", index=False)
    pd.DataFrame(transport_rows).reindex(columns=TRANSPORT_COLUMNS).to_csv(
        metric_dir / "transport_history.csv", index=False
    )
    pd.DataFrame(anchor_rows).reindex(columns=ANCHOR_COLUMNS).to_csv(
        metric_dir / "anchor_history.csv", index=False
    )
    torch.save(metric_model.state_dict(), metric_dir / "final_model.pt")
    save_adata(metric_dir / "final_adata.h5ad")
    event_handle.close()

    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["duration_seconds"] = (
        datetime.fromisoformat(manifest["completed_at"]) - start_time
    ).total_seconds()
    manifest["final_constraint_metrics"] = {
        key: epoch_rows[-1][key]
        for key in (
            "teacher_top15_recall_at_50",
            "top5_overlap",
            "top15_overlap",
            "top50_overlap",
            "preservation_multiplier",
        )
    }
    manifest["final_model_sha256"] = _sha256(metric_dir / "final_model.pt")
    manifest["final_adata_sha256"] = _sha256(metric_dir / "final_adata.h5ad")
    (metric_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )

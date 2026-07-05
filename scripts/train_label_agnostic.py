#!/usr/bin/env python3
"""Fine-tune SATURN embeddings without constructing or loading cell labels."""

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/macrogenes-numba-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/macrogenes-matplotlib")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.saturn_model import SATURNMetricModel  # noqa: E402

from label_agnostic.artifacts import (  # noqa: E402
    assert_label_free_batch,
    load_label_free_artifact,
)
from label_agnostic.metrics import (  # noqa: E402
    species_mixing_fraction,
    topology_recall_at_50,
)
from label_agnostic.objectives import (  # noqa: E402
    build_cross_species_positives,
    build_preservation_graph,
    estimate_mmd_bandwidth,
    fused_teacher_view,
    multi_positive_infonce_loss,
    multi_species_mmd,
    partial_ot_alignment_loss,
    preservation_distillation_loss,
    within_species_graph_infonce_loss,
)


class LabelFreeDataset(Dataset):
    """The only values exposed to optimization."""

    def __init__(self, macrogenes, species_codes):
        self.macrogenes = torch.as_tensor(
            macrogenes, dtype=torch.float32
        )
        self.species_codes = torch.as_tensor(
            species_codes, dtype=torch.long
        )

    def __len__(self):
        return len(self.macrogenes)

    def __getitem__(self, index):
        return self.macrogenes[index], self.species_codes[index], index


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clone_state_dict(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def load_metric_model(
    checkpoint,
    input_dim,
    hidden_dim,
    model_dim,
    species_names,
    device,
):
    model = SATURNMetricModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        embed_dim=model_dim,
        dropout=0.1,
        species_to_gene_idx={
            species_name: (0, 0) for species_name in species_names
        },
        vae=False,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    transferable = {
        key: value
        for key, value in state.items()
        if key.startswith("encoder.")
        or key.startswith("cl_layer_norm.")
    }
    missing, unexpected = model.load_state_dict(
        transferable, strict=False
    )
    if missing or unexpected:
        raise RuntimeError(
            f"Could not initialize metric model; missing={missing}, "
            f"unexpected={unexpected}"
        )
    return model.to(device)


@torch.no_grad()
def embed_all(model, macrogenes, device, batch_size):
    model.eval()
    values = torch.as_tensor(macrogenes, dtype=torch.float32)
    output = []
    for start in range(0, len(values), batch_size):
        output.append(
            model(values[start : start + batch_size].to(device))
            .detach()
            .cpu()
        )
    return torch.cat(output)


def gradient_norm(loss, parameters, retain_graph):
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    squared = torch.zeros((), device=loss.device)
    for gradient in gradients:
        if gradient is not None:
            squared = squared + torch.sum(gradient.pow(2))
    return float(torch.sqrt(squared).detach().cpu())


def objective_loss(
    objective,
    embeddings,
    global_indices,
    species_codes,
    embedding_bank,
    teacher_embeddings,
    macrogenes,
    positive_indices,
    target_indices,
    global_to_local,
    mmd_bandwidth,
    args,
):
    if objective == "infonce":
        return multi_positive_infonce_loss(
            embeddings,
            global_indices,
            embedding_bank,
            positive_indices,
            target_indices,
            global_to_local,
            temperature=args.infonce_temperature,
        )
    if objective == "mmd":
        return (
            multi_species_mmd(
                embeddings, species_codes, mmd_bandwidth
            ),
            len(embeddings),
        )
    if objective == "ot":
        loss, transported = partial_ot_alignment_loss(
            embeddings,
            teacher_embeddings[global_indices],
            macrogenes[global_indices],
            species_codes,
            epsilon=args.ot_epsilon,
            transported_mass=args.ot_mass,
            iterations=args.ot_iterations,
        )
        return loss, float(np.mean(transported)) if transported else 0
    raise ValueError(f"Unknown objective: {objective}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--objective", required=True, choices=["infonce", "mmd", "ot"]
    )
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--pretrain-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-num", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--preservation-temperature", type=float, default=0.1)
    parser.add_argument("--local-graph-temperature", type=float, default=0.1)
    parser.add_argument("--local-positive-k", type=int, default=5)
    parser.add_argument("--local-graph-weight", type=float, default=0.3)
    parser.add_argument("--infonce-temperature", type=float, default=0.1)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--positives-per-species", type=int, default=3)
    parser.add_argument("--preservation-target", type=float, default=0.70)
    parser.add_argument(
        "--preservation-gradient-ratio", type=float, default=0.25
    )
    parser.add_argument("--ot-epsilon", type=float, default=0.05)
    parser.add_argument("--ot-mass", type=float, default=0.8)
    parser.add_argument("--ot-iterations", type=int, default=100)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.device_num)
        device = torch.device(f"cuda:{args.device_num}")
    else:
        device = torch.device("cpu")

    artifact = load_label_free_artifact(args.artifact)
    species_names, species_codes_np = np.unique(
        artifact["species"], return_inverse=True
    )
    dataset = LabelFreeDataset(
        artifact["macrogenes"], species_codes_np
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )

    model = load_metric_model(
        args.pretrain_checkpoint,
        input_dim=artifact["macrogenes"].shape[1],
        hidden_dim=args.hidden_dim,
        model_dim=args.model_dim,
        species_names=species_names,
        device=device,
    )
    teacher_embeddings = torch.as_tensor(
        artifact["embeddings"], dtype=torch.float32, device=device
    )
    macrogenes = torch.as_tensor(
        artifact["macrogenes"], dtype=torch.float32, device=device
    )
    (
        preservation_candidates_np,
        preservation_probabilities_np,
        preservation_similarities_np,
        teacher_neighbors,
    ) = build_preservation_graph(
        artifact["embeddings"],
        artifact["species"],
        neighbor_k=15,
        negative_k=15,
        temperature=args.preservation_temperature,
        seed=args.seed,
    )
    preservation_candidates = torch.as_tensor(
        preservation_candidates_np, dtype=torch.long
    )
    preservation_probabilities = torch.as_tensor(
        preservation_probabilities_np, dtype=torch.float32
    )
    preservation_similarities = torch.as_tensor(
        preservation_similarities_np, dtype=torch.float32
    )
    teacher_neighbors_tensor = torch.as_tensor(
        teacher_neighbors, dtype=torch.long
    )

    positive_indices = None
    if args.objective == "infonce":
        positive_indices_np, positive_species_names = (
            build_cross_species_positives(
                artifact["embeddings"],
                artifact["macrogenes"],
                artifact["species"],
                candidate_k=args.candidate_k,
                positives_per_species=args.positives_per_species,
            )
        )
        if not np.array_equal(positive_species_names, species_names):
            raise RuntimeError("Species coding changed during positive mining")
        positive_indices = torch.as_tensor(
            positive_indices_np, dtype=torch.long
        )
        positive_coverage = float(
            np.mean(np.any(positive_indices_np >= 0, axis=(1, 2)))
        )
        if positive_coverage == 0:
            raise RuntimeError("No reciprocal cross-species positives found")
    else:
        positive_coverage = None

    target_indices = []
    global_to_local = []
    for target_code in range(len(species_names)):
        indices = torch.as_tensor(
            np.flatnonzero(species_codes_np == target_code),
            dtype=torch.long,
        )
        mapping = torch.full((len(dataset),), -1, dtype=torch.long)
        mapping[indices] = torch.arange(len(indices))
        target_indices.append(indices)
        global_to_local.append(mapping.to(device))

    mmd_bandwidth = estimate_mmd_bandwidth(
        fused_teacher_view(
            artifact["embeddings"], artifact["macrogenes"]
        ),
        seed=args.seed,
    )

    initial_raw = embed_all(
        model, artifact["macrogenes"], device, args.batch_size
    )
    embedding_bank = F.normalize(initial_raw, dim=1).to(device)
    initial_mixing = species_mixing_fraction(
        initial_raw.numpy(), artifact["species"], k=15
    )
    initial_recall = topology_recall_at_50(
        initial_raw.numpy(),
        artifact["species"],
        teacher_neighbors,
    )
    best_state = clone_state_dict(model)
    best_epoch = 0
    best_mixing = initial_mixing
    best_recall = initial_recall

    calibration_batch = next(iter(loader))
    assert_label_free_batch(calibration_batch)
    calibration_values, calibration_species, calibration_indices = (
        calibration_batch
    )
    calibration_values = calibration_values.to(device)
    calibration_species = calibration_species.to(device)
    calibration_indices = calibration_indices.to(device)
    model.train()
    calibration_embeddings = F.normalize(
        model(calibration_values), dim=1
    )
    alignment_loss, _ = objective_loss(
        args.objective,
        calibration_embeddings,
        calibration_indices,
        calibration_species,
        embedding_bank,
        teacher_embeddings,
        macrogenes,
        positive_indices,
        target_indices,
        global_to_local,
        mmd_bandwidth,
        args,
    )
    preservation_loss = preservation_distillation_loss(
        calibration_embeddings,
        calibration_indices,
        embedding_bank,
        preservation_candidates,
        preservation_probabilities,
        preservation_similarities,
        temperature=args.preservation_temperature,
    )
    parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    alignment_gradient_norm = gradient_norm(
        alignment_loss, parameters, retain_graph=True
    )
    preservation_gradient_norm = gradient_norm(
        preservation_loss, parameters, retain_graph=False
    )
    preservation_weight = float(
        np.clip(
            args.preservation_gradient_ratio
            * alignment_gradient_norm
            / max(preservation_gradient_norm, 1e-12),
            1e-3,
            100.0,
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate
    )
    history = []
    for epoch in range(1, args.epochs + 1):
        embedding_bank = F.normalize(
            embed_all(
                model,
                artifact["macrogenes"],
                device,
                args.batch_size,
            ),
            dim=1,
        ).to(device)
        model.train()
        batch_rows = []
        for batch in loader:
            assert_label_free_batch(batch)
            values, batch_species, global_indices = batch
            values = values.to(device)
            batch_species = batch_species.to(device)
            global_indices = global_indices.to(device)
            optimizer.zero_grad()
            embeddings = F.normalize(model(values), dim=1)
            alignment_loss, coverage = objective_loss(
                args.objective,
                embeddings,
                global_indices,
                batch_species,
                embedding_bank,
                teacher_embeddings,
                macrogenes,
                positive_indices,
                target_indices,
                global_to_local,
                mmd_bandwidth,
                args,
            )
            preservation_loss = preservation_distillation_loss(
                embeddings,
                global_indices,
                embedding_bank,
                preservation_candidates,
                preservation_probabilities,
                preservation_similarities,
                temperature=args.preservation_temperature,
            )
            local_graph_loss, local_coverage = (
                within_species_graph_infonce_loss(
                    embeddings,
                    global_indices,
                    batch_species,
                    embedding_bank,
                    teacher_neighbors_tensor,
                    target_indices,
                    global_to_local,
                    positive_k=args.local_positive_k,
                    temperature=args.local_graph_temperature,
                )
            )
            total_loss = (
                alignment_loss
                + preservation_weight * preservation_loss
                + args.local_graph_weight * local_graph_loss
            )
            if not torch.isfinite(total_loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch}"
                )
            total_loss.backward()
            optimizer.step()
            batch_rows.append(
                {
                    "total_loss": float(total_loss.detach().cpu()),
                    "alignment_loss": float(
                        alignment_loss.detach().cpu()
                    ),
                    "preservation_loss": float(
                        preservation_loss.detach().cpu()
                    ),
                    "local_graph_loss": float(
                        local_graph_loss.detach().cpu()
                    ),
                    "local_coverage": float(local_coverage),
                    "coverage": float(coverage),
                }
            )

        epoch_raw = embed_all(
            model, artifact["macrogenes"], device, args.batch_size
        )
        mixing = species_mixing_fraction(
            epoch_raw.numpy(), artifact["species"], k=15
        )
        recall = topology_recall_at_50(
            epoch_raw.numpy(),
            artifact["species"],
            teacher_neighbors,
        )
        feasible = recall >= args.preservation_target
        selected = feasible and mixing > best_mixing
        if selected:
            best_state = clone_state_dict(model)
            best_epoch = epoch
            best_mixing = mixing
            best_recall = recall

        row = {
            "epoch": epoch,
            "metric_loss": float(
                np.mean([item["total_loss"] for item in batch_rows])
            ),
            "alignment_loss": float(
                np.mean(
                    [item["alignment_loss"] for item in batch_rows]
                )
            ),
            "preservation_loss": float(
                np.mean(
                    [item["preservation_loss"] for item in batch_rows]
                )
            ),
            "local_graph_loss": float(
                np.mean(
                    [item["local_graph_loss"] for item in batch_rows]
                )
            ),
            "mean_local_coverage_per_batch": float(
                np.mean(
                    [item["local_coverage"] for item in batch_rows]
                )
            ),
            "preservation_weight": preservation_weight,
            "local_graph_weight": args.local_graph_weight,
            "mean_coverage_per_batch": float(
                np.mean([item["coverage"] for item in batch_rows])
            ),
            "species_mixing_fraction": mixing,
            "teacher_top15_recall_at_50": recall,
            "checkpoint_feasible": feasible,
            "checkpoint_selected": selected,
            "objective": args.objective,
        }
        history.append(row)
        print(
            f"Epoch {epoch}: loss={row['metric_loss']:.6f}, "
            f"mixing={mixing:.4f}, recall@50={recall:.4f}, "
            f"lambda={preservation_weight:.4f}"
        )
        preservation_weight = float(
            np.clip(
                preservation_weight
                * math.exp(
                    2.0 * (args.preservation_target - recall)
                ),
                1e-3,
                100.0,
            )
        )

    model.load_state_dict(best_state)
    final_raw = embed_all(
        model, artifact["macrogenes"], device, args.batch_size
    ).numpy()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "final_model.pt")
    np.savez_compressed(
        output_dir / "final_embeddings.npz",
        embeddings=final_raw.astype(np.float32),
        species=artifact["species"].astype(str),
        obs_ids=artifact["obs_ids"].astype(str),
    )
    pd.DataFrame(history).to_csv(
        output_dir / "metric_history.csv", index=False
    )
    summary = {
        "schema_version": 1,
        "trainer_version": 4,
        "objective": args.objective,
        "label_free": True,
        "artifact_keys_seen_by_trainer": sorted(artifact),
        "pretrain_checkpoint_sha256": sha256(
            args.pretrain_checkpoint
        ),
        "label_free_artifact_sha256": sha256(args.artifact),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "initial_species_mixing_fraction": initial_mixing,
        "initial_teacher_top15_recall_at_50": initial_recall,
        "alignment_gradient_norm": alignment_gradient_norm,
        "preservation_gradient_norm": preservation_gradient_norm,
        "initial_preservation_weight": history[0][
            "preservation_weight"
        ],
        "selected_epoch": best_epoch,
        "selected_species_mixing_fraction": best_mixing,
        "selected_teacher_top15_recall_at_50": best_recall,
        "selection_uses_labels": False,
        "selection_rule": (
            "maximize species_mixing_fraction subject to "
            f"teacher_top15_recall_at_50 >= {args.preservation_target}"
        ),
        "mmd_bandwidth": mmd_bandwidth,
        "infonce_positive_cell_coverage": positive_coverage,
        "configuration": vars(args),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

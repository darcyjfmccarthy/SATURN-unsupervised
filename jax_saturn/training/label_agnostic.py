"""Strict label-free fine-tuning with frozen CPU graphs and native checkpoint resume."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

from jax_saturn.contracts.validation import load_npz, validate_metric_history, validate_npz, validate_run_summary
from jax_saturn.data.batching import metric_batches
from jax_saturn.data.cache import file_sha256
from jax_saturn.data.graphs import (
    build_cross_species_positives, build_preservation_graph, estimate_mmd_bandwidth,
    fused_teacher_view, normalize_numpy, species_mixing_fraction, topology_recall_at_50,
)
from jax_saturn.distributed.checkpoint import directory_sha256, load_pretrain_metric_params, restore_checkpoint, save_checkpoint
from jax_saturn.losses.core import normalize
from jax_saturn.losses.objectives import (
    multi_positive_infonce_loss, multi_species_mmd, partial_ot_alignment_loss,
    preservation_distillation_loss, within_species_graph_infonce_loss,
)
from jax_saturn.models.saturn import SaturnMetricModule
from jax_saturn.models.precision import matmul_dtype, resolve_precision
from .baseline import embed_metric


@dataclass(frozen=True)
class LabelConfig:
    seed: int = 0
    epochs: int = 20
    batch_size: int = 512
    learning_rate: float = .001
    hidden_dim: int = 256
    model_dim: int = 256
    preservation_temperature: float = .1
    local_graph_temperature: float = .1
    local_positive_k: int = 5
    local_graph_weight: float = .3
    infonce_temperature: float = .1
    candidate_k: int = 20
    positives_per_species: int = 3
    preservation_target: float = .70
    preservation_gradient_ratio: float = .25
    ot_epsilon: float = .05
    ot_mass: float = .8
    ot_iterations: int = 100
    mixed_precision: str = "fp32"


@struct.dataclass
class LabelAgnosticTrainState:
    params: object
    opt_state: object
    rng: jax.Array
    step: jax.Array
    epoch: jax.Array
    best_params: object
    best_epoch: int
    best_step: int
    best_mixing: float
    best_recall: float
    preservation_weight: float
    config: object = struct.field(pytree_node=False)
    diagnostics: object = struct.field(pytree_node=False)


def _validate_config(config):
    if resolve_precision(config.mixed_precision) != config.mixed_precision:
        raise ValueError("LabelConfig mixed_precision must be fp32 or bf16")
    for field in ("epochs", "batch_size", "hidden_dim", "model_dim", "local_positive_k",
                  "candidate_k", "positives_per_species", "ot_iterations"):
        if type(getattr(config, field)) is not int or getattr(config, field) < 1:
            raise ValueError(f"{field} must be a positive integer")
    for field in ("learning_rate", "preservation_temperature", "local_graph_temperature", "infonce_temperature", "ot_epsilon"):
        if not np.isfinite(getattr(config, field)) or getattr(config, field) <= 0:
            raise ValueError(f"{field} must be positive and finite")
    if (not 0 <= config.preservation_target <= 1 or not 0 < config.ot_mass <= 1
            or not np.isfinite(config.local_graph_weight) or not np.isfinite(config.preservation_gradient_ratio)
            or config.local_graph_weight < 0 or config.preservation_gradient_ratio < 0 or config.seed < 0):
        raise ValueError("Invalid label-free weights, seed or transport/selection fractions")


def build_graphs(artifact, objective, config):
    """Only the four strict artifact fields enter graph construction."""
    names, codes = np.unique(artifact["species"], return_inverse=True)
    if len(names) < 2 or len(codes) >= 2 ** 31:
        raise ValueError("Label-free benchmark requires >=2 species and fewer than 2^31 cells")
    candidates, probabilities, similarities, neighbors = build_preservation_graph(
        artifact["embeddings"], artifact["species"], temperature=config.preservation_temperature, seed=config.seed)
    positives, positive_coverage = None, None
    if objective == "infonce":
        positives, positive_names = build_cross_species_positives(
            artifact["embeddings"], artifact["macrogenes"], artifact["species"],
            candidate_k=config.candidate_k, positives_per_species=config.positives_per_species)
        if not np.array_equal(names, positive_names):
            raise ValueError("Species coding changed during positive mining")
        positive_coverage = float(np.mean(np.any(positives >= 0, axis=(1, 2))))
        if positive_coverage == 0:
            raise ValueError("No reciprocal cross-species positives found")
    targets, maps = [], []
    for code in range(len(names)):
        indices = np.flatnonzero(codes == code).astype(np.int32)
        mapping = np.full(len(codes), -1, dtype=np.int32)
        mapping[indices] = np.arange(len(indices), dtype=np.int32)
        targets.append(jnp.asarray(indices))
        maps.append(jnp.asarray(mapping))
    bandwidth = estimate_mmd_bandwidth(fused_teacher_view(artifact["embeddings"], artifact["macrogenes"]), seed=config.seed)
    arrays = {"candidates": jnp.asarray(candidates, dtype=jnp.int32), "probabilities": jnp.asarray(probabilities),
              "similarities": jnp.asarray(similarities), "neighbors": jnp.asarray(neighbors, dtype=jnp.int32),
              "positives": None if positives is None else jnp.asarray(positives, dtype=jnp.int32),
              "targets": tuple(targets), "maps": tuple(maps),
              "teacher": jnp.asarray(artifact["embeddings"]), "macrogenes": jnp.asarray(artifact["macrogenes"])}
    return names, codes.astype(np.int32), arrays, neighbors, bandwidth, positive_coverage


def label_losses_from_embeddings(output, batch, bank, graphs, objective, bandwidth, config):
    indices, codes, mask = batch["global_indices"], batch["species_codes"], batch["valid_mask"]
    if objective == "infonce":
        alignment, coverage = multi_positive_infonce_loss(output, indices, bank, graphs["positives"],
            graphs["targets"], graphs["maps"], temperature=config.infonce_temperature, valid_mask=mask)
    elif objective == "mmd":
        alignment = multi_species_mmd(output, codes, bandwidth, num_species=len(graphs["targets"]), valid_mask=mask)
        coverage = jnp.sum(mask)
    else:
        alignment, masses = partial_ot_alignment_loss(output, graphs["teacher"][indices], graphs["macrogenes"][indices], codes,
            epsilon=config.ot_epsilon, transported_mass=config.ot_mass, iterations=config.ot_iterations,
            num_species=len(graphs["targets"]), valid_mask=mask)
        coverage = jnp.sum(masses) / jnp.maximum(jnp.sum(masses > 0), 1)
    preservation = preservation_distillation_loss(output, indices, bank, graphs["candidates"], graphs["probabilities"],
        graphs["similarities"], temperature=config.preservation_temperature, valid_mask=mask)
    local, local_coverage = within_species_graph_infonce_loss(output, indices, codes, bank, graphs["neighbors"],
        graphs["targets"], graphs["maps"], positive_k=config.local_positive_k,
        temperature=config.local_graph_temperature, valid_mask=mask)
    return alignment, preservation, local, coverage, local_coverage


def make_label_losses(module, objective, graphs, bandwidth, config):
    def losses(params, batch, bank, dropout_key, *, graph_arrays=graphs):
        output = normalize(module.apply({"params": params}, batch["macrogenes"], train=True,
                                        rngs={"dropout": dropout_key}))
        return label_losses_from_embeddings(output, batch, bank, graph_arrays, objective, bandwidth, config)
    return losses


def make_label_step(module, optimizer, objective, graphs, bandwidth, config):
    losses = make_label_losses(module, objective, graphs, bandwidth, config)

    @jax.jit
    def update(params, opt_state, rng, batch, bank, weight, graph_arrays):
        next_rng, dropout_key = jax.random.split(rng)
        def total(params):
            alignment, preservation, local, coverage, local_coverage = losses(
                params, batch, bank, dropout_key, graph_arrays=graph_arrays)
            loss = alignment + weight * preservation + config.local_graph_weight * local
            return loss, {"alignment_loss": alignment, "preservation_loss": preservation,
                          "local_graph_loss": local, "coverage": coverage, "local_coverage": local_coverage}
        (loss, metrics), gradients = jax.value_and_grad(total, has_aux=True)(params)
        updates, opt_state = optimizer.update(gradients, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, next_rng, {**metrics, "metric_loss": loss,
                                                                         "gradient_norm": optax.global_norm(gradients)}

    def step(state, batch, bank):
        # Host selection fractions/weight retain Python precision across updates.
        params, opt_state, rng, metrics = update(state.params, state.opt_state, state.rng, batch, bank,
                                                 jnp.float32(state.preservation_weight), graphs)
        return state.replace(params=params, opt_state=opt_state, rng=rng, step=state.step + 1), metrics
    return step, losses


def _device_batch(batch):
    allowed = {"macrogenes", "species_codes", "global_indices", "valid_mask"}
    if set(batch) != allowed:
        raise ValueError("Label-free batches contain unexpected fields")
    return {key: jnp.asarray(value, dtype=jnp.int32 if key == "global_indices" else None) for key, value in batch.items()}


def train_label_free(artifact_path, pretrain_checkpoint, output_dir, *, objective, config=LabelConfig(), resume=None, devices=None):
    if objective not in {"infonce", "mmd", "ot"}:
        raise ValueError("Unknown label-free objective")
    _validate_config(config)
    if devices is not None:
        from jax_saturn.distributed.metric import unreplicate_state, validate_devices
        from jax_saturn.distributed.label_agnostic import make_label_pmap, replicate_training_state
        devices = validate_devices(devices, batch_size=config.batch_size)
    artifact = load_npz(artifact_path, "label_free")
    if artifact["embeddings"].shape[1] != config.model_dim:
        raise ValueError("Teacher embedding dimension differs from model_dim")
    params, source_metadata = load_pretrain_metric_params(pretrain_checkpoint,
        input_dim=artifact["macrogenes"].shape[1], hidden_dim=config.hidden_dim, model_dim=config.model_dim)
    names, codes, graphs, neighbors, bandwidth, positive_coverage = build_graphs(artifact, objective, config)
    if names.tolist() != source_metadata["species_names"]:
        raise ValueError("Artifact species differ from pretrain checkpoint")
    module = SaturnMetricModule(artifact["macrogenes"].shape[1], config.hidden_dim, config.model_dim,
                               dtype=matmul_dtype(config.mixed_precision))
    optimizer = optax.adam(config.learning_rate, eps=1e-8)
    hyperparameters = asdict(config)
    hyperparameters.pop("epochs")
    if config.mixed_precision == "fp32":
        hyperparameters.pop("mixed_precision")
    source_hash = directory_sha256(pretrain_checkpoint)
    artifact_hash = file_sha256(artifact_path)
    hyperparameters.update(objective=objective, label_free_artifact_sha256=artifact_hash,
                            pretrain_checkpoint_sha256=source_hash, checkpoint_role="epoch")
    if devices is not None:
        hyperparameters.update(execution="pmap", local_device_count=len(devices))
    state = LabelAgnosticTrainState(params, optimizer.init(params), jax.random.key(config.seed), jnp.int32(0), jnp.int32(0),
                                   params, 0, 0, 0., 0., 1., hyperparameters, {})

    def metadata(state, *, selected=False):
        return {"schema_version": 1, "implementation": "jax", "model_kind": "label_agnostic",
                "epoch": int(state.epoch), "step": int(state.step), "seed": config.seed,
                "hyperparameters": {**hyperparameters, "checkpoint_role": "selected" if selected else "epoch"},
                "species_names": names.tolist(), "input_shapes": {"macrogenes": [config.batch_size, module.input_dim]},
                "source_manifest_sha256": source_metadata["source_manifest_sha256"]}

    step, losses = make_label_step(module, optimizer, objective, graphs, bandwidth, config)
    if devices is not None:
        parallel_step, parallel_calibration = make_label_pmap(module, optimizer, objective, graphs, bandwidth, config, devices)
    history = []
    if resume is not None:
        state, _, history = restore_checkpoint(resume, state, expected_metadata=metadata(state))
        validate_metric_history(pd.DataFrame(history), epochs=int(state.epoch))
    else:
        initial = embed_metric(module, params, artifact["macrogenes"], batch_size=config.batch_size, devices=devices)
        mixing = species_mixing_fraction(initial, artifact["species"])
        recall = topology_recall_at_50(initial, artifact["species"], neighbors)
        if not np.isfinite(mixing) or not np.isfinite(recall):
            raise ValueError("Initial label-free selection metrics are undefined; species need enough cells for topology evaluation")
        bank = jnp.asarray(normalize_numpy(initial))
        calibration = _device_batch(next(metric_batches(artifact["macrogenes"], codes, config.batch_size, seed=config.seed, epoch=0)))
        next_rng, key = jax.random.split(state.rng)
        if devices is None:
            gradients = [jax.jit(jax.grad(lambda p, batch, bank, key, graphs:
                losses(p, batch, bank, key, graph_arrays=graphs)[position]))(
                    params, calibration, bank, key, graphs) for position in (0, 1)]
        else:
            gradients = parallel_calibration(replicate_training_state(state, devices), calibration, bank)
        alignment_norm, preservation_norm = [float(optax.global_norm(gradient)) for gradient in gradients]
        weight = float(np.clip(config.preservation_gradient_ratio * alignment_norm / max(preservation_norm, 1e-12), 1e-3, 100.))
        if not np.isfinite(weight) or not np.isfinite(alignment_norm) or not np.isfinite(preservation_norm):
            raise FloatingPointError("Nonfinite label-free gradient calibration")
        diagnostics = {"initial_mixing": mixing, "initial_recall": recall, "alignment_gradient_norm": alignment_norm,
                       "preservation_gradient_norm": preservation_norm, "initial_weight": weight}
        state = state.replace(rng=next_rng, best_mixing=mixing, best_recall=recall, preservation_weight=weight, diagnostics=diagnostics)
    if int(state.epoch) > config.epochs:
        raise ValueError("Label-free checkpoint exceeds requested epochs")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if devices is not None:
        replica_state = replicate_training_state(state, devices)
    for epoch in range(int(state.epoch) + 1, config.epochs + 1):
        bank = jnp.asarray(normalize_numpy(embed_metric(module, state.params, artifact["macrogenes"], batch_size=config.batch_size, devices=devices)))
        batch_metrics = []
        for batch in metric_batches(artifact["macrogenes"], codes, config.batch_size, seed=config.seed, epoch=epoch):
            if devices is None:
                state, metrics = step(state, _device_batch(batch), bank)
            else:
                replica_state, metrics = parallel_step(replica_state, _device_batch(batch), bank, state.preservation_weight)
            metrics = {key: float(value) for key, value in jax.device_get(metrics).items()}
            if not all(np.isfinite(value) for value in metrics.values()):
                raise FloatingPointError(f"Nonfinite label-free loss or gradient at epoch {epoch}")
            batch_metrics.append(metrics)
        if devices is not None:
            plain = unreplicate_state(replica_state)
            state = state.replace(params=plain.params, opt_state=plain.opt_state, rng=plain.rng, step=plain.step)
        raw = embed_metric(module, state.params, artifact["macrogenes"], batch_size=config.batch_size, devices=devices)
        mixing = species_mixing_fraction(raw, artifact["species"])
        recall = topology_recall_at_50(raw, artifact["species"], neighbors)
        if not np.isfinite(mixing) or not np.isfinite(recall):
            raise FloatingPointError("Nonfinite label-free selection metrics")
        feasible = recall >= config.preservation_target
        selected = feasible and mixing > state.best_mixing
        if selected:
            state = state.replace(best_params=state.params, best_epoch=epoch, best_step=int(state.step), best_mixing=mixing, best_recall=recall)
        row = {"epoch": epoch, **{key: float(np.mean([batch[key] for batch in batch_metrics]))
                                  for key in ("metric_loss", "alignment_loss", "preservation_loss", "local_graph_loss")},
               "mean_local_coverage_per_batch": float(np.mean([batch["local_coverage"] for batch in batch_metrics])),
               "preservation_weight": state.preservation_weight, "local_graph_weight": config.local_graph_weight,
               "mean_coverage_per_batch": float(np.mean([batch["coverage"] for batch in batch_metrics])),
               "species_mixing_fraction": mixing, "teacher_top15_recall_at_50": recall,
               "checkpoint_feasible": bool(feasible), "checkpoint_selected": bool(selected), "objective": objective}
        history.append(row)
        weight = float(np.clip(state.preservation_weight * math.exp(2 * (config.preservation_target - recall)), 1e-3, 100.))
        state = state.replace(epoch=jnp.int32(epoch), preservation_weight=weight)
        save_checkpoint(output_dir / "checkpoints" / f"epoch_{epoch:04d}", state, metadata(state), history=history)
        pd.DataFrame(history).to_csv(output_dir / "metric_history.csv", index=False)
        print(f"{objective} epoch {epoch}: loss={row['metric_loss']:.6f}, mixing={mixing:.4f}, recall={recall:.4f}", flush=True)
    validate_metric_history(pd.DataFrame(history), epochs=config.epochs)
    final = {"embeddings": embed_metric(module, state.best_params, artifact["macrogenes"], batch_size=config.batch_size, devices=devices),
             "species": artifact["species"], "obs_ids": artifact["obs_ids"]}
    validate_npz(final, "final_embeddings")
    selected_path = output_dir / "selected_orbax" / f"epoch_{int(state.epoch):04d}"
    selected_state = state.replace(params=state.best_params, opt_state=optimizer.init(state.best_params))
    if selected_path.exists():
        restored, existing_metadata, existing_history = restore_checkpoint(
            selected_path, selected_state, expected_metadata=metadata(selected_state, selected=True))
        if existing_metadata != metadata(selected_state, selected=True) or existing_history != history:
            raise ValueError("Existing selected checkpoint differs from completed run")
        for old, new in zip(jax.tree_util.tree_leaves(restored.params), jax.tree_util.tree_leaves(selected_state.params)):
            if not np.array_equal(old, new):
                raise ValueError("Existing selected checkpoint parameters differ from completed run")
    else:
        save_checkpoint(selected_path, selected_state, metadata(selected_state, selected=True), history=history)
    np.savez_compressed(output_dir / "final_embeddings.npz", **final)
    pd.DataFrame(history).to_csv(output_dir / "metric_history.csv", index=False)
    diagnostics = state.diagnostics
    summary = {"schema_version": 1, "trainer_version": 4, "objective": objective, "label_free": True,
               "artifact_keys_seen_by_trainer": sorted(artifact), "pretrain_checkpoint_sha256": source_hash,
               "label_free_artifact_sha256": artifact_hash, "seed": config.seed, "epochs": config.epochs,
               "batch_size": config.batch_size, "learning_rate": config.learning_rate,
               "initial_species_mixing_fraction": diagnostics["initial_mixing"],
               "initial_teacher_top15_recall_at_50": diagnostics["initial_recall"],
               "alignment_gradient_norm": diagnostics["alignment_gradient_norm"],
               "preservation_gradient_norm": diagnostics["preservation_gradient_norm"],
               "initial_preservation_weight": diagnostics["initial_weight"], "selected_epoch": state.best_epoch,
               "selected_species_mixing_fraction": state.best_mixing, "selected_teacher_top15_recall_at_50": state.best_recall,
               "selection_uses_labels": False, "selection_rule": f"maximize species_mixing_fraction subject to teacher_top15_recall_at_50 >= {config.preservation_target}",
               "mmd_bandwidth": bandwidth, "infonce_positive_cell_coverage": positive_coverage,
               "configuration": {**asdict(config), "objective": objective, "artifact": str(artifact_path),
                                 "pretrain_checkpoint": str(pretrain_checkpoint), "output_dir": str(output_dir)},
               "implementation": "jax", "orbax_checkpoint_path": str(selected_path.resolve()),
               "jax_devices": [str(device) for device in jax.devices()], "global_batch_size": config.batch_size}
    validate_run_summary(summary)
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return state, history, summary

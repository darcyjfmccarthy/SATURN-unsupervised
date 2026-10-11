"""Synchronous Orbax components and strict external checkpoint metadata."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

from jax_saturn.contracts.validation import load_json, validate_checkpoint_metadata


def latest_checkpoint(path):
    """Resolve a native checkpoint root to its latest committed epoch."""
    path = Path(path)
    if (path / "metadata.json").is_file():
        load_json(path / "metadata.json", "checkpoint")
        return path
    epochs = []
    for candidate in (path / "checkpoints").glob("epoch_*"):
        if (candidate / "metadata.json").is_file():
            metadata = load_json(candidate / "metadata.json", "checkpoint")
            if candidate.name != f"epoch_{metadata['epoch']:04d}":
                raise ValueError("Checkpoint directory name differs from metadata epoch")
            epochs.append((metadata["epoch"], candidate))
    if not epochs:
        raise FileNotFoundError(f"No completed epoch checkpoints under {path}")
    return max(epochs, key=lambda item: item[0])[1]


def save_checkpoint(path, state, metadata, *, history):
    """Publish metadata only after all Orbax components commit successfully."""
    validate_checkpoint_metadata(metadata)
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"Checkpoint already exists: {path}")
    training = {"epoch": int(state.epoch), "step": int(state.step), "history": history}
    for name in ("best_epoch", "best_step", "best_mixing", "best_recall", "preservation_weight"):
        if hasattr(state, name):
            value = getattr(state, name)
            training[name] = int(value) if name in {"best_epoch", "best_step"} else float(value)
    if hasattr(state, "diagnostics"):
        training["diagnostics"] = state.diagnostics
    if training["epoch"] != metadata["epoch"] or training["step"] != metadata["step"]:
        raise ValueError("Checkpoint metadata counters differ from training state")
    encoded = json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with ocp.Checkpointer(ocp.CompositeCheckpointHandler()) as checkpointer:
        components = dict(params=ocp.args.StandardSave(state.params),
                          opt_state=ocp.args.StandardSave(state.opt_state),
                          rng=ocp.args.StandardSave({"key": jax.random.key_data(state.rng)}),
                          training_state=ocp.args.JsonSave(training))
        if hasattr(state, "best_params"):
            components["best_params"] = ocp.args.StandardSave(state.best_params)
        checkpointer.save(path, args=ocp.args.Composite(**components))
    temporary = path / "metadata.json.tmp"
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path / "metadata.json")


def restore_checkpoint(path, template_state, *, expected_metadata=None):
    path = Path(path).resolve()
    metadata = load_json(path / "metadata.json", "checkpoint")
    if expected_metadata is not None:
        for key in ("model_kind", "seed", "hyperparameters", "species_names", "input_shapes", "source_manifest_sha256"):
            if metadata[key] != expected_metadata[key]:
                raise ValueError(f"Checkpoint {key} differs from requested run")
    with ocp.Checkpointer(ocp.CompositeCheckpointHandler()) as checkpointer:
        components = dict(params=ocp.args.StandardRestore(template_state.params),
                          opt_state=ocp.args.StandardRestore(template_state.opt_state),
                          rng=ocp.args.StandardRestore({"key": jax.random.key_data(template_state.rng)}),
                          training_state=ocp.args.JsonRestore())
        if hasattr(template_state, "best_params"):
            components["best_params"] = ocp.args.StandardRestore(template_state.best_params)
        restored = checkpointer.restore(path, args=ocp.args.Composite(**components))
    training = restored.training_state
    if training["epoch"] != metadata["epoch"] or training["step"] != metadata["step"]:
        raise ValueError("Restored counters differ from checkpoint metadata")
    state = template_state.replace(params=restored.params, opt_state=restored.opt_state,
        rng=jax.random.wrap_key_data(restored.rng["key"]),
        epoch=jnp.asarray(training["epoch"], dtype=jnp.int32), step=jnp.asarray(training["step"], dtype=jnp.int32))
    extra = {name: training[name] for name in ("best_epoch", "best_step", "best_mixing", "best_recall", "preservation_weight")
             if hasattr(template_state, name)}
    if hasattr(template_state, "best_params"):
        extra["best_params"] = restored.best_params
    if hasattr(template_state, "diagnostics"):
        extra["diagnostics"] = training["diagnostics"]
    if extra:
        state = state.replace(**extra)
    return state, metadata, training["history"]


def directory_sha256(path):
    """Canonical checksum of immutable checkpoint contents and relative paths."""
    import hashlib
    from jax_saturn.data.cache import file_sha256

    root = Path(path)
    records = [{"path": file.relative_to(root).as_posix(), "sha256": file_sha256(file)}
               for file in sorted(root.rglob("*")) if file.is_file()]
    if not records:
        raise ValueError("Checkpoint directory is empty")
    return hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_pretrain_metric_params(path, *, input_dim, hidden_dim, model_dim):
    """Read only a pretrain checkpoint's params, without labels or optimizer state."""
    from jax_saturn.models.saturn import SaturnPretrainModule
    from jax_saturn.training.baseline import metric_params_from_pretrain

    path = Path(path).resolve()
    metadata = load_json(path / "metadata.json", "checkpoint")
    if metadata["model_kind"] != "pretrain":
        raise ValueError("Label-free initialization requires an unlabeled pretrain checkpoint")
    config = metadata["hyperparameters"]
    if (config["num_macrogenes"], config["hidden_dim"], config["embed_dim"]) != (input_dim, hidden_dim, model_dim):
        raise ValueError("Pretrain checkpoint model dimensions differ from requested metric model")
    names = tuple(metadata["species_names"])
    counts = tuple(metadata["input_shapes"][name][1] for name in names)
    module = SaturnPretrainModule(names, counts, input_dim, hidden_dim, model_dim, dropout=config["dropout"])
    target = module.init(jax.random.key(0), jnp.ones((2, counts[0])), 0)["params"]
    with ocp.StandardCheckpointer() as checkpointer:
        params = checkpointer.restore(path / "params", target=target)
    return metric_params_from_pretrain(params), metadata

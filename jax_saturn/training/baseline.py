"""Frozen-macrogene labeled metric learning with CPU mining and compiled losses."""

import hashlib
import json
from pathlib import Path
from functools import partial

from flax import struct
from flax.core import freeze
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

from jax_saturn.contracts.validation import validate_anndata, validate_metric_history
from jax_saturn.data.batching import metric_batches
from jax_saturn.distributed.checkpoint import restore_checkpoint, save_checkpoint
from jax_saturn.losses.core import normalize, triplet_margin_loss
from jax_saturn.models.saturn import SaturnMetricModule
from jax_saturn.models.precision import matmul_dtype, resolve_precision
from .mining import cross_species_candidates, filter_triplets, normalize_numpy, pad_triplets


@struct.dataclass
class MetricTrainState:
    params: object
    opt_state: object
    rng: jax.Array
    step: jax.Array
    epoch: jax.Array


def metric_params_from_pretrain(params):
    return freeze({key: params[key] for key in ("encoder", "cl_layer_norm")})


def create_metric_state(params, optimizer, seed):
    return MetricTrainState(params, optimizer.init(params), jax.random.key(seed), jnp.int32(0), jnp.int32(0))


@partial(jax.jit, static_argnums=(0,))
def _metric_inference(module, params, values):
    return module.apply({"params": params}, values)


def embed_metric(module, params, macrogenes, *, batch_size=512, devices=None):
    if batch_size < 1 or len(macrogenes) < 1:
        raise ValueError("Inference requires positive batch size and nonempty macrogenes")
    if devices is not None:
        from jax_saturn.distributed.inference import metric_embeddings
        return metric_embeddings(module, params, macrogenes, batch_size=batch_size, devices=devices)
    outputs = []
    for start in range(0, len(macrogenes), batch_size):
        values = np.asarray(macrogenes[start:start + batch_size], dtype=np.float32)
        count = len(values)
        padded = np.zeros((batch_size, values.shape[1]), dtype=np.float32)
        padded[:count] = values
        outputs.append(np.asarray(_metric_inference(module, params, padded)[:count], dtype=np.float32))
    return np.concatenate(outputs)


def make_metric_step(module, optimizer, *, margin=.2):
    @jax.jit
    def preview(state, values):
        _, dropout_key = jax.random.split(state.rng)
        return module.apply({"params": state.params}, values, train=True, rngs={"dropout": dropout_key})

    @jax.jit
    def step(state, values, indices, valid_mask):
        rng, dropout_key = jax.random.split(state.rng)
        def objective(params):
            output = module.apply({"params": params}, values, train=True, rngs={"dropout": dropout_key})
            # Reference training normalizes before invoking the normalized distance.
            output = normalize(output)
            return triplet_margin_loss(output, indices, margin=margin, valid_mask=valid_mask)
        loss, gradients = jax.value_and_grad(objective)(state.params)
        updates, opt_state = optimizer.update(gradients, state.opt_state, state.params)
        return state.replace(params=optax.apply_updates(state.params, updates), opt_state=opt_state,
                             rng=rng, step=state.step + 1), {"loss": loss, "gradient_norm": optax.global_norm(gradients)}
    return preview, step


def emit_metric_anndata(module, params, pretrain_adata, path, *, batch_size=512, devices=None):
    import anndata as ad

    result = ad.AnnData(embed_metric(module, params, pretrain_adata.obsm["macrogenes"], batch_size=batch_size, devices=devices),
                        obs=pretrain_adata.obs.copy(), obsm={"macrogenes": np.asarray(pretrain_adata.obsm["macrogenes"], dtype=np.float32)})
    validate_anndata(result, expected_obs_ids=pretrain_adata.obs_names, expected_species=pretrain_adata.obs["species"])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    result.write_h5ad(path)
    return result


def train_baseline(pretrain_params, pretrain_adata, *, output_dir, checkpoint_dir=None,
                   epochs=30, batch_size=512, learning_rate=.001, hidden_dim=256,
                   model_dim=256, seed=0, margin=.2, polling_freq=5,
                   source_manifest_sha256, resume=None, devices=None, mixed_precision="fp32"):
    validate_anndata(pretrain_adata)
    mixed_precision = resolve_precision(mixed_precision)
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or polling_freq < 1:
        raise ValueError("Invalid baseline epochs, batch size, learning rate or polling frequency")
    if devices is not None:
        from jax_saturn.distributed.metric import make_metric_pmap, replicate_state, unreplicate_state, validate_devices
        devices = validate_devices(devices, batch_size=batch_size)
    output_dir = Path(output_dir)
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else output_dir.parent / "shared/metric_orbax"
    output_dir.mkdir(parents=True, exist_ok=True)
    macros = np.asarray(pretrain_adata.obsm["macrogenes"], dtype=np.float32)
    species_names, species_codes = np.unique(np.asarray(pretrain_adata.obs["species"]).astype(str), return_inverse=True)
    labels = pd.Categorical(pretrain_adata.obs["labels"]).codes.astype(np.int32)
    refs = pd.Categorical(pretrain_adata.obs["ref_labels"]).codes.astype(np.int32)
    module = SaturnMetricModule(macros.shape[1], hidden_dim, model_dim, dtype=matmul_dtype(mixed_precision))
    optimizer = optax.adam(learning_rate, eps=1e-8)
    state = create_metric_state(metric_params_from_pretrain(pretrain_params), optimizer, seed)
    fingerprint = hashlib.sha256(macros.astype("<f4").tobytes())
    fingerprint.update(json.dumps({"obs_ids": list(pretrain_adata.obs_names), "species": list(map(str, pretrain_adata.obs["species"])),
                                   "labels": list(map(str, pretrain_adata.obs["labels"])), "refs": list(map(str, pretrain_adata.obs["ref_labels"]))},
                                  sort_keys=True, separators=(",", ":")).encode())
    config = {"seed": seed, "batch_size": batch_size, "learning_rate": learning_rate,
              "hidden_dim": hidden_dim, "embed_dim": model_dim, "input_dim": macros.shape[1],
              "dropout": module.dropout, "margin": margin, "miner_type": "cross_species",
              "triplet_type": "semihard", "mnn": True, "pretrain_artifact_sha256": fingerprint.hexdigest()}
    if mixed_precision != "fp32":
        config["mixed_precision"] = mixed_precision
    if devices is not None:
        config.update(execution="pmap", local_device_count=len(devices))

    def metadata(state):
        return {"schema_version": 1, "implementation": "jax", "model_kind": "baseline_metric",
                "epoch": int(state.epoch), "step": int(state.step), "seed": seed,
                "hyperparameters": config, "species_names": species_names.tolist(),
                "input_shapes": {"macrogenes": [batch_size, macros.shape[1]]},
                "source_manifest_sha256": source_manifest_sha256}

    history = []
    if resume is not None:
        state, _, history = restore_checkpoint(resume, state, expected_metadata=metadata(state))
        validate_metric_history(pd.DataFrame(history), epochs=int(state.epoch))
    if int(state.epoch) > epochs:
        raise ValueError("Baseline checkpoint epoch exceeds requested epochs")
    first_epoch = int(state.epoch) + 1
    if devices is None:
        preview, step = make_metric_step(module, optimizer, margin=margin)
    else:
        preview, step = make_metric_pmap(module, optimizer, devices, margin=margin)
        state = replicate_state(state, devices)
    for epoch in range(first_epoch, epochs + 1):
        losses, candidates, active, mined = [], 0, 0, 0
        for batch in metric_batches(macros, species_codes, batch_size, seed=seed, epoch=epoch, labels=labels, ref_labels=refs):
            values = jnp.asarray(batch["macrogenes"])
            count = int(batch["valid_mask"].sum())
            embeddings = normalize_numpy(np.asarray(preview(state, values))[:count])
            step_number = int(state.step if devices is None else state.step[0])
            rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, step_number]))
            tuples = cross_species_candidates(embeddings, batch["labels"][:count], batch["species_codes"][:count], rng=rng)
            candidates += len(tuples[0])
            tuples = filter_triplets(embeddings, tuples, margin=margin)
            mined += len(tuples[0])
            if len(tuples[0]):
                normalized = normalize_numpy(embeddings)
                a, p, n = tuples
                violation = np.sum(normalized[a] * normalized[n], axis=-1) - np.sum(normalized[a] * normalized[p], axis=-1) + margin
                active += int(np.count_nonzero(violation > 0))
            padded, mask = pad_triplets(tuples)
            state, metrics = step(state, values, tuple(jnp.asarray(x) for x in padded), jnp.asarray(mask))
            if not all(np.isfinite(np.asarray(x)).all() for x in metrics.values()):
                raise FloatingPointError(f"Nonfinite metric loss or gradient at epoch {epoch}")
            losses.append(float(metrics["loss"]))
        history.append({"epoch": epoch, "metric_loss": float(np.mean(losses)), "cross_metric_loss": float(np.mean(losses)),
                        "cross_candidate_triplets": candidates, "cross_active_triplets": active,
                        "cross_active_triplet_fraction": active / mined if mined else 0.,
                        "metric_miner_type": "cross_species", "metric_triplet_type": "semihard"})
        state = state.replace(epoch=jnp.int32(epoch) if devices is None else jnp.full((len(devices),), epoch, dtype=jnp.int32))
        checkpoint_state = state if devices is None else unreplicate_state(state)
        save_checkpoint(checkpoint_dir / "checkpoints" / f"epoch_{epoch:04d}", checkpoint_state, metadata(checkpoint_state), history=history)
        pd.DataFrame(history).to_csv(output_dir / "metric_history.csv", index=False)
        if epoch % polling_freq == 0:
            emit_metric_anndata(module, checkpoint_state.params, pretrain_adata, output_dir / f"adata_epoch_{epoch}.h5ad", batch_size=batch_size, devices=devices)
        print(f"Metric epoch {epoch}: loss={history[-1]['metric_loss']:.6f}, candidates={candidates}, active={active}", flush=True)
    if devices is not None:
        state = unreplicate_state(state)
    validate_metric_history(pd.DataFrame(history), epochs=epochs)
    pd.DataFrame(history).to_csv(output_dir / "metric_history.csv", index=False)
    result = emit_metric_anndata(module, state.params, pretrain_adata, output_dir / "final_adata.h5ad", batch_size=batch_size, devices=devices)
    final_checkpoint = checkpoint_dir / "checkpoints" / f"epoch_{int(state.epoch):04d}"
    if not final_checkpoint.exists() and resume is not None:
        final_checkpoint = Path(resume)
    (output_dir / "metric_summary.json").write_text(json.dumps({
        "implementation": "jax", "stage": "baseline_metric", "epoch": int(state.epoch),
        "step": int(state.step), "orbax_checkpoint_path": str(final_checkpoint.resolve()),
        "source_manifest_sha256": source_manifest_sha256,
    }, indent=2, sort_keys=True) + "\n")
    return state, history, result

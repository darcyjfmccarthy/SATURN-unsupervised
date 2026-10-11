"""CPU/GPU pretraining with fixed-shape mixed batches and Orbax epoch resume."""

from pathlib import Path
import hashlib
from functools import partial

from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

from jax_saturn.contracts.validation import validate_anndata
from jax_saturn.data.batching import equal_species_batches
from jax_saturn.distributed.checkpoint import restore_checkpoint, save_checkpoint
from jax_saturn.losses.core import gene_weight_ranking_loss, l1_loss, zinb_reconstruction_loss
from jax_saturn.models.saturn import init_pretrain_params


@struct.dataclass
class PretrainTrainState:
    params: object
    opt_state: object
    rng: jax.Array
    step: jax.Array
    epoch: jax.Array


def create_state(module, scores, optimizer, seed):
    key, params_key = jax.random.split(jax.random.key(seed))
    params = init_pretrain_params(module, params_key, scores)
    return PretrainTrainState(params, optimizer.init(params), key, jnp.int32(0), jnp.int32(0))


def make_pretrain_step(module, optimizer, gene_embeddings, *, l1_penalty=0., pe_sim_penalty=.2):
    gene_embeddings = jnp.asarray(gene_embeddings, dtype=jnp.float32)

    @partial(jax.jit, static_argnames=("code",))
    def reconstruction_gradient(params, values, valid_mask, dropout_rng, *, code):
        def objective(params):
            outputs = module.apply({"params": params}, values, code, train=True,
                                   rngs={"dropout": dropout_rng})
            reconstruction = zinb_reconstruction_loss(values, outputs.px_rate, outputs.px_r,
                                                       outputs.px_drop, valid_mask=valid_mask)
            return reconstruction / jnp.maximum(jnp.sum(valid_mask), 1)
        return jax.value_and_grad(objective)(params)

    @jax.jit
    def regularizer_gradient(params, proteins, dropout_rng, ranking_rng):
        def objective(params):
            log_weights = params["macrogene"]["log_gene_to_macrogene"]
            lasso = l1_penalty * l1_loss(log_weights)
            learned = module.apply({"params": params}, train=True, method=module.gene_weight_embeddings,
                                   rngs={"dropout": dropout_rng})
            ranking = pe_sim_penalty * gene_weight_ranking_loss(learned, proteins, key=ranking_rng)
            return lasso + ranking, {"l1_loss": lasso, "ranking_loss": ranking}
        return jax.value_and_grad(objective, has_aux=True)(params)

    @jax.jit
    def add_gradients(left, right):
        return jax.tree_util.tree_map(jnp.add, left, right)

    @jax.jit
    def apply_gradients(state, gradients, next_rng):
        updates, opt_state = optimizer.update(gradients, state.opt_state, state.params)
        new_state = state.replace(params=optax.apply_updates(state.params, updates), opt_state=opt_state,
                                  rng=next_rng, step=state.step + 1)
        return new_state, optax.global_norm(gradients)

    def step(state, batch):
        # Separate species programs keep XLA's GPU fusion analysis bounded.
        # All gradients use the same parameters; Adam runs once per mixed batch.
        next_rng, dropout_rng, ranking_rng = jax.random.split(state.rng, 3)
        gradients, species_losses = None, []
        for code, record in enumerate(batch):
            loss, species_gradients = reconstruction_gradient(state.params, record["values"], record["valid_mask"],
                jax.random.fold_in(dropout_rng, code), code=code)
            species_losses.append(loss)
            gradients = species_gradients if gradients is None else add_gradients(gradients, species_gradients)
        # Keep protein vectors in a device buffer rather than a large constant.
        (regularization, metrics), regularizer_gradients = regularizer_gradient(
            state.params, gene_embeddings, jax.random.fold_in(dropout_rng, len(batch)), ranking_rng)
        gradients = add_gradients(gradients, regularizer_gradients)
        new_state, gradient_norm = apply_gradients(state, gradients, next_rng)
        species_losses = jnp.stack(species_losses)
        return new_state, {**metrics, "species_losses": species_losses,
                           "loss": jnp.sum(species_losses) + regularization, "gradient_norm": gradient_norm}

    return step


def checkpoint_metadata(module, data, state, config):
    return {"schema_version": 1, "implementation": "jax", "model_kind": "pretrain",
            "epoch": int(state.epoch), "step": int(state.step), "seed": config["seed"],
            "hyperparameters": config, "species_names": list(module.species_names),
            "input_shapes": {name: [config["batch_size"], count]
                             for name, count in zip(module.species_names, module.gene_counts)},
            "source_manifest_sha256": data.source_manifest_sha256}


def train_pretrain(module, data, scores, *, output_dir, epochs=20, batch_size=512, learning_rate=.0005,
                   seed=0, l1_penalty=0., pe_sim_penalty=.2, resume=None, devices=None):
    if epochs < 0 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("Invalid pretraining epochs, batch size or learning rate")
    if devices is not None:
        from jax_saturn.distributed.metric import replicate_state, unreplicate_state, validate_devices
        from jax_saturn.distributed.pretrain import make_pretrain_pmap
        devices = validate_devices(devices, batch_size=batch_size)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = optax.adam(learning_rate, eps=1e-8)
    state = create_state(module, scores, optimizer, seed)
    config = {"seed": seed, "batch_size": batch_size, "learning_rate": learning_rate,
              "l1_penalty": l1_penalty, "pe_sim_penalty": pe_sim_penalty,
              "num_macrogenes": module.num_macrogenes, "hidden_dim": module.hidden_dim,
              "embed_dim": module.embed_dim, "dropout": module.dropout, "dtype": str(jnp.dtype(module.dtype)),
              "embedding_cache_sha256": data.embedding_cache_sha256,
              "centroid_scores_sha256": hashlib.sha256(np.asarray(scores, dtype="<f4").tobytes()).hexdigest(),
              "gene_names": [list(names) for names in data.gene_names]}
    if devices is not None:
        config.update(execution="pmap", local_device_count=len(devices))
    history = []
    if resume is not None:
        state, _, history = restore_checkpoint(resume, state, expected_metadata=checkpoint_metadata(module, data, state, config))
        if [row["epoch"] for row in history] != list(range(1, int(state.epoch) + 1)):
            raise ValueError("Checkpoint history does not match completed epochs")
    if int(state.epoch) > epochs:
        raise ValueError("Checkpoint epoch exceeds requested training epochs")
    first_epoch = int(state.epoch) + 1
    if devices is None:
        step = make_pretrain_step(module, optimizer, data.gene_embeddings,
                                 l1_penalty=l1_penalty, pe_sim_penalty=pe_sim_penalty)
    else:
        step = make_pretrain_pmap(module, optimizer, data.gene_embeddings, devices,
                                 l1_penalty=l1_penalty, pe_sim_penalty=pe_sim_penalty)
        state = replicate_state(state, devices)
    for epoch in range(first_epoch, epochs + 1):
        measurements = []
        species_measurements = [[] for _ in module.species_names]
        for batch in equal_species_batches(data.values, batch_size, seed=seed, epoch=epoch):
            # Only loss inputs cross the device boundary; metadata remains host-side.
            device_batch = tuple({key: jnp.asarray(record[key]) for key in ("values", "valid_mask")} for record in batch)
            state, metrics = step(state, device_batch)
            metrics = jax.device_get(metrics)
            if not all(np.isfinite(value).all() for value in metrics.values()):
                raise FloatingPointError(f"Nonfinite pretrain loss or gradient at epoch {epoch}")
            measurements.append(metrics)
            for code, record in enumerate(batch):
                if record["valid_mask"].any():
                    species_measurements[code].append(float(metrics["species_losses"][code]))
        row = {"epoch": epoch, **{key: float(np.mean([m[key] for m in measurements]))
                                 for key in ("loss", "l1_loss", "ranking_loss", "gradient_norm")}}
        row.update({name: float(np.mean(species_measurements[code])) for code, name in enumerate(module.species_names)})
        history.append(row)
        state = state.replace(epoch=jnp.int32(epoch) if devices is None else jnp.full((len(devices),), epoch, dtype=jnp.int32))
        checkpoint_state = state if devices is None else unreplicate_state(state)
        path = output_dir / "checkpoints" / f"epoch_{epoch:04d}"
        save_checkpoint(path, checkpoint_state, checkpoint_metadata(module, data, checkpoint_state, config), history=history)
        pd.DataFrame(history).to_csv(output_dir / "pretrain_losses.csv", index=False)
        print(f"Pretrain epoch {epoch}: loss={row['loss']:.6f}", flush=True)
    return state if devices is None else unreplicate_state(state), history


def emit_pretrain_anndata(module, params, data, path, *, batch_size=512, devices=None):
    import anndata as ad

    matrices, macrogenes = [], []
    if devices is not None:
        from jax_saturn.distributed.inference import pretrain_embeddings
        embedding, macros = pretrain_embeddings(module, params, data.values, batch_size=batch_size, devices=devices)
        matrices, macrogenes = [embedding], [macros]
    for code, values in enumerate(data.values if devices is None else ()):
        infer = jax.jit(lambda x: module.apply({"params": params}, x, code))
        for start in range(0, len(values), batch_size):
            count = min(batch_size, len(values) - start)
            padded = np.zeros((batch_size, values.shape[1]), dtype=np.float32)
            padded[:count] = values[start:start + count]
            output = infer(padded)
            matrices.append(np.asarray(output.embedding[:count], dtype=np.float32))
            macrogenes.append(np.asarray(output.macrogenes[:count], dtype=np.float32))
    adata = ad.AnnData(np.concatenate(matrices), obs=data.obs.copy(),
                      obsm={"macrogenes": np.concatenate(macrogenes)})
    validate_anndata(adata, expected_obs_ids=data.obs.index, expected_species=data.obs["species"])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)
    return adata

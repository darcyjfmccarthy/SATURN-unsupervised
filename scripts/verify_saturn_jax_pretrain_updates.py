#!/usr/bin/env python3
"""Bounded pretraining parity using cached HV genes and actual atlas counts."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--centroids", type=Path, required=True)
    parser.add_argument("--embedding-cache-dir", type=Path, required=True)
    parser.add_argument("--pretrain-checkpoint", type=Path,
                        help="Optional trained-weight stress check; default checks fresh reference initialization")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--compilation-cache-dir", type=Path)
    args = parser.parse_args()
    if not 1 <= args.updates <= 10 or min(args.batch_size, args.hidden_dim, args.model_dim) < 1 or args.seed < 0:
        parser.error("Positive dimensions and 1–10 updates are required")
    if args.output.exists():
        parser.error("Use a fresh receipt path")

    import anndata as ad
    from flax import linen as nn
    from flax.traverse_util import flatten_dict
    import hashlib
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    import torch
    from jax_saturn.data.cache import file_sha256, load_cache
    from jax_saturn.data.manifest import load_manifest
    from jax_saturn.models.conversion import torch_to_flax
    from jax_saturn.models.saturn import SaturnPretrainModule
    from jax_saturn.training.pretrain import create_state, make_pretrain_step
    from model.saturn_model import SATURNPretrainModel

    if args.compilation_cache_dir:
        jax.config.update("jax_compilation_cache_dir", str(args.compilation_cache_dir.resolve()))
    if args.reference_device == "cuda" and not torch.cuda.is_available():
        parser.error("PyTorch CUDA unavailable")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    rows = load_manifest(args.manifest, check_paths=True)
    cache, metadata = load_cache(args.centroids, kind="centroid")
    names = tuple(row.species for row in rows)
    if names != tuple(cache["species_names"]) or file_sha256(args.manifest) != metadata["source_manifest_sha256"]:
        parser.error("Centroid species/provenance differs from manifest")
    if args.batch_size < len(names):
        parser.error("Batch size must include every species")
    offsets = {name: (int(start), int(end)) for name, start, end in zip(names, cache["species_gene_starts"], cache["species_gene_ends"])}
    rng = np.random.default_rng(args.seed)
    batches = [[] for _ in range(args.updates)]
    proteins, cache_hashes, sampled_obs = [], [], {}
    for code, row in enumerate(rows):
        start, end = offsets[row.species]
        genes = [str(gene)[len(row.species) + 1:] for gene in cache["all_gene_names"][start:end]]
        embedding_path = args.embedding_cache_dir / f"{row.species}.npz"
        embedding, embedding_metadata = load_cache(embedding_path, kind="embedding")
        if embedding_metadata["species"] != row.species or embedding_metadata["source_sha256"] != file_sha256(row.embedding_path):
            parser.error(f"Embedding cache provenance differs for {row.species}")
        cache_hashes.append({"species": row.species, "sha256": file_sha256(embedding_path)})
        lookup = {gene: index for index, gene in enumerate(embedding["gene_symbols"])}
        proteins.append(embedding["embeddings"][[lookup[gene.lower()] for gene in genes]])
        atlas = ad.read_h5ad(row.path, backed="r")
        try:
            indices = atlas.var_names.get_indexer(genes)
            if np.any(indices < 0):
                parser.error(f"Cached genes absent from atlas for {row.species}")
            sampled_obs[row.species] = []
            for update in range(args.updates):
                valid_count = args.batch_size // len(names) + (code < args.batch_size % len(names))
                valid_count = min(valid_count, atlas.n_obs)
                selected = np.sort(rng.choice(atlas.n_obs, valid_count, replace=False))
                expression = atlas[selected, indices].X
                expression = expression.toarray() if hasattr(expression, "toarray") else np.asarray(expression)
                if not np.isfinite(expression).all() or np.any(expression < 0):
                    parser.error("Expression must contain finite nonnegative counts")
                values = np.zeros((args.batch_size, len(genes)), dtype=np.float32)
                values[:valid_count] = expression
                batches[update].append({"values": jnp.asarray(values), "valid_mask": jnp.arange(args.batch_size) < valid_count})
                sampled_obs[row.species].append(atlas.obs_names[selected].tolist())
        finally:
            atlas.file.close()
    cache_digest = hashlib.sha256(json.dumps(cache_hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if cache_digest != metadata["embedding_cache_sha256"]:
        parser.error("Embedding cache identity differs from centroid provenance")
    proteins = np.concatenate(proteins)
    counts = tuple(end - start for start, end in offsets.values())
    reference = SATURNPretrainModel(torch.tensor(cache["scores"]), hidden_dim=args.hidden_dim,
                                   embed_dim=args.model_dim, dropout=0., species_to_gene_idx=offsets)
    if args.pretrain_checkpoint:
        reference.load_state_dict(torch.load(args.pretrain_checkpoint, map_location="cpu", weights_only=True), strict=True)
    reference.to(args.reference_device)
    for layer in reference.modules():
        if isinstance(layer, torch.nn.Dropout):
            layer.p = 0.
    module = SaturnPretrainModule(names, counts, cache["scores"].shape[1], args.hidden_dim, args.model_dim, dropout=0.)
    optimizer = optax.adam(.0005, eps=1e-8)
    state = create_state(module, cache["scores"], optimizer, args.seed)
    params = torch_to_flax(reference.state_dict(), state.params, model_kind="pretrain")
    state = state.replace(params=params, opt_state=optimizer.init(params))
    step = make_pretrain_step(module, optimizer, proteins, l1_penalty=0., pe_sim_penalty=.2)
    torch_optimizer = torch.optim.Adam(reference.parameters(), lr=.0005, eps=1e-8)
    protein_tensor = torch.tensor(proteins, device=args.reference_device)
    records, failures = [], []
    with patch.object(nn.Dropout, "__call__", lambda self, inputs, **kwargs: inputs):
        for update, batch in enumerate(batches):
            update_started = time.monotonic()
            torch_optimizer.zero_grad(set_to_none=True)
            losses = []
            for name, record in zip(names, batch):
                values = np.asarray(record["values"])
                tensor = torch.tensor(values, device=args.reference_device)
                output = reference(tensor, name)
                valid = torch.tensor(np.asarray(record["valid_mask"]), device=args.reference_device)
                losses.append(reference.get_reconstruction_loss(tensor, *output[4:])[valid].mean())
            _, _, ranking_key = jax.random.split(state.rng, 3)
            pairs = np.asarray(jax.random.randint(ranking_key, (sum(counts),), 0, sum(counts)))
            with patch("torch.randint", return_value=torch.tensor(pairs, dtype=torch.long, device=args.reference_device)):
                ranking = .2 * reference.gene_weight_ranking_loss(reference.p_weights.exp(), protein_tensor)
            loss = sum(losses) + ranking
            loss.backward()
            norm = torch.sqrt(sum(parameter.grad.square().sum() for parameter in reference.parameters() if parameter.grad is not None))
            torch_optimizer.step()
            state, metrics = step(state, tuple(batch))
            for key, wanted in (("loss", loss), ("ranking_loss", ranking), ("gradient_norm", norm)):
                actual, wanted = float(metrics[key]), float(wanted.detach().cpu())
                if not np.isclose(actual, wanted, rtol=1e-4, atol=1e-5):
                    failures.append({"update": update + 1, "field": key, "reference": wanted, "jax": actual})
            expected = flatten_dict(torch_to_flax(reference.state_dict(), params, model_kind="pretrain"))
            max_error = 0.
            for path, actual in flatten_dict(state.params).items():
                actual, wanted = np.asarray(actual), np.asarray(expected[path])
                parameter_error = float(np.max(np.abs(actual - wanted)))
                max_error = max(max_error, parameter_error)
                mismatches = ~np.isclose(actual, wanted, rtol=1e-4, atol=1e-5)
                if np.any(mismatches):
                    reference_moments = {name: torch_optimizer.state[parameter].get("exp_avg", torch.zeros_like(parameter))
                                         for name, parameter in reference.named_parameters()}
                    reference_moments = flatten_dict(torch_to_flax(reference_moments, params, model_kind="pretrain"))
                    actual_moments = flatten_dict(state.opt_state[0].mu)
                    index = np.unravel_index(np.argmax(np.where(mismatches, np.abs(actual - wanted), -1)), actual.shape)
                    failures.append({"update": update + 1, "field": "/".join(path),
                                     "mismatched_entries": int(np.sum(mismatches)), "max_absolute_error": parameter_error,
                                     "worst_index": list(map(int, index)), "reference": float(wanted[index]), "jax": float(actual[index]),
                                     "reference_adam_first_moment": float(np.asarray(reference_moments[path])[index]),
                                     "jax_adam_first_moment": float(np.asarray(actual_moments[path])[index])})
            record = {"update": update + 1, "reference_loss": float(loss.detach().cpu()), "jax_loss": float(metrics["loss"]),
                      "max_parameter_absolute_error": max_error, "elapsed_seconds": time.monotonic() - update_started}
            records.append(record)
            print(json.dumps(record), flush=True)
            if failures:
                break
    report = {"schema_version": 1, "success": not failures, "failures": failures, "updates": records,
              "timestamp_utc": datetime.now(timezone.utc).isoformat(), "requested_updates": args.updates,
              "scope": "Matched pretraining updates on real counts in cached HV-gene order, copied reference weights and matched ranking draws, with dropout disabled. Does not verify HV selection, whole epochs or scientific quality.",
              "manifest_sha256": file_sha256(args.manifest), "centroids_sha256": file_sha256(args.centroids),
              "starting_weights": "trained_checkpoint" if args.pretrain_checkpoint else "fresh_reference_initialization",
              "pretrain_checkpoint_sha256": file_sha256(args.pretrain_checkpoint) if args.pretrain_checkpoint else None,
              "embedding_cache_sha256": cache_digest, "species": list(names), "gene_counts": list(counts),
              "num_macrogenes": module.num_macrogenes, "hidden_dim": args.hidden_dim, "model_dim": args.model_dim,
              "batch_size": args.batch_size, "seed": args.seed, "sampled_obs": sampled_obs,
              "reference_device": args.reference_device, "jax_devices": [str(device) for device in jax.devices()],
              "rtol": 1e-4, "atol": 1e-5, "xla_flags": os.environ.get("XLA_FLAGS", ""),
              "elapsed_seconds": time.monotonic() - started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

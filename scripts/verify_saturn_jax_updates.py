#!/usr/bin/env python3
"""Bounded baseline Adam parity on real artifacts and frozen reference triplets."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--triplets", type=Path, required=True)
    parser.add_argument("--pretrain-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--compilation-cache-dir", type=Path)
    args = parser.parse_args()
    if not 1 <= args.updates <= 10:
        parser.error("--updates must be between 1 and 10; this command is bounded")
    if args.batch_size < 3 or args.hidden_dim < 1 or args.seed < 0:
        parser.error("Invalid batch size, hidden dimension or seed")
    if args.output.exists():
        parser.error("Use a fresh receipt path")

    from flax.traverse_util import flatten_dict
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    import torch
    from distances.cosine_similarity import CosineSimilarity
    from losses.triplet_margin_loss import TripletMarginLoss
    from model.saturn_model import SATURNMetricModel
    from jax_saturn.contracts.validation import load_npz
    from jax_saturn.data.cache import file_sha256
    from jax_saturn.models.conversion import torch_to_flax
    from jax_saturn.models.saturn import SaturnMetricModule
    from jax_saturn.training.baseline import create_metric_state, make_metric_step
    from jax_saturn.training.mining import pad_triplets

    if args.compilation_cache_dir:
        jax.config.update("jax_compilation_cache_dir", str(args.compilation_cache_dir.resolve()))
    if args.reference_device == "cuda" and not torch.cuda.is_available():
        parser.error("PyTorch CUDA reference requested but unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    started = time.monotonic()
    artifact = load_npz(args.artifact, "label_free")
    triplets = load_npz(args.triplets, "evaluation_triplets")
    if not np.array_equal(artifact["obs_ids"], triplets["obs_ids"]):
        parser.error("Artifact and frozen triplet observation order differs")
    count = len(triplets["anchor"])
    if count < 1:
        parser.error("Frozen triplets must be nonempty")
    macros = artifact["macrogenes"]
    model_dim = artifact["embeddings"].shape[1]
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    reference = SATURNMetricModel(input_dim=macros.shape[1], hidden_dim=args.hidden_dim,
                                 embed_dim=model_dim, dropout=0.)
    # Only trusted local reference checkpoints should be passed to torch.load.
    source = torch.load(args.pretrain_checkpoint, map_location="cpu", weights_only=True)
    selected = {key: value for key, value in source.items()
                if key.startswith(("encoder.", "cl_layer_norm."))}
    reference.load_state_dict(selected, strict=True)
    reference.to(args.reference_device)
    module = SaturnMetricModule(macros.shape[1], args.hidden_dim, model_dim, dropout=0.)
    template = module.init(jax.random.key(args.seed), np.zeros((args.batch_size, macros.shape[1]), np.float32))["params"]
    params = torch_to_flax(selected, template, model_kind="metric")
    optimizer = optax.adam(.001, eps=1e-8)
    state = create_metric_state(params, optimizer, args.seed)
    _, step = make_metric_step(module, optimizer)
    torch_optimizer = torch.optim.Adam(reference.parameters(), lr=.001, eps=1e-8)
    criterion = TripletMarginLoss(margin=.2, distance=CosineSimilarity())
    rng = np.random.default_rng(args.seed)
    records, observed, failures = [], set(), []
    for update in range(args.updates):
        update_started = time.monotonic()
        # At most B/3 triplets ensures every involved row fits in the fixed batch.
        chosen = rng.choice(count, size=min(count, args.batch_size // 3), replace=False)
        global_indices = np.stack([triplets[key][chosen] for key in ("anchor", "positive", "negative")])
        rows, local = np.unique(global_indices, return_inverse=True)
        local = local.reshape(global_indices.shape)
        observed.update(rows.tolist())
        values = np.zeros((args.batch_size, macros.shape[1]), dtype=np.float32)
        values[:len(rows)] = macros[rows]
        padded, mask = pad_triplets(tuple(local))
        torch_optimizer.zero_grad(set_to_none=True)
        output = torch.nn.functional.normalize(reference(torch.tensor(values, device=args.reference_device)))
        loss = criterion(output, torch.arange(len(values), device=args.reference_device),
                         tuple(torch.tensor(indices, device=args.reference_device) for indices in local))
        loss.backward()
        norm = torch.sqrt(sum(parameter.grad.square().sum() for parameter in reference.parameters()
                              if parameter.grad is not None))
        torch_optimizer.step()
        state, metrics = step(state, values, tuple(jnp.asarray(indices) for indices in padded), mask)
        for key, expected in (("loss", loss), ("gradient_norm", norm)):
            expected = float(expected.detach().cpu())
            actual = float(metrics[key])
            if not np.isclose(actual, expected, rtol=1e-4, atol=1e-5):
                failures.append({"update": update + 1, "field": key,
                                 "reference": expected, "jax": actual})
        expected = flatten_dict(torch_to_flax(reference.state_dict(), params, model_kind="metric"))
        reference_moments = {name: torch_optimizer.state[parameter].get("exp_avg", torch.zeros_like(parameter))
                             for name, parameter in reference.named_parameters()}
        expected_moments = flatten_dict(torch_to_flax(reference_moments, params, model_kind="metric"))
        actual_moments = flatten_dict(state.opt_state[0].mu)
        max_error = 0.
        for path, actual in flatten_dict(state.params).items():
            actual = np.asarray(actual)
            wanted = np.asarray(expected[path])
            errors = np.abs(actual - wanted)
            parameter_error = float(np.max(errors))
            max_error = max(max_error, parameter_error)
            mismatches = ~np.isclose(actual, wanted, rtol=1e-4, atol=1e-5)
            if np.any(mismatches):
                index = np.unravel_index(np.argmax(np.where(mismatches, errors, -1.)), errors.shape)
                failures.append({"update": update + 1, "field": "/".join(path),
                                 "mismatched_entries": int(np.sum(mismatches)),
                                 "max_absolute_error": parameter_error,
                                 "worst_index": list(map(int, index)),
                                 "reference": float(wanted[index]), "jax": float(actual[index]),
                                 "reference_adam_first_moment": float(np.asarray(expected_moments[path])[index]),
                                 "jax_adam_first_moment": float(np.asarray(actual_moments[path])[index])})
        if int(state.step) != update + 1:
            raise AssertionError("Optimizer step count differs")
        record = {"update": update + 1, "triplets": len(chosen), "cells": len(rows),
                  "species": sorted(np.unique(artifact["species"][rows]).tolist()),
                  "reference_loss": float(loss.detach()), "jax_loss": float(metrics["loss"]),
                  "max_parameter_absolute_error": max_error,
                  "elapsed_seconds": time.monotonic() - update_started}
        records.append(record)
        print(json.dumps(record), flush=True)
        if failures:
            break
    report = {"schema_version": 1, "success": not failures, "failures": failures,
              "requested_updates": args.updates, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Matched baseline Adam updates on sampled real cells with fixed reference triplets and dropout disabled. Does not verify full epochs, candidate mining, scientific quality, or label-free objectives.",
              "artifact_sha256": file_sha256(args.artifact), "triplets_sha256": file_sha256(args.triplets),
              "pretrain_checkpoint_sha256": file_sha256(args.pretrain_checkpoint),
              "atlas_cells": len(macros), "checked_cells": len(observed), "frozen_triplet_count": count,
              "species": sorted(np.unique(artifact["species"]).tolist()),
              "dimensions": {"macrogenes": macros.shape[1], "hidden_dim": args.hidden_dim, "model_dim": model_dim},
              "batch_size": args.batch_size, "seed": args.seed, "rtol": 1e-4, "atol": 1e-5,
              "jax_devices": [str(device) for device in jax.devices()], "reference_device": args.reference_device,
              "tf32": False, "reference_padded_batch": True,
              "xla_flags": os.environ.get("XLA_FLAGS", ""),
              "elapsed_seconds": time.monotonic() - started, "updates": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")
    print(f"{'Verified' if not failures else 'Failed'} {len(records)} matched updates; receipt: {args.output}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

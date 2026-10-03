#!/usr/bin/env python3
"""Bounded single-host pretrain smoke and fresh-process Orbax resume check."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["initial", "resume"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "gpu", "tpu"], default="tpu")
    parser.add_argument("--expected-local-device-count", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--mixed-precision", choices=["fp32", "bf16"])
    parser.add_argument("--project")
    parser.add_argument("--zone")
    parser.add_argument("--tpu-type")
    parser.add_argument("--chip-count", type=int)
    parser.add_argument("--usd-per-chip-hour", type=float)
    parser.add_argument("--quota-mode", choices=["on-demand", "spot", "flex-start"])
    args = parser.parse_args()
    if args.chip_count is not None and args.chip_count < 1:
        parser.error("--chip-count must be positive")
    if args.usd_per_chip_hour is not None and (not math.isfinite(args.usd_per_chip_hour) or args.usd_per_chip_hour <= 0):
        parser.error("--usd-per-chip-hour must be positive and finite")
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from flax.traverse_util import flatten_dict
    from jax_saturn.contracts.validation import validate_npz
    from jax_saturn.distributed.checkpoint import restore_checkpoint, save_checkpoint
    from jax_saturn.distributed.inference import pretrain_embeddings
    from jax_saturn.distributed.metric import replicate_state, unreplicate_state, validate_devices
    from jax_saturn.distributed.pretrain import make_pretrain_pmap
    from jax_saturn.models.precision import matmul_dtype, resolve_precision
    from jax_saturn.models.saturn import SaturnPretrainModule
    from jax_saturn.training.pretrain import create_state

    try:
        devices = jax.local_devices(backend=args.device)
        if len(devices) != args.expected_local_device_count:
            raise ValueError("Actual device count differs from expected count")
        devices = validate_devices(devices, batch_size=args.global_batch_size)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    precision = resolve_precision(args.mixed_precision, platform=args.device)
    names, counts = ("a", "b", "c"), (7, 5, 9)
    rng = np.random.default_rng(42)
    scores = rng.uniform(.1, 1, (sum(counts), 8)).astype(np.float32)
    proteins = rng.normal(size=(sum(counts), 8)).astype(np.float32)
    batch, inputs = [], []
    for code, genes in enumerate(counts):
        values = rng.poisson(3, (args.global_batch_size, genes)).astype(np.float32)
        valid = np.arange(args.global_batch_size) < max(args.global_batch_size - code - 1, 1)
        values[~valid] = 0
        batch.append({"values": jnp.asarray(values), "valid_mask": jnp.asarray(valid)})
        inputs.append(values[:min(genes, len(values))])
    source_hash = hashlib.sha256(scores.tobytes() + proteins.tobytes() + b"".join(value.tobytes() for value in inputs)).hexdigest()
    module = SaturnPretrainModule(names, counts, 8, 16, 8, dtype=matmul_dtype(precision))
    optimizer = optax.adam(.0005)
    template = create_state(module, scores, optimizer, 0)
    config = {"fixture": "synthetic-smoke-v1", "mixed_precision": precision,
              "global_batch_size": args.global_batch_size, "local_device_count": len(devices),
              "num_macrogenes": 8, "hidden_dim": 16, "embed_dim": 8, "dropout": module.dropout,
              "dtype": str(jnp.dtype(module.dtype))}

    def metadata(state):
        return {"schema_version": 1, "implementation": "jax", "model_kind": "pretrain",
                "epoch": int(state.epoch), "step": int(state.step), "seed": 0,
                "hyperparameters": config, "species_names": list(names),
                "input_shapes": {name: [args.global_batch_size, genes] for name, genes in zip(names, counts)},
                "source_manifest_sha256": source_hash}

    def arrays(state, metrics):
        output = {"params/" + "/".join(path): np.asarray(value)
                  for path, value in flatten_dict(state.params).items()}
        output.update({f"opt/{index}": np.asarray(value) for index, value in enumerate(jax.tree_util.tree_leaves(state.opt_state))})
        output.update({"rng": np.asarray(jax.random.key_data(state.rng)), "step": np.asarray(state.step),
                       "loss": np.asarray(metrics["loss"])})
        return output

    root = args.output_dir.resolve()
    expected_path = root / "expected_next_step.npz"
    report_path = root / f"{args.phase}_receipt.json"
    if report_path.exists() or (args.phase == "initial" and root.exists() and any(root.iterdir())):
        parser.error("Use a fresh output directory; smoke receipts and checkpoints are immutable")
    root.mkdir(parents=True, exist_ok=True)
    step = make_pretrain_pmap(module, optimizer, proteins, devices)
    started = time.monotonic()
    if args.phase == "initial":
        state, metrics = step(replicate_state(template, devices), tuple(batch))
        state = unreplicate_state(state).replace(epoch=jnp.int32(1))
        jax.block_until_ready(state.params)
        save_checkpoint(root / "checkpoints/epoch_0001", state, metadata(state), history=[{"epoch": 1, "loss": float(metrics["loss"])}])
        following, following_metrics = step(replicate_state(state, devices), tuple(batch))
        following_arrays = arrays(unreplicate_state(following), following_metrics)
        if not all(np.isfinite(value).all() for value in following_arrays.values()):
            raise FloatingPointError("Nonfinite expected next-step state")
        np.savez(expected_path, **following_arrays)
    else:
        state, _, _ = restore_checkpoint(root / "checkpoints/epoch_0001", template, expected_metadata=metadata(template))
        state, metrics = step(replicate_state(state, devices), tuple(batch))
        state = unreplicate_state(state).replace(epoch=jnp.int32(2))
        actual = arrays(state, metrics)
        with np.load(expected_path, allow_pickle=False) as expected:
            if set(actual) != set(expected.files):
                raise ValueError("Resume state arrays differ from the initial reference")
            for name, value in actual.items():
                np.testing.assert_allclose(value, expected[name], rtol=1e-4, atol=1e-5, err_msg=name)
    if not all(np.isfinite(value).all() for value in arrays(state, metrics).values()):
        raise FloatingPointError("Nonfinite smoke state or loss")
    embeddings, macros = pretrain_embeddings(module, state.params, tuple(inputs), batch_size=args.global_batch_size, devices=devices)
    species = np.concatenate([np.repeat(name, len(value)) for name, value in zip(names, inputs)])
    artifact = {"embeddings": embeddings, "macrogenes": macros, "species": species,
                "obs_ids": np.array([f"{name}_{index}" for name, value in zip(names, inputs) for index in range(len(value))])}
    validate_npz(artifact, "label_free")
    np.savez(root / f"{args.phase}_embeddings.npz", **artifact)
    elapsed = time.monotonic() - started
    report = {"success": True, "phase": args.phase, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "platform": args.device, "devices": [str(device) for device in devices],
              "device_kinds": [device.device_kind for device in devices], "local_device_count": len(devices),
              "global_device_count": jax.device_count(), "process_index": jax.process_index(), "process_count": jax.process_count(),
              "mixed_precision": precision, "global_batch_size": args.global_batch_size, "step": int(state.step),
              "loss": float(metrics["loss"]), "output_rows": len(species), "elapsed_seconds": elapsed,
              "project": args.project, "zone": args.zone, "tpu_type": args.tpu_type, "chip_count": args.chip_count,
              "usd_per_chip_hour": args.usd_per_chip_hour, "quota_mode": args.quota_mode,
              "compute_chip_hours": None if args.chip_count is None else args.chip_count * elapsed / 3600,
              "billing_note": "Compute timing excludes setup and idle VM time; it is not the billed lifetime."}
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

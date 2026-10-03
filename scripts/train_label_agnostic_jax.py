#!/usr/bin/env python3
"""Strict label-free JAX SATURN fine-tuning from a native pretrain checkpoint."""

import argparse
from dataclasses import fields
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from jax_saturn.training.label_agnostic import LabelConfig, train_label_free

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objective", required=True, choices=["infonce", "mmd", "ot"])
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--pretrain-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=["cpu", "gpu", "cuda", "tpu"], default="cpu")
    parser.add_argument("--device-num", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--distributed", action="store_true", help="Single-host pmap; --batch-size is global")
    parser.add_argument("--expected-local-device-count", type=int)
    parser.add_argument("--validation-profile", choices=["short", "full"], default="short")
    defaults = LabelConfig()
    for field in fields(defaults):
        value = getattr(defaults, field.name)
        if field.name == "mixed_precision":
            parser.add_argument("--mixed-precision", choices=["fp32", "bf16"],
                                help="Matmul dtype; defaults to bf16 on TPU, fp32 otherwise")
            continue
        parser.add_argument("--" + field.name.replace("_", "-"), type=type(value),
                            default=None if field.name == "epochs" else value)
    args = parser.parse_args()
    if args.epochs is None:
        args.epochs = 30 if args.validation_profile == "full" else 2
    import jax
    from jax_saturn.models.precision import resolve_precision
    args.mixed_precision = resolve_precision(args.mixed_precision, platform=args.device)

    try:
        devices = jax.devices("gpu" if args.device == "cuda" else args.device)
    except RuntimeError as error:
        parser.error(f"Requested devices unavailable: {error}")
    if not 0 <= args.device_num < len(devices):
        parser.error("--device-num is outside the available range")
    local_devices = jax.local_devices(backend="gpu" if args.device == "cuda" else args.device)
    if args.expected_local_device_count is not None and args.expected_local_device_count != len(local_devices):
        parser.error("Local device count differs from --expected-local-device-count")
    config = LabelConfig(**{field.name: getattr(args, field.name) for field in fields(defaults)})
    if args.distributed:
        from jax_saturn.distributed.metric import validate_devices
        try:
            validate_devices(local_devices, batch_size=config.batch_size)
        except ValueError as error:
            parser.error(str(error))
    import json
    print(json.dumps({"jax_devices": [str(device) for device in devices],
                      "local_device_count": len(local_devices), "global_device_count": len(devices),
                      "device_kinds": [device.device_kind for device in local_devices],
                      "process_index": jax.process_index(), "process_count": jax.process_count(),
                      "mixed_precision": config.mixed_precision,
                      "execution": "pmap" if args.distributed else "jit"}), flush=True)
    with jax.default_device(devices[args.device_num]):
        train_label_free(args.artifact, args.pretrain_checkpoint, args.output_dir,
                         objective=args.objective, config=config, resume=args.resume,
                         devices=local_devices if args.distributed else None)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Parallel SATURN JAX pretraining and labeled baseline metric learning."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in_data", default="data/human_monkey_mouse.csv")
    parser.add_argument("--work_dir", default="out/human_monkey_mouse_jax")
    parser.add_argument("--device", choices=["cpu", "gpu", "cuda", "tpu"], default="cpu")
    parser.add_argument("--device_num", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ref_label_col", default="cellType")
    parser.add_argument("--hv_genes", type=int, default=2000)
    parser.add_argument("--num_macrogenes", type=int, default=200)
    parser.add_argument("--model_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--validation-profile", choices=["short", "full"], default="short")
    parser.add_argument("--mixed-precision", choices=["fp32", "bf16"],
                        help="Matmul dtype; defaults to bf16 on TPU, fp32 otherwise")
    parser.add_argument("--pretrain-mixed-precision", choices=["fp32", "bf16"],
                        help="Override pretraining precision, e.g. to reuse an fp32 checkpoint with a bf16 baseline")
    parser.add_argument("--pretrain_epochs", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--pretrain_batch_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--pretrain_lr", type=float, default=.0005)
    parser.add_argument("--metric_lr", type=float, default=.001)
    parser.add_argument("--pretrain", choices=["true", "false"], default="true", type=str.lower)
    parser.add_argument("--pretrain_model_path", type=Path)
    parser.add_argument("--metric_model_path", type=Path)
    parser.add_argument("--polling_freq", type=int, default=5)
    parser.add_argument("--embedding_model", default="ESM1b")
    parser.add_argument("--centroid_score_func", choices=["default", "one_hot", "smoothed"], default="default")
    parser.add_argument("--centroids_init_path", type=Path)
    parser.add_argument("--pe_sim_penalty", type=float, default=.2)
    parser.add_argument("--l1_penalty", type=float, default=0.)
    parser.add_argument("--resume", type=Path, help="Specific completed Orbax epoch checkpoint")
    parser.add_argument("--metric-resume", type=Path, help="Completed labeled metric epoch checkpoint")
    parser.add_argument("--metric-distributed", action="store_true",
                        help="Run baseline pmap over local devices; --batch_size is the global batch size")
    parser.add_argument("--pretrain-distributed", action="store_true",
                        help="Run pretraining pmap over local devices; --pretrain_batch_size is global")
    parser.add_argument("--expected-local-device-count", type=int,
                        help="Fail before training if the selected platform's local device count differs")
    parser.add_argument("--pytorch-compat-pretrain-checkpoint", type=Path)
    parser.add_argument("--pytorch-compat-metric-checkpoint", type=Path)
    args = parser.parse_args()
    if args.pretrain_epochs is None:
        args.pretrain_epochs = 20 if args.validation_profile == "full" else 2
    if args.epochs is None:
        args.epochs = 30 if args.validation_profile == "full" else 2
    if args.epochs < 0:
        parser.error("--epochs must be nonnegative")
    if args.pretrain_epochs < 1:
        parser.error("--pretrain_epochs must be positive")
    if args.pretrain_model_path is not None and args.pretrain_model_path.suffix == ".pt":
        parser.error("--pretrain_model_path must be an Orbax directory; .pt state dicts are reference-only")
    if args.metric_model_path is not None and args.metric_model_path.suffix == ".pt":
        parser.error("--metric_model_path must be an Orbax directory")
    if args.pytorch_compat_metric_checkpoint is not None and not args.epochs:
        parser.error("Metric compatibility export requires metric training")
    if args.metric_resume is not None and not args.epochs:
        parser.error("--metric-resume requires metric training")
    if args.metric_distributed and not args.epochs:
        parser.error("--metric-distributed requires metric training")

    import jax
    from jax_saturn.data.preparation import prepare_centroids, prepare_data
    from jax_saturn.models.saturn import SaturnPretrainModule
    from jax_saturn.models.precision import matmul_dtype, resolve_precision
    from jax_saturn.training.pretrain import emit_pretrain_anndata, train_pretrain
    from jax_saturn.training.baseline import train_baseline
    from jax_saturn.distributed.checkpoint import latest_checkpoint

    checkpoint_dir = args.pretrain_model_path or Path(args.work_dir) / "shared/pretrain_orbax"
    if args.pretrain == "false" or args.resume is not None:
        args.resume = latest_checkpoint(args.resume or checkpoint_dir)
    if args.metric_resume is not None:
        args.metric_resume = latest_checkpoint(args.metric_resume)

    platform = "gpu" if args.device == "cuda" else args.device
    precision = resolve_precision(args.mixed_precision, platform=platform)
    pretrain_precision = resolve_precision(args.pretrain_mixed_precision or precision)
    try:
        devices = jax.devices(platform)
    except RuntimeError as error:
        parser.error(f"Requested {args.device} devices are unavailable: {error}")
    if not 0 <= args.device_num < len(devices):
        parser.error("--device_num is outside the available device range")
    local_devices = jax.local_devices(backend=platform)
    if args.expected_local_device_count is not None and args.expected_local_device_count != len(local_devices):
        parser.error("Local device count differs from --expected-local-device-count")
    if args.metric_distributed or args.pretrain_distributed:
        from jax_saturn.distributed.metric import validate_devices
        try:
            if args.metric_distributed:
                validate_devices(local_devices, batch_size=args.batch_size)
            if args.pretrain_distributed:
                validate_devices(local_devices, batch_size=args.pretrain_batch_size)
        except ValueError as error:
            parser.error(str(error))
    print(json.dumps({"jax_devices": [str(device) for device in devices],
                      "platform": platform, "implementation": "jax",
                      "global_device_count": len(devices), "local_device_count": len(local_devices),
                      "device_kinds": [device.device_kind for device in local_devices],
                      "process_index": jax.process_index(), "process_count": jax.process_count(),
                      "mixed_precision": precision,
                      "pretrain_mixed_precision": pretrain_precision,
                      "metric_execution": "pmap" if args.metric_distributed else "jit",
                      "pretrain_execution": "pmap" if args.pretrain_distributed else "jit"}), flush=True)
    work = Path(args.work_dir)
    data = prepare_data(args.in_data, cache_dir=work / "shared/embedding_cache",
                        hv_genes=args.hv_genes, ref_label_col=args.ref_label_col)
    centroid_path = args.centroids_init_path or work / "shared/centroids.npz"
    legacy = centroid_path if centroid_path.suffix == ".pkl" and centroid_path.exists() else None
    if centroid_path.suffix == ".pkl":
        centroid_path = centroid_path.with_suffix(".npz")
    centroids = prepare_centroids(data, centroid_path, seed=args.seed, hv_genes=args.hv_genes,
                                 num_macrogenes=args.num_macrogenes, score_func=args.centroid_score_func,
                                 legacy_path=legacy)
    model = SaturnPretrainModule(data.species_names, data.gene_counts, args.num_macrogenes,
                                args.hidden_dim, args.model_dim, dtype=matmul_dtype(pretrain_precision))
    epochs = args.pretrain_epochs
    if args.pretrain == "false":
        metadata = json.loads((args.resume / "metadata.json").read_text())
        epochs = metadata["epoch"]
    with jax.default_device(devices[args.device_num]):
        state, history = train_pretrain(model, data, centroids["scores"], output_dir=checkpoint_dir,
            epochs=epochs, batch_size=args.pretrain_batch_size, learning_rate=args.pretrain_lr,
            seed=args.seed, l1_penalty=args.l1_penalty, pe_sim_penalty=args.pe_sim_penalty, resume=args.resume,
            devices=local_devices if args.pretrain_distributed else None)
        result_dir = work / "saturn_results"
        pretrain_adata = emit_pretrain_anndata(model, state.params, data, result_dir / "adata_pretrain.h5ad",
                                             batch_size=args.pretrain_batch_size,
                                             devices=local_devices if args.pretrain_distributed else None)
        if args.epochs:
            metric_state, metric_history, _ = train_baseline(state.params, pretrain_adata,
                output_dir=result_dir, checkpoint_dir=args.metric_model_path or work / "shared/metric_orbax",
                epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.metric_lr,
                hidden_dim=args.hidden_dim, model_dim=args.model_dim, seed=args.seed, polling_freq=args.polling_freq,
                source_manifest_sha256=data.source_manifest_sha256, resume=args.metric_resume,
                devices=local_devices if args.metric_distributed else None, mixed_precision=precision)
        if args.pytorch_compat_pretrain_checkpoint is not None:
            from jax_saturn.models.conversion import export_pytorch_checkpoint
            export_pytorch_checkpoint(args.pytorch_compat_pretrain_checkpoint, state.params)
        if args.pytorch_compat_metric_checkpoint is not None:
            from jax_saturn.models.conversion import export_pytorch_checkpoint
            export_pytorch_checkpoint(args.pytorch_compat_metric_checkpoint, metric_state.params)
    result_dir.mkdir(parents=True, exist_ok=True)
    if history:
        import pandas as pd
        pd.DataFrame(history).to_csv(result_dir / "pretrain_losses.csv", index=False)
    checkpoint_path = checkpoint_dir / "checkpoints" / f"epoch_{int(state.epoch):04d}"
    if not checkpoint_path.exists() and args.resume is not None:
        checkpoint_path = args.resume
    summary = {"implementation": "jax", "stage": "pretrain", "epoch": int(state.epoch),
               "step": int(state.step), "orbax_checkpoint_path": str(checkpoint_path.resolve()),
               "source_manifest_sha256": data.source_manifest_sha256,
               "species_names": list(data.species_names)}
    (result_dir / "pretrain_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

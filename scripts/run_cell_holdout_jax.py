#!/usr/bin/env python3
"""Prepare, validate, train and report a strict cell holdout within one VM window."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('NUMBA_NUM_THREADS', '1')
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/macrogenes-numba-cache')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/macrogenes-matplotlib')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jax_saturn.data.holdout import write_json
from jax_saturn.experiments.cell_holdout import HoldoutConfig, TRIALS, prepare, preflight
from jax_saturn.progress import Progress
from run_hmmff_gpu import TimeBudgetError, parse_start, read_events, run_process


def effective_deadline(start, provider_deadline=None):
    ceiling = parse_start(start) + 21600
    return min(ceiling, parse_start(provider_deadline)) if provider_deadline else ceiling


def epoch_at(path):
    from jax_saturn.distributed.checkpoint import latest_checkpoint
    try:
        checkpoint = latest_checkpoint(path)
    except FileNotFoundError:
        return 0
    return int(json.loads((checkpoint / 'metadata.json').read_text())['epoch'])


def stage_paths(config):
    return {
        'pretrain': (config.foundation / 'shared/pretrain_orbax', config.foundation / 'shared/pretrain_orbax/execution.jsonl', config.pretrain_epochs),
        'baseline': (config.foundation / 'baseline/metric_orbax', config.foundation / 'baseline/saturn_results/execution.jsonl', config.metric_epochs),
        **{name: (config.run / name, config.run / name / 'execution.jsonl', config.metric_epochs) for name in config.trials[1:]},
    }


def foundation_complete(config, stage):
    """A committed last epoch alone does not prove its exports were published."""
    import pandas as pd
    path, _, target = stage_paths(config)[stage]
    if epoch_at(path) != target:
        return False
    results = config.foundation / 'baseline/saturn_results'
    summary = results / ('pretrain_summary.json' if stage == 'pretrain' else 'metric_summary.json')
    history = results / ('pretrain_losses.csv' if stage == 'pretrain' else 'metric_history.csv')
    representation = results / ('adata_pretrain.h5ad' if stage == 'pretrain' else 'final_adata.h5ad')
    if not all(p.is_file() for p in (summary, history, representation)):
        return False
    return (json.loads(summary.read_text())['epoch'] == target
            and pd.read_csv(history).epoch.tolist() == list(range(1, target + 1)))


def stage_baseline(config):
    """Publish the derived report copy atomically, preserving interrupted copies."""
    from jax_saturn.data.cache import file_sha256
    source = config.foundation / 'baseline/saturn_results'
    destination = config.run / 'baseline/saturn_results'
    staging = destination.with_name('saturn_results.staging')
    for candidate in (destination, staging):
        if not candidate.exists():
            continue
        if candidate == destination and all((candidate / p.relative_to(source)).is_file()
                and file_sha256(candidate / p.relative_to(source)) == file_sha256(p)
                for p in source.rglob('*') if p.is_file()):
            return
        preserved = config.run / 'baseline/interrupted' / f'{candidate.name}_{time.time_ns()}'
        preserved.parent.mkdir(parents=True, exist_ok=True)
        candidate.rename(preserved)
    staging.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, staging)
    staging.replace(destination)


def timing_gate(config, deadline, *, final_reserve=3600., unmeasured_reserve=1800.):
    stages = {}
    work = 0.
    for name, (path, event_path, target) in stage_paths(config).items():
        completed = epoch_at(path)
        if completed > target:
            raise ValueError(f'{name} checkpoint exceeds requested epochs')
        events = [e for e in read_events(event_path) if e['event'] == 'epoch_complete']
        steady = [max(0., e['seconds'] - (e.get('first_update_seconds', 0.) if e.get('invocation_first_epoch') else 0.))
                  for e in events if e['epoch'] >= 2]
        remaining = target - completed
        if not remaining:
            estimate = 0.
        elif steady:
            estimate = remaining * max(steady[-2:]) + max(e.get('first_update_seconds', 0.) for e in events)
        else:
            estimate = unmeasured_reserve
        work += estimate
        stages[name] = {'completed': completed, 'target': target, 'estimated_remaining_seconds': estimate,
                        'measured_steady_epoch_seconds': max(steady[-2:]) if steady else None,
                        'unmeasured_stage_reserve_seconds': unmeasured_reserve if remaining and not steady else 0.}
    available = deadline - time.time() - 300
    required = 1.3 * (work + 600 + final_reserve)
    record = {'stages': stages, 'safety_factor': 1.3, 'setup_and_compilation_reserve_seconds': 600,
              'final_evaluation_figures_backup_reserve_seconds': final_reserve, 'required_seconds': required,
              'available_seconds': available, 'passes': required <= available,
              'deadline_utc': datetime.fromtimestamp(deadline, timezone.utc).isoformat()}
    write_json(config.root / 'timing_gate.json', record)
    with (config.root / 'timing_gates.jsonl').open('a') as handle:
        handle.write(json.dumps(record) + '\n')
    if not record['passes']:
        raise TimeBudgetError('Remaining production work does not fit the single six-hour window; checkpoints retained')
    return record


def foundation_command(config, stage, target):
    pretrain_root, baseline_root = stage_paths(config)['pretrain'][0], stage_paths(config)['baseline'][0]
    command = [sys.executable, ROOT / 'scripts/train_saturn_jax.py',
        '--in_data', config.root / 'split/train.csv', '--work_dir', config.foundation / 'baseline',
        '--device', config.device, '--seed', str(config.seed), '--hv_genes', str(config.hv_genes),
        '--num_macrogenes', str(config.num_macrogenes), '--hidden_dim', str(config.hidden_dim), '--model_dim', str(config.model_dim),
        '--pretrain_batch_size', str(config.batch_size), '--batch_size', str(config.batch_size),
        '--mixed-precision', 'fp32', '--pretrain-mixed-precision', 'fp32', '--validation-profile', 'full',
        '--pretrain_model_path', pretrain_root, '--metric_model_path', baseline_root,
        '--centroids_init_path', config.foundation / 'shared/centroids.npz', '--preprocessing-out', config.preprocessing,
        '--pretrain_epochs', str(target if stage == 'pretrain' else config.pretrain_epochs),
        '--epochs', str(0 if stage == 'pretrain' else target), '--pretrain', 'true' if stage == 'pretrain' else 'false']
    if epoch_at(pretrain_root):
        command += ['--resume', pretrain_root]
    if stage == 'baseline' and epoch_at(baseline_root):
        command += ['--metric-resume', baseline_root]
    return command


def trial_command(config, objective, target):
    path = config.run / objective
    command = [sys.executable, ROOT / 'scripts/train_label_agnostic_jax.py', '--objective', objective,
        '--artifact', config.foundation / 'shared/label_free_artifact.npz',
        '--pretrain-checkpoint', config.foundation / f'shared/pretrain_orbax/checkpoints/epoch_{config.pretrain_epochs:04d}',
        '--output-dir', path, '--device', config.device, '--epochs', str(target), '--seed', str(config.seed),
        '--batch-size', str(config.batch_size), '--hidden-dim', str(config.hidden_dim), '--model-dim', str(config.model_dim),
        '--mixed-precision', 'fp32', '--neighbor-backend', config.neighbor_backend,
        '--query-block', str(config.query_block), '--reference-block', str(config.reference_block), '--cache-dir', config.root / 'cache']
    if epoch_at(path):
        command += ['--resume', path / f'checkpoints/epoch_{epoch_at(path):04d}']
    return command


def foundation_inventory(config):
    from jax_saturn.data.cache import file_sha256
    result = {}
    for path in sorted(config.foundation.rglob('*')):
        relative = path.relative_to(config.foundation)
        if (not path.is_file() or 'interrupted' in relative.parts
                or any(p.endswith(('.tmp', '.staging', '.orbax-checkpoint-tmp')) for p in relative.parts)):
            continue
        committed = True
        for index, part in enumerate(relative.parts):
            if index and relative.parts[index - 1] == 'checkpoints' and part.startswith('epoch_'):
                committed = (config.foundation.joinpath(*relative.parts[:index + 1]) / 'metadata.json').is_file()
        if committed:
            result[str(relative)] = file_sha256(path)
    return result


def verify_outputs(config):
    import anndata as ad
    import numpy as np
    import pandas as pd
    from jax_saturn.experiments.hmmff import trial_complete
    for name, (path, _, target) in stage_paths(config).items():
        if epoch_at(path) != target:
            raise ValueError(f'Incomplete production stage: {name}')
    for name in config.trials[1:]:
        if not trial_complete(config.run / name, name, epochs=config.metric_epochs):
            raise ValueError(f'Incomplete trial: {name}')
    if foundation_inventory(config) != json.loads((config.root / 'foundation_manifest.json').read_text()):
        raise ValueError('Training foundation changed during inference/reporting')
    for name in ('executed.ipynb', 'heldout_metrics.csv', 'holdout_report.json', 'report_assets/heldout_umaps.png',
                 'report_assets/heldout_umaps.pdf', 'report_assets/heldout_umaps.html', 'run/comparison.csv', 'run/acceptance.json'):
        if not (config.root / name).is_file():
            raise FileNotFoundError(config.root / name)
    split = pd.read_csv(config.root / 'split/split.csv', dtype=str, keep_default_na=False)
    for name in config.trials:
        history = pd.read_csv(config.run / name / 'history.csv')
        if history.epoch.tolist() != list(range(1, config.metric_epochs + 1)):
            raise ValueError(f'Incomplete epoch history: {name}')
        for population in ('train', 'test'):
            atlas = ad.read_h5ad(config.run / name / f'{population}_evaluated_adata.h5ad')
            expected = split.loc[split['split'] == population, 'obs_id'].to_numpy(str)
            if (not np.array_equal(np.asarray(atlas.obs_names, str), expected)
                    or atlas.shape != (len(expected), config.model_dim) or not np.isfinite(atlas.X).all()
                    or not np.isfinite(atlas.obsm['X_umap']).all()):
                raise ValueError(f'Invalid evaluated observations/embeddings: {name}/{population}')
    if os.environ.get('DISEASE_METADATA'):
        from disease_pilot.report import verify_disease_report
        verify_disease_report(config, ROOT / os.environ['DISEASE_METADATA'])
    return json.loads((config.run / 'acceptance.json').read_text())


def run(config, *, start, skip_notebook=False):
    from jax_saturn.experiments.hmmff import quarantine_uncommitted, trial_complete
    deadline = effective_deadline(start, os.environ.get('SATURN_VM_DEADLINE'))
    if not 0 <= time.time() - parse_start(start) < 21600:
        raise ValueError('VM start time must be in the past and less than six hours ago')
    env = {**config.execution_env(), 'PYTHONUNBUFFERED': '1'}
    config.root.mkdir(parents=True, exist_ok=True)
    with Progress(config.root / 'execution.jsonl') as progress:
        try:
            with progress.phase('preflight'):
                preflight(config, gpu=config.device in {'gpu', 'cuda'})
                manifest = ROOT / 'source_manifest.json'
                if manifest.exists():
                    from package_saturn_source import verify_tree
                    verify_tree(manifest)
            options_path = config.root / 'execution_options.json'
            options = {'vm_start_time': start, 'total_window_seconds': 21600,
                'deadline_utc': datetime.fromtimestamp(deadline, timezone.utc).isoformat(),
                'neighbor_backend': config.neighbor_backend, 'query_block': config.query_block,
                'reference_block': config.reference_block, 'device': config.device, 'restart_authorized': False}
            if options_path.exists() and json.loads(options_path.read_text()) != options:
                raise ValueError('Resume requires the original VM deadline and identical numerical/device policy')
            write_json(options_path, options)
            for stage, target in (('pretrain', config.pretrain_epochs), ('baseline', config.metric_epochs)):
                path = stage_paths(config)[stage][0]
                if epoch_at(path) > target:
                    raise ValueError(f'{stage} checkpoint exceeds the configured target')
                for stop in sorted(set([min(2, target), target])):
                    if epoch_at(path) >= stop and (stop < target or foundation_complete(config, stage)):
                        continue
                    quarantine_uncommitted(path)
                    run_process(foundation_command(config, stage, stop), env=env, deadline=deadline, progress=progress,
                                stage=f'{stage}_through_{stop}')
                    progress.event('timing_gate', **timing_gate(config, deadline))
            artifact = config.foundation / 'shared/label_free_artifact.npz'
            artifact_outputs = (artifact, config.foundation / 'shared/evaluation_triplets.npz',
                                config.foundation / 'shared/artifact_metadata.json')
            if not all(path.is_file() for path in artifact_outputs):
                if (config.root / 'foundation_manifest.json').exists():
                    raise ValueError('Frozen foundation artifacts are missing; restore the verified backup')
                run_process([sys.executable, ROOT / 'scripts/prepare_label_agnostic_artifacts.py',
                    '--pretrain-adata', config.foundation / 'baseline/saturn_results/adata_pretrain.h5ad',
                    '--artifact', artifact, '--triplets', config.foundation / 'shared/evaluation_triplets.npz',
                    '--metadata', config.foundation / 'shared/artifact_metadata.json', '--seed', str(config.seed),
                    '--batch-size', str(config.batch_size)], env=env, deadline=deadline, progress=progress, stage='training_only_artifacts')
            preflight(config)
            inventory = foundation_inventory(config)
            inventory_path = config.root / 'foundation_manifest.json'
            if inventory_path.exists() and json.loads(inventory_path.read_text()) != inventory:
                raise ValueError('Previously frozen foundation changed')
            write_json(inventory_path, inventory)
            stage_baseline(config)
            for objective in config.trials[1:]:
                if epoch_at(config.run / objective) < min(2, config.metric_epochs):
                    quarantine_uncommitted(config.run / objective)
                    run_process(trial_command(config, objective, min(2, config.metric_epochs)), env=env,
                                deadline=deadline, progress=progress, stage=f'{objective}_production_warmup')
                    progress.event('timing_gate', **timing_gate(config, deadline))
            for objective in config.trials[1:]:
                if not trial_complete(config.run / objective, objective, epochs=config.metric_epochs):
                    quarantine_uncommitted(config.run / objective)
                    run_process(trial_command(config, objective, config.metric_epochs), env=env,
                                deadline=deadline, progress=progress, stage=f'{objective}_through_{config.metric_epochs}')
            command = [sys.executable, ROOT / 'scripts/run_cell_holdout_jax.py', 'report']
            run_process(command, env=env, deadline=deadline, progress=progress, stage='heldout_inference_evaluation_figures')
            if not skip_notebook:
                report_env = {**env, 'SATURN_HOLDOUT_REPORT_ONLY': '1'}
                run_process([sys.executable, ROOT / 'scripts/execute_saturn_notebook.py', '--notebook',
                    ROOT / os.environ.get('REPORT_NOTEBOOK', 'notebooks/jax/human_monkey_mouse_frog_fish_holdout.ipynb'), '--output', config.root / 'executed.ipynb'],
                    env=report_env, deadline=deadline, progress=progress, stage='executed_notebook')
                acceptance = verify_outputs(config)
                write_json(config.root / 'execution_result.json', {'execution_success': True, 'scientific_acceptance': acceptance,
                    'backup_status': 'Final backup and local checksum verification required', 'six_hour_window_extended': False})
                progress.event('execution_success')
            return 0
        except TimeBudgetError as error:
            write_json(config.root / 'execution_result.json', {'execution_success': False, 'status': 'time_budget_stopped',
                'reason': str(error), 'checkpoints_preserved': True, 'restart_authorized': False})
            progress.event('time_budget_stopped', reason=str(error))
            return 3
        except BaseException as error:
            write_json(config.root / 'execution_result.json', {'execution_success': False, 'status': 'failed', 'reason': str(error)})
            raise


def generate_report(config):
    from jax_saturn.experiments.holdout_report import report
    preflight(config)
    if foundation_inventory(config) != json.loads((config.root / 'foundation_manifest.json').read_text()):
        raise ValueError('Frozen foundation differs from the recorded training state')
    import subprocess
    subprocess.run([sys.executable, ROOT / 'scripts/evaluate_label_agnostic_benchmark.py', '--root', config.run,
        '--truth-adata', config.foundation / 'baseline/saturn_results/adata_pretrain.h5ad',
        '--triplets', config.foundation / 'shared/evaluation_triplets.npz', '--seed', str(config.seed),
        '--neighbor-backend', config.neighbor_backend, '--neighbor-device', 'gpu' if config.device in {'gpu', 'cuda'} else 'cpu',
        '--query-block', str(config.query_block), '--reference-block', str(config.reference_block),
        '--cache-dir', config.root / 'cache', '--save-umap-model', '--trials', *config.trials], check=True, cwd=ROOT)
    report(config)
    if os.environ.get('DISEASE_METADATA'):
        from disease_pilot.report import disease_report
        disease_report(config, ROOT / os.environ['DISEASE_METADATA'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'preflight', 'run', 'report', 'diagnostics'])
    parser.add_argument('--vm-start-time', default=os.environ.get('SATURN_VM_START'))
    parser.add_argument('--skip-notebook', action='store_true', help='Used when training is launched inside the source notebook')
    args = parser.parse_args()
    config = HoldoutConfig.from_env()
    if args.command == 'prepare':
        split = prepare(config)
        print(split.groupby(['species', 'split']).size().unstack().to_string())
    elif args.command == 'preflight':
        print(json.dumps(preflight(config, gpu=config.device in {'gpu', 'cuda'}), indent=2))
    elif args.command == 'report':
        generate_report(config)
    elif args.command == 'diagnostics':
        from jax_saturn.experiments.holdout_diagnostics import diagnostics
        diagnostics(config)
    else:
        if not args.vm_start_time:
            parser.error('run requires --vm-start-time (the initial GCE start timestamp)')
        return run(config, start=args.vm_start_time, skip_notebook=args.skip_notebook)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

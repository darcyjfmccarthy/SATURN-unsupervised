#!/usr/bin/env python3
"""Reuse the completed foundation; warm up with real epochs, gate, finish and report."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jax_saturn.experiments.hmmff import (FOUNDATION, OUTPUT, OBJECTIVES, PRETRAIN, atomic_json,
    committed_epoch, inventory, stage_foundation, trial_complete, validate_foundation, validate_trial_resume,
    quarantine_uncommitted, validate_report_inputs)
from jax_saturn.progress import Progress

STOP_COMMAND = 'gcloud compute instances stop saturn-a100-pilot --project=macrogenes --zone=asia-southeast1-a'


class TimeBudgetError(RuntimeError):
    pass


def parse_start(value):
    stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if stamp.tzinfo is None:
        raise ValueError('--vm-start-time must include a timezone')
    return stamp.timestamp()


def cleanup_process(process):
    import psutil
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    for child in reversed(children):
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    _, alive = psutil.wait_procs(children, timeout=2)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass


def run_process(command, *, env, deadline, progress, stage):
    if time.time() >= deadline - 300:
        raise TimeBudgetError('Less than five minutes remain for checkpoint backup')
    with progress.phase(stage):
        # Inherit stdout/stderr; notebook child separately forwards its kernel.
        process = subprocess.Popen([str(p) for p in command], cwd=ROOT, env=env, start_new_session=True)
        try:
            while process.poll() is None:
                if time.time() >= deadline - 300:
                    raise TimeBudgetError('Reached the five-minute backup reserve before the six-hour ceiling')
                time.sleep(.2)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
        except BaseException:
            cleanup_process(process)
            raise


def read_events(path):
    if not path.exists():
        return []
    events = []
    for line in path.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            # A terminated append may leave just its last record incomplete.
            continue
    return events


def remaining_estimate(output, *, target=30, setup_reserve=600., final_reserve=3600.):
    """Epoch 2 is steady work. First-update compilation is reserved separately."""
    details = {}
    compilation = 0.
    for objective in OBJECTIVES:
        trial = Path(output) / objective
        completed = committed_epoch(trial)
        events = read_events(trial / 'execution.jsonl')
        timing = [e for e in events if e['event'] == 'epoch_complete' and e['epoch'] >= 2]
        if completed < target and not timing:
            raise TimeBudgetError(f'No production steady-epoch timing for {objective}; refusing an unmeasured projection')
        # A restarted process may compile on epoch 2; its first update is
        # measured separately. The reserve pays that compilation again.
        seconds = max((e['seconds'] - (e.get('first_update_seconds', 0.) if e.get('invocation_first_epoch',False) else 0.)
                       for e in timing[-2:]), default=0.)
        compile_bound = max((e.get('first_update_seconds', 0.) for e in events if e['event'] == 'epoch_complete'), default=0.)
        search_compilation = {}
        for event in events:
            if event['event'] == 'neighbor_compilation' and event['dimension'] == 384 and event['k'] in (15,50):
                key=(event['k'],event['query_block'],event['reference_block'])
                search_compilation[key]=max(search_compilation.get(key,0.),event['seconds'])
        search_compile_seconds=sum(search_compilation.values())
        if completed < target:
            compilation += compile_bound + search_compile_seconds
        details[objective] = {'completed_epochs':completed, 'remaining_epochs':max(0,target-completed),
                              'steady_epoch_seconds':seconds, 'first_update_compilation_bound_seconds':compile_bound,
                              'neighbor_compilation_seconds':search_compile_seconds}
    epoch_work = sum(d['remaining_epochs'] * d['steady_epoch_seconds'] for d in details.values())
    total = setup_reserve + compilation + epoch_work + final_reserve
    return {'schema_version':1, 'objectives':details, 'remaining_setup_seconds':setup_reserve,
            'remaining_compilation_seconds':compilation, 'remaining_epoch_seconds':epoch_work,
            'final_evaluation_figures_backup_seconds':final_reserve, 'safety_factor':1.3,
            'required_seconds':1.3 * total}


def training_command(args, objective, epochs):
    output = args.output / objective
    command = [sys.executable, ROOT / 'scripts/train_label_agnostic_jax.py', '--objective',objective,
        '--artifact',args.output / 'shared/label_free_artifact.npz', '--pretrain-checkpoint',args.output / PRETRAIN,
        '--output-dir',output, '--device','gpu', '--mixed-precision','fp32', '--seed','0', '--batch-size','512',
        '--hidden-dim','384', '--model-dim','384', '--epochs',str(epochs),
        '--neighbor-backend',args.backend, '--query-block',str(args.query_block),
        '--reference-block',str(args.reference_block), '--cache-dir',args.cache]
    epoch = committed_epoch(output)
    if epoch:
        command += ['--resume', output / f'checkpoints/epoch_{epoch:04d}']
    return command


def verify_outputs(output):
    import anndata as ad
    from jax_saturn.contracts.validation import validate_anndata
    for objective in OBJECTIVES:
        if not trial_complete(output / objective, objective):
            raise ValueError(f'{objective} does not have a validated 30-epoch result')
    required = ['executed.ipynb','comparison.csv','acceptance.json']
    required += [f'{t}/{name}' for t in ('baseline',*OBJECTIVES) for name in ('evaluated_adata.h5ad','umap.npz','metrics.json','history.csv')]
    for name in required:
        if not (output / name).is_file():
            raise FileNotFoundError(output / name)
    for stem in ('pipeline_schematic','pretrained_umap','training_selection_trajectories','loss_histories',
                 'final_metric_dashboard','mixing_metrics','trial_umaps','local_neighbor_diagnostics'):
        for suffix in ('png','pdf'):
            if not (output/'report_assets'/f'{stem}.{suffix}').is_file():
                raise FileNotFoundError(output/'report_assets'/f'{stem}.{suffix}')
    for trial in ('baseline',*OBJECTIVES):
        atlas = ad.read_h5ad(output / trial / 'evaluated_adata.h5ad')
        validate_anndata(atlas, evaluated=True)
        if atlas.shape != (148727,384):
            raise ValueError(f'Evaluated representation differs: {trial}')
    return json.loads((output / 'acceptance.json').read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preflight','run'])
    parser.add_argument('--foundation', type=Path, default=ROOT / FOUNDATION)
    parser.add_argument('--output', type=Path, default=ROOT / OUTPUT)
    parser.add_argument('--backend', choices=['jax','sklearn'], default='jax')
    parser.add_argument('--query-block', type=int, default=1024)
    parser.add_argument('--reference-block', type=int, default=8192)
    parser.add_argument('--cache', type=Path, default=ROOT / 'out/hmmff_gpu_work_cache')
    parser.add_argument('--vm-start-time', help='Actual lastStartTimestamp from GCE; ceiling is exactly six hours later')
    parser.add_argument('--setup-reserve-seconds', type=float, default=600.)
    parser.add_argument('--final-reserve-seconds', type=float, default=3600., help='Evaluation, figures and backup estimate; recorded, never hidden')
    args = parser.parse_args()
    args.foundation, args.output, args.cache = [p.resolve() for p in (args.foundation,args.output,args.cache)]
    if min(args.setup_reserve_seconds,args.final_reserve_seconds) < 0 or args.query_block < 1 or args.reference_block < 1:
        parser.error('Blocks must be positive; timing reserves nonnegative')
    if args.command == 'run' and not args.vm_start_time:
        parser.error('run requires the actual --vm-start-time to count deployment and warm-up against six hours')
    deadline = parse_start(args.vm_start_time) + 21600 if args.vm_start_time else float('inf')
    if args.command == 'run' and not 0 <= time.time() - (deadline - 21600) < 21600:
        parser.error('VM start time must be in the past and less than six hours ago')
    from jax_saturn.data.neighbors import NeighborSearch
    search = NeighborSearch(args.backend,args.query_block,args.reference_block,'gpu')
    with Progress(args.output / 'execution.jsonl') as progress:
        try:
            with progress.phase('foundation_validation'):
                record = validate_foundation(args.foundation)
                inputs = validate_report_inputs(ROOT)
                search.validate_device()
                # Even a sklearn search backend must train on the requested GPU.
                import jax
                if jax.__version__ != '0.6.2':
                    raise ValueError('This experiment requires JAX 0.6.2')
                jax.devices('gpu')
                from jax_saturn.distributed.checkpoint import load_pretrain_metric_params
                load_pretrain_metric_params(args.foundation / PRETRAIN,input_dim=300,hidden_dim=384,model_dim=384)
            for objective in OBJECTIVES:
                epoch = committed_epoch(args.output / objective)
                if epoch > 30:
                    raise ValueError(f'{objective} checkpoint exceeds 30 epochs')
                if epoch:
                    validate_trial_resume(args.output / objective,objective,args.foundation)
                    metadata = json.loads((args.output / objective / f'checkpoints/epoch_{epoch:04d}/metadata.json').read_text())
                    from jax_saturn.data.cache import file_sha256
                    from jax_saturn.distributed.checkpoint import directory_sha256
                    hp = metadata['hyperparameters']
                    if hp['label_free_artifact_sha256'] != file_sha256(args.foundation / 'shared/label_free_artifact.npz') or hp['pretrain_checkpoint_sha256'] != directory_sha256(args.foundation / PRETRAIN):
                        raise ValueError(f'{objective} checkpoint uses another foundation')
                    previous = [e for e in read_events(args.output / objective / 'execution.jsonl') if e['event']=='execution_identity']
                    if previous and previous[-1]['neighbor_search'] != (search.identity() if args.backend=='jax' else {'backend':'sklearn','policy':'legacy-drop-first'}):
                        raise ValueError('Resume requires the same numerical/backend policy; start a new trial for a backend change')
            atomic_json(args.output / 'preflight.json',record)
            previous_inputs=args.output/'preflight_inputs.json'
            if previous_inputs.exists() and json.loads(previous_inputs.read_text())!=inputs:
                raise ValueError('Raw report/input contents changed since this experiment was staged')
            atomic_json(previous_inputs,inputs)
            progress.event('preflight_passed',cells=record['cells'],species=record['species'],architecture=record['architecture'],
                           completed_training_stages_skipped=['pretrain','baseline'])
            if args.command == 'preflight':
                return 0
            with progress.phase('stage_foundation'):
                stage_foundation(args.foundation,args.output,record)
            import importlib.metadata
            from jax_saturn.data.work_cache import canonical
            source_manifest = ROOT / 'source_manifest.json'
            if source_manifest.exists():
                # Refuse an archive whose installed files have changed silently.
                from package_saturn_source import verify_tree
                verify_tree(source_manifest)
                source_identity=json.loads(source_manifest.read_text())
            else:
                from package_saturn_source import manifest
                source_identity=manifest()
            environment = {d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata.get('Name')}
            execution_record={'neighbor_search':search.identity(), 'cache':str(args.cache),
                'source_manifest':source_identity,
                'environment':environment, 'command':sys.argv, 'vm_start_time':args.vm_start_time,
                'deadline_utc':datetime.fromtimestamp(deadline,timezone.utc).isoformat()}
            atomic_json(args.output/'execution_records'/f'{time.time_ns()}.json',execution_record)
            atomic_json(args.output / 'execution_options.json',execution_record)
            atomic_json(args.output/'experiment_record.json',{'question':'Compare label-free InfoNCE, MMD and OT with the completed labelled baseline',
                'changed_variables':['label-free objective'],'foundation':record,
                'scientific_configuration':{'architecture':[300,384,384],'seed':0,'batch_size':512,'precision':'fp32','epochs_per_objective':30},
                'input_record':json.loads((args.output/'input_record.json').read_text()) if (args.output/'input_record.json').exists() else None,
                'current_input_hashes':inputs,
                'historical_input_hashes':'Unavailable at original fit time; current input hashes do not establish historical input identity',
                'execution_options':'execution_options.json','timings':'execution.jsonl','runtime_gate':'timing_gate.json',
                'results':'execution_result.json','record_to_complete':'EXPERIMENT_RECORD.md'})
            env = os.environ.copy()
            env.update(PYTHONUNBUFFERED='1',NUMBA_NUM_THREADS='1',XLA_PYTHON_CLIENT_PREALLOCATE='false',
                HMMFF_FOUNDATION=str(args.foundation),HMMFF_OUTPUT_DIR=str(args.output),HMMFF_REUSE_FOUNDATION='1',
                HMMFF_NEIGHBOR_BACKEND=args.backend,HMMFF_QUERY_BLOCK=str(args.query_block),
                HMMFF_REFERENCE_BLOCK=str(args.reference_block),HMMFF_CACHE_DIR=str(args.cache))
            for objective in OBJECTIVES:
                if committed_epoch(args.output / objective) < 2:
                    for path in quarantine_uncommitted(args.output/objective):
                        progress.event('interrupted_write_preserved',path=path)
                    run_process(training_command(args,objective,2),env=env,deadline=deadline,progress=progress,stage=f'{objective}_production_epochs_1_2')
            estimate = remaining_estimate(args.output,setup_reserve=args.setup_reserve_seconds,final_reserve=args.final_reserve_seconds)
            estimate['time_remaining_seconds'] = deadline - time.time()
            estimate['passes'] = estimate['required_seconds'] <= estimate['time_remaining_seconds']
            atomic_json(args.output / 'timing_gate.json',estimate)
            progress.event('timing_gate',**estimate)
            if not estimate['passes']:
                raise TimeBudgetError('1.3 × remaining work exceeds time before automatic shutdown; checkpoints retained')
            for objective in OBJECTIVES:
                if not trial_complete(args.output / objective,objective):
                    for path in quarantine_uncommitted(args.output/objective):
                        progress.event('interrupted_write_preserved',path=path)
                    run_process(training_command(args,objective,30),env=env,deadline=deadline,progress=progress,stage=f'{objective}_production_through_30')
            run_process([sys.executable,ROOT/'scripts/execute_saturn_notebook.py',
                '--notebook',ROOT/'notebooks/jax/human_monkey_mouse_frog_fish.ipynb', '--output',args.output/'executed.ipynb'],
                env=env,deadline=deadline,progress=progress,stage='evaluation_and_notebook_report')
            with progress.phase('final_verification'):
                acceptance = verify_outputs(args.output)
                if inventory(args.foundation) != record['files']:
                    raise ValueError('Original foundation hashes changed')
            atomic_json(args.output/'execution_result.json',{'execution_success':True,'scientific_acceptance':acceptance,
                'foundation_hashes_unchanged':True,'backup_status':'User must verify final bucket backup and local download'})
            progress.event('execution_success',scientific_acceptance=acceptance)
            print('Verify the final bucket backup and local download, then run:\n'+STOP_COMMAND,flush=True)
            return 0
        except TimeBudgetError as error:
            atomic_json(args.output/'execution_result.json',{'execution_success':False,'status':'time_gate_stopped','reason':str(error)})
            print(str(error)+'\nProduction checkpoints are preserved. Back up, then run:\n'+STOP_COMMAND,flush=True)
            return 3


if __name__ == '__main__':
    raise SystemExit(main())

"""Immutable five-species foundation checks and production completion checks."""

import json
import hashlib
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from jax_saturn.contracts.validation import load_json, load_npz, validate_metric_history
from jax_saturn.data.cache import file_sha256

SPECIES = ['d_rerio', 'h_sapiens', 'm_murinus', 'm_musculus', 'x_laevis']
OBJECTIVES = ('infonce', 'mmd', 'ot')
PRETRAIN = 'shared/pretrain_orbax/checkpoints/epoch_0020'
BASELINE = 'baseline/metric_orbax/checkpoints/epoch_0030'
OUTPUT = 'out/human_monkey_mouse_frog_fish_jax_gpu_exact'
FOUNDATION = 'out/human_monkey_mouse_frog_fish_jax_benchmark_walkthrough'


def inventory(root):
    root = Path(root)
    paths = []
    for folder in (PRETRAIN, BASELINE):
        checkpoint = root / folder
        if not checkpoint.is_dir():
            raise FileNotFoundError(checkpoint)
        for component in ('params', 'opt_state', 'rng', 'training_state'):
            if not (checkpoint / component).is_dir() or not any((checkpoint / component).rglob('*')):
                raise ValueError(f'Missing Orbax component: {checkpoint / component}')
        paths.extend(p for p in checkpoint.rglob('*') if p.is_file())
    for name in ('shared/label_free_artifact.npz', 'shared/evaluation_triplets.npz', 'shared/artifact_metadata.json',
                 'baseline/saturn_results/adata_pretrain.h5ad', 'baseline/saturn_results/final_adata.h5ad',
                 'baseline/saturn_results/metric_history.csv', 'baseline/saturn_results/metric_summary.json',
                 'baseline/saturn_results/pretrain_losses.csv', 'baseline/saturn_results/pretrain_summary.json'):
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path)
    return [{'path': p.relative_to(root).as_posix(), 'size': p.stat().st_size, 'sha256': file_sha256(p)} for p in sorted(paths)]


def validate_foundation(root):
    """Fail before any training; no trainer is imported or called here."""
    import anndata as ad
    root = Path(root).resolve()
    records = inventory(root)
    saved = root / 'preservation_manifest.json'
    if saved.exists():
        previous = {row['path']: row for row in json.loads(saved.read_text())['files']}
        for row in records:
            if row != previous.get(row['path']):
                raise ValueError(f'Foundation checksum changed: {row["path"]}')
    pretrain = load_json(root / PRETRAIN / 'metadata.json', 'checkpoint')
    baseline = load_json(root / BASELINE / 'metadata.json', 'checkpoint')
    for metadata, kind, epoch in ((pretrain, 'pretrain', 20), (baseline, 'baseline_metric', 30)):
        hp = metadata['hyperparameters']
        if (metadata['model_kind'] != kind or metadata['epoch'] != epoch or metadata['seed'] != 0
                or metadata['species_names'] != SPECIES or hp['batch_size'] != 512
                or hp['hidden_dim'] != 384 or hp['embed_dim'] != 384
                or hp.get('dtype', 'float32') != 'float32' or hp.get('mixed_precision','fp32') != 'fp32'):
            raise ValueError(f'Incompatible completed {kind} checkpoint')
    if pretrain['hyperparameters']['num_macrogenes'] != 300:
        raise ValueError('Foundation must have 300 macrogenes')
    if pretrain['source_manifest_sha256'] != baseline['source_manifest_sha256']:
        raise ValueError('Pretrain/baseline source identities differ')
    for filename, metadata in (('pretrain_summary.json',pretrain),('metric_summary.json',baseline)):
        summary=json.loads((root/'baseline/saturn_results'/filename).read_text())
        if any(summary.get(key)!=metadata[key] for key in ('epoch','step','implementation','source_manifest_sha256')):
            raise ValueError(f'Foundation summary/checkpoint differ: {filename}')
    history = pd.read_csv(root / 'baseline/saturn_results/metric_history.csv')
    validate_metric_history(history, epochs=30)
    pretrain_history = pd.read_csv(root / 'baseline/saturn_results/pretrain_losses.csv')
    if not np.array_equal(pretrain_history['epoch'].to_numpy(), np.arange(1, 21)):
        raise ValueError('Pretraining history must contain epochs 1–20')
    artifact = load_npz(root / 'shared/label_free_artifact.npz', 'label_free')
    triplets = load_npz(root / 'shared/evaluation_triplets.npz', 'evaluation_triplets')
    if artifact['embeddings'].shape != (148727, 384) or artifact['macrogenes'].shape != (148727, 300):
        raise ValueError('Foundation must preserve all 148727 cells and the approved dimensions')
    if np.unique(artifact['species']).tolist() != SPECIES or not np.array_equal(artifact['obs_ids'], triplets['obs_ids']):
        raise ValueError('Artifact and triplet observations/species differ')
    if len(triplets['anchor']) != 746974:
        raise ValueError('Frozen evaluation triplets differ from the completed run')
    artifact_metadata = json.loads((root / 'shared/artifact_metadata.json').read_text())
    for key, expected in {'cell_count':148727, 'embedding_dimension':384, 'macrogene_dimension':300,
                          'species':SPECIES, 'seed':0, 'batch_size':512, 'evaluation_triplet_count':746974}.items():
        if artifact_metadata.get(key) != expected:
            raise ValueError(f'Artifact metadata differs: {key}')
    for filename in ('adata_pretrain.h5ad', 'final_adata.h5ad'):
        atlas = ad.read_h5ad(root / 'baseline/saturn_results' / filename)
        if (atlas.shape != (148727, 384) or not np.array_equal(atlas.obs_names.astype(str), artifact['obs_ids'])
                or not np.array_equal(atlas.obs['species'].astype(str), artifact['species'])
                or atlas.obsm['macrogenes'].shape != (148727, 300)):
            raise ValueError(f'Foundation representation identity differs: {filename}')
        if filename == 'adata_pretrain.h5ad':
            if not np.array_equal(atlas.X, artifact['embeddings']) or not np.array_equal(atlas.obsm['macrogenes'], artifact['macrogenes']):
                raise ValueError('Label-free teacher differs from the pretrained representation')
            fingerprint=hashlib.sha256(np.asarray(atlas.obsm['macrogenes'],dtype='<f4').tobytes())
            fingerprint.update(json.dumps({'obs_ids':list(atlas.obs_names),'species':list(map(str,atlas.obs['species'])),
                'labels':list(map(str,atlas.obs['labels'])),'refs':list(map(str,atlas.obs['ref_labels']))},
                sort_keys=True,separators=(',',':')).encode())
            if fingerprint.hexdigest()!=baseline['hyperparameters']['pretrain_artifact_sha256']:
                raise ValueError('Baseline checkpoint was trained on another pretrained artifact')
        if not np.isfinite(atlas.X).all():
            raise ValueError(f'Nonfinite foundation: {filename}')
        del atlas
    return {'schema_version':1, 'foundation':str(root), 'files':records,
            'species':SPECIES, 'cells':148727, 'architecture':[300,384,384],
            'source_manifest_sha256':pretrain['source_manifest_sha256'],
            'obs_ids_sha256':__import__('hashlib').sha256(artifact['obs_ids'].tobytes()).hexdigest()}


def validate_report_inputs(repo_root):
    """Report/raw-input dependencies must fail before any expensive training."""
    import csv
    import anndata as ad
    root=Path(repo_root)
    manifest=root/'data/human_monkey_mouse_frog_fish.csv'
    rows=list(csv.DictReader(manifest.open()))
    if len(rows)!=5 or sorted(row['species'] for row in rows)!=SPECIES:
        raise ValueError('Report manifest must contain exactly the five approved species')
    records=[];cells=0
    for row in rows:
        for key in ('path','embedding_path'):
            relative=Path(row[key])
            if relative.is_absolute() or '..' in relative.parts or relative.parts[0]!='data':
                raise ValueError(f'Unsafe report input path: {relative}')
            path=root/relative
            records.append({'species':row['species'],'role':key,'path':relative.as_posix(),
                'size':path.stat().st_size,'sha256':file_sha256(path)})
        atlas=ad.read_h5ad(root/row['path'],backed='r')
        if row['in_label_col'] not in atlas.obs:
            raise ValueError(f'Missing report label column: {row["species"]}')
        cells+=atlas.n_obs
        atlas.file.close()
    if cells!=148727:raise ValueError('Report inputs must retain all 148727 original cells')
    return {'manifest_sha256':file_sha256(manifest),'files':records,'cells':cells,
            'scope':'Current raw-input contents; historical fit-time input hashes were not captured'}


def stage_foundation(foundation, output, record):
    foundation, output = Path(foundation).resolve(), Path(output).resolve()
    if output == foundation or foundation in output.parents or output in foundation.parents:
        raise ValueError('Output and foundation must be distinct, non-nested directories')
    output.mkdir(parents=True, exist_ok=True)
    provenance = output / 'foundation.json'
    if provenance.exists() and json.loads(provenance.read_text()) != record:
        raise ValueError('Output belongs to a different or changed foundation')
    for row in record['files']:
        source, destination = foundation / row['path'], output / row['path']
        if destination.exists():
            if file_sha256(destination) != row['sha256']:
                raise ValueError(f'Staged foundation changed: {destination}')
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Actual copies keep backups self-contained and avoid accidental
            # writes through a baseline/shared directory symlink.
            temporary = destination.with_name(destination.name + '.staging')
            shutil.copy2(source, temporary)
            if temporary.stat().st_size!=row['size'] or file_sha256(temporary)!=row['sha256']:
                raise ValueError(f'Staged copy checksum differs: {temporary}')
            temporary.replace(destination)
    atomic_json(provenance, record)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def committed_epoch(trial):
    """A checkpoint's commit marker, never the presence of exported embeddings."""
    trial = Path(trial)
    epochs = []
    for path in (trial / 'checkpoints').glob('epoch_*'):
        if not (path / 'metadata.json').is_file():
            continue  # interrupted writes have no published metadata
        metadata = load_json(path / 'metadata.json', 'checkpoint')
        if path.name != f'epoch_{metadata["epoch"]:04d}' or metadata['model_kind'] != 'label_agnostic':
            raise ValueError(f'Invalid trial checkpoint: {path}')
        for component in ('params', 'best_params', 'opt_state', 'rng', 'training_state'):
            if not (path / component).is_dir():
                raise ValueError(f'Incomplete committed trial checkpoint: {path}')
        epochs.append(metadata['epoch'])
    return max(epochs, default=0)


def trial_complete(trial, objective, *, epochs=30):
    trial = Path(trial)
    if committed_epoch(trial) != epochs:
        return False
    required = ['final_embeddings.npz', 'run_summary.json', 'metric_history.csv', 'completion.json']
    if not all((trial / name).is_file() for name in required):
        return False
    completion=json.loads((trial/'completion.json').read_text())
    if completion['epochs']!=epochs:
        return False
    summary = load_json(trial / 'run_summary.json', 'run_summary')
    if summary['epochs'] != epochs or summary['objective'] != objective or summary['selection_uses_labels']:
        return False
    if completion['selected_epoch']!=summary['selected_epoch']:
        return False
    for name in ('final_embeddings.npz','run_summary.json','metric_history.csv'):
        if file_sha256(trial/name)!=completion['files'][name]:
            raise ValueError(f'Completed trial checksum differs: {trial/name}')
    validate_metric_history(pd.read_csv(trial / 'metric_history.csv'), epochs=epochs)
    checkpoint = load_json(trial / f'checkpoints/epoch_{epochs:04d}/metadata.json', 'checkpoint')
    hp = checkpoint['hyperparameters']
    if (hp['objective'] != objective or hp['label_free_artifact_sha256'] != summary['label_free_artifact_sha256']
            or hp['pretrain_checkpoint_sha256'] != summary['pretrain_checkpoint_sha256']):
        raise ValueError('Summary/checkpoint identities differ')
    final = load_npz(trial / 'final_embeddings.npz', 'final_embeddings')
    if final['embeddings'].shape[1] != summary['configuration']['model_dim']:
        raise ValueError('Final embedding dimension differs from summary')
    selected = Path(summary['orbax_checkpoint_path'])
    # Archived outputs may move; selected role/counters are verified locally.
    selected = trial / 'selected_orbax' / selected.name
    metadata = load_json(selected / 'metadata.json', 'checkpoint')
    if metadata['epoch'] != epochs or metadata['hyperparameters']['checkpoint_role'] != 'selected':
        raise ValueError('Missing matching selected checkpoint')
    from jax_saturn.distributed.checkpoint import directory_sha256
    if directory_sha256(selected)!=completion['selected_checkpoint_sha256'] or directory_sha256(trial/f'checkpoints/epoch_{epochs:04d}')!=completion['epoch_checkpoint_sha256']:
        raise ValueError('Completed trial checkpoint checksum differs')
    training=json.loads((trial/f'checkpoints/epoch_{epochs:04d}/training_state/metadata').read_text())
    if training['best_epoch']!=summary['selected_epoch'] or training['best_mixing']!=summary['selected_species_mixing_fraction'] or training['best_recall']!=summary['selected_teacher_top15_recall_at_50']:
        raise ValueError('Summary selected state differs from saved training state')
    return True


def quarantine_uncommitted(trial):
    """Keep interrupted writes for diagnosis and free the next epoch's path."""
    import time
    trial=Path(trial)
    moved=[]
    for group in ('checkpoints','selected_orbax'):
        for path in (trial/group).glob('epoch_*'):
            if path.is_dir() and not (path/'metadata.json').is_file():
                target=trial/'interrupted'/str(time.time_ns())/group/path.name
                target.parent.mkdir(parents=True,exist_ok=True)
                path.rename(target)
                moved.append(str(target))
    return moved


def validate_trial_resume(trial, objective, foundation):
    """Restore every state component during preflight, before any objective trains."""
    from dataclasses import asdict
    import jax
    import jax.numpy as jnp
    import optax
    from jax_saturn.training.label_agnostic import LabelConfig,LabelAgnosticTrainState
    from jax_saturn.distributed.checkpoint import directory_sha256,load_pretrain_metric_params,restore_checkpoint
    epoch=committed_epoch(trial)
    if not epoch:return
    trial,foundation=Path(trial),Path(foundation)
    path=trial/f'checkpoints/epoch_{epoch:04d}'
    params,source=load_pretrain_metric_params(foundation/PRETRAIN,input_dim=300,hidden_dim=384,model_dim=384)
    config=LabelConfig(epochs=30,hidden_dim=384,model_dim=384,batch_size=512,seed=0,mixed_precision='fp32')
    hp=asdict(config);hp.pop('epochs');hp.pop('mixed_precision')
    hp.update(objective=objective,label_free_artifact_sha256=file_sha256(foundation/'shared/label_free_artifact.npz'),
        pretrain_checkpoint_sha256=directory_sha256(foundation/PRETRAIN),checkpoint_role='epoch')
    expected={'model_kind':'label_agnostic','seed':0,'hyperparameters':hp,'species_names':SPECIES,
        'input_shapes':{'macrogenes':[512,300]},'source_manifest_sha256':source['source_manifest_sha256']}
    optimizer=optax.adam(config.learning_rate,eps=1e-8)
    template=LabelAgnosticTrainState(params,optimizer.init(params),jax.random.key(0),jnp.int32(0),jnp.int32(0),
        params,0,0,0.,0.,1.,hp,{})
    state,_,history=restore_checkpoint(path,template,expected_metadata=expected)
    validate_metric_history(pd.DataFrame(history),epochs=epoch)
    if not 0<=state.best_epoch<=epoch or not np.isfinite(state.preservation_weight) or state.preservation_weight<=0:
        raise ValueError('Invalid selected state/adaptive preservation weight in resume checkpoint')
    if set(state.diagnostics)!={'initial_mixing','initial_recall','alignment_gradient_norm','preservation_gradient_norm','initial_weight'}:
        raise ValueError('Missing resume calibration/selection diagnostics')
    for leaf in jax.tree_util.tree_leaves((state.params,state.best_params,state.opt_state)):
        if not np.isfinite(np.asarray(leaf)).all():raise ValueError('Nonfinite resume parameters or optimizer state')

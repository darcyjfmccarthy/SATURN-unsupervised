"""Frozen JAX inference and training-fitted visualizations for held-out cells."""

import json
from pathlib import Path
import pickle

import anndata as ad
import numpy as np
import pandas as pd

from jax_saturn.data.cache import file_sha256
from jax_saturn.data.holdout import frozen_counts, identity, write_json
from jax_saturn.data.manifest import load_manifest
from .cell_holdout import TRIALS


def ensure_umap_compatibility():
    """Support the repository's older local sklearn as well as the pinned GPU env."""
    import inspect
    import umap.umap_ as umap_module
    original = umap_module.check_array
    if 'ensure_all_finite' in inspect.signature(original).parameters or getattr(original, '_saturn_compatible', False):
        return
    def compatible(*args, **kwargs):
        if 'ensure_all_finite' in kwargs:
            kwargs['force_all_finite'] = kwargs.pop('ensure_all_finite')
        return original(*args, **kwargs)
    compatible._saturn_compatible = True
    umap_module.check_array = compatible


def restore_params(path, module, example, *args):
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    target = module.init(jax.random.key(0), jnp.asarray(example), *args)['params']
    with ocp.StandardCheckpointer() as checkpointer:
        return checkpointer.restore(Path(path).resolve() / 'params', target=target)


def heldout_embeddings(config):
    """Labels are deliberately absent from this inference interface and its NPZs."""
    import jax
    from jax_saturn.models.saturn import SaturnPretrainModule, SaturnMetricModule
    from jax_saturn.training.baseline import embed_metric
    state = json.loads(config.preprocessing.read_text())
    checkpoint = config.foundation / f'shared/pretrain_orbax/checkpoints/epoch_{config.pretrain_epochs:04d}'
    metadata = json.loads((checkpoint / 'metadata.json').read_text())
    hp = metadata['hyperparameters']
    names = tuple(metadata['species_names'])
    counts = tuple(metadata['input_shapes'][name][1] for name in names)
    if names != tuple(state['species_names']) or counts != tuple(len(state['genes'][name]) for name in names):
        raise ValueError('Checkpoint and frozen feature mappings disagree')
    if (hp['num_macrogenes'], hp['hidden_dim'], hp['embed_dim']) != (config.num_macrogenes, config.hidden_dim, config.model_dim):
        raise ValueError('Checkpoint and experiment architecture disagree')
    expected_genes = [[name + '_' + gene for gene in state['genes'][name]] for name in names]
    if hp['gene_names'] != expected_genes or metadata['source_manifest_sha256'] != state['source_manifest_sha256']:
        raise ValueError('Checkpoint and frozen gene order/provenance disagree')
    module = SaturnPretrainModule(names, counts, config.num_macrogenes, config.hidden_dim, config.model_dim,
                                 dropout=hp['dropout'])
    params = restore_params(checkpoint, module, np.ones((2, counts[0]), np.float32), 0)
    macros, teacher, ids, species = [], [], [], []
    rows = load_manifest(config.root / 'split/test.csv', check_paths=True)
    if tuple(r.species for r in rows) != names:
        raise ValueError('Test manifest species differ from frozen model')
    for code, row in enumerate(rows):
        atlas = ad.read_h5ad(row.path)
        values = frozen_counts(atlas, row.species, state)
        infer = jax.jit(lambda x: module.apply({'params': params}, x, code, train=False))
        for start in range(0, len(values), config.batch_size):
            batch = values[start:start + config.batch_size]
            size = len(batch)
            padded = np.zeros((config.batch_size, values.shape[1]), np.float32)
            padded[:size] = batch
            output = infer(padded)
            macros.append(np.asarray(output.macrogenes[:size], np.float32))
            teacher.append(np.asarray(output.embedding[:size], np.float32))
        ids.extend(atlas.obs_names.astype(str))
        species.extend([row.species] * len(atlas))
    macros = np.concatenate(macros)
    teacher = np.concatenate(teacher)
    if not np.isfinite(macros).all() or not np.isfinite(teacher).all():
        raise ValueError('Nonfinite frozen pretrained test representation')
    species, ids = np.asarray(species, str), np.asarray(ids, str)
    expected = pd.read_csv(config.root / 'split/split.csv', dtype=str, keep_default_na=False)
    if not np.array_equal(ids, expected.loc[expected['split'] == 'test', 'obs_id'].to_numpy(str)):
        raise ValueError('Test inference observation alignment changed')
    shared = config.root / 'heldout'
    shared.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(shared / 'pretrained.npz', embeddings=teacher, macrogenes=macros,
                        species=species, obs_ids=ids)
    metric = SaturnMetricModule(config.num_macrogenes, config.hidden_dim, config.model_dim)
    for trial in config.trials:
        trial_dir = config.run / trial
        if trial == 'baseline':
            selected = config.foundation / f'baseline/metric_orbax/checkpoints/epoch_{config.metric_epochs:04d}'
        else:
            selected = trial_dir / f'selected_orbax/epoch_{config.metric_epochs:04d}'
        selected_metadata = json.loads((selected / 'metadata.json').read_text())
        hp = selected_metadata['hyperparameters']
        if (hp.get('input_dim', config.num_macrogenes), hp['hidden_dim'], hp.get('embed_dim', hp.get('model_dim'))) != (config.num_macrogenes, config.hidden_dim, config.model_dim):
            raise ValueError(f'Incompatible metric checkpoint: {trial}')
        params = restore_params(selected, metric, np.ones((2, config.num_macrogenes), np.float32))
        embeddings = embed_metric(metric, params, macros, batch_size=config.batch_size)
        if embeddings.shape != (len(ids), config.model_dim) or not np.isfinite(embeddings).all():
            raise ValueError(f'Invalid test embeddings: {trial}')
        np.savez_compressed(trial_dir / 'test_embeddings.npz', embeddings=embeddings, species=species, obs_ids=ids)


def fit_project_umap(training, test, *, seed=0, reducer=None):
    if reducer is None:
        ensure_umap_compatibility()
        import umap
        reducer = umap.UMAP(n_neighbors=15, min_dist=.3, n_components=2, metric='cosine',
                            random_state=seed, transform_seed=seed, n_jobs=1)
    train_coordinates = np.asarray(reducer.fit_transform(training), np.float32)
    test_coordinates = np.asarray(reducer.transform(test), np.float32)
    if not np.isfinite(train_coordinates).all() or not np.isfinite(test_coordinates).all():
        raise ValueError('Nonfinite UMAP coordinates')
    return reducer, train_coordinates, test_coordinates


def query_metrics(query, reference, query_obs, reference_obs, search, *, k=15):
    k = min(k, len(reference))
    _, neighbors = search.search(query, reference, k)
    q_species = query_obs['species'].astype(str).to_numpy()
    r_species = reference_obs['species'].astype(str).to_numpy()
    q_labels = query_obs['labels2'].astype(str).to_numpy()
    r_labels = reference_obs['labels2'].astype(str).to_numpy()
    cross = np.mean(r_species[neighbors] != q_species[:, None], axis=1)
    same = np.mean(r_labels[neighbors] == q_labels[:, None], axis=1)
    rows = []
    for name in ['all', *sorted(set(q_species))]:
        mask = np.ones(len(query), bool) if name == 'all' else q_species == name
        rows.append({'species': name, 'queries': int(mask.sum()), 'reference_cells': len(reference), 'k': k,
                     'species_mixing_fraction': float(cross[mask].mean()),
                     'label_same_neighbor_fraction': float(same[mask].mean()),
                     'query_population': 'test', 'reference_population': 'train'})
    return rows


def test_truth(config):
    frames = []
    for row in load_manifest(config.root / 'split/test.csv', check_paths=True):
        atlas = ad.read_h5ad(row.path, backed='r')
        labels = row.species + '_' + atlas.obs[row.in_label_col].astype(str)
        frames.append(pd.DataFrame({'species': row.species, 'labels': labels,
                                   'labels2': labels.str.split('_').str[-1],
                                   'ref_labels': atlas.obs['cellType'].astype(str)}, index=atlas.obs_names))
        atlas.file.close()
    obs = pd.concat(frames)
    for column in obs:
        obs[column] = pd.Categorical(obs[column])
    return obs


def report(config, *, infer=True):
    """Post-hoc only: model selection is complete before test labels are read."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from jax_saturn.data.neighbors import NeighborSearch
    from jax_saturn.experiments.hmmff import trial_complete
    ensure_umap_compatibility()
    for trial in config.trials[1:]:
        if not trial_complete(config.run / trial, trial, epochs=config.metric_epochs):
            raise ValueError(f'{trial}: complete training before held-out reporting')
    baseline_checkpoint = config.foundation / f'baseline/metric_orbax/checkpoints/epoch_{config.metric_epochs:04d}/metadata.json'
    if not baseline_checkpoint.is_file():
        raise ValueError('Baseline has not completed the requested epochs')
    if infer:
        heldout_embeddings(config)
    truth = test_truth(config)
    assets = config.root / 'report_assets'
    assets.mkdir(parents=True, exist_ok=True)
    search = NeighborSearch(config.neighbor_backend, config.query_block, config.reference_block,
                            'gpu' if config.device in {'gpu', 'cuda'} else 'cpu')
    figure, axes = plt.subplots(len(config.trials), 2, figsize=(14, 5.5 * len(config.trials)), squeeze=False)
    interactive = make_subplots(rows=(len(config.trials) + 1) // 2, cols=2, subplot_titles=list(config.trials))
    species_names = sorted(truth.species.astype(str).unique())
    colors = plt.get_cmap('tab10')
    from matplotlib.colors import to_hex
    species_colors = {name: to_hex(colors(i)) for i, name in enumerate(species_names)}
    metrics = []
    for row, trial in enumerate(config.trials):
        directory = config.run / trial
        train = ad.read_h5ad(directory / 'evaluated_adata.h5ad')
        with np.load(directory / 'test_embeddings.npz', allow_pickle=False) as values:
            if set(values.files) != {'embeddings', 'species', 'obs_ids'}:
                raise ValueError('Unexpected test inference fields')
            obs = truth.loc[values['obs_ids'].astype(str)].copy()
            if not np.array_equal(values['species'].astype(str), obs.species.astype(str)):
                raise ValueError('Test species alignment differs')
            test = ad.AnnData(np.asarray(values['embeddings'], np.float32), obs=obs)
        fitted = directory / 'training_umap.pkl'
        if fitted.exists():
            from jax_saturn.data.work_cache import array_identity
            metadata = json.loads((directory / 'training_umap_metadata.json').read_text())
            if (metadata['sha256'] != file_sha256(fitted)
                    or metadata['fit_identity']['content'] != array_identity({'embeddings': np.asarray(train.X, np.float32)})
                    or metadata['fit_identity']['seed'] != config.seed
                    or metadata['training_obs_ids'] != array_identity({'obs_ids': np.asarray(train.obs_names, str)})):
                raise ValueError('Fitted UMAP does not match the training reference')
            with fitted.open('rb') as handle:
                reducer = pickle.load(handle)
            train_xy = np.asarray(reducer.embedding_, np.float32)
            test_xy = np.asarray(reducer.transform(test.X), np.float32)
        else:
            reducer, train_xy, test_xy = fit_project_umap(np.asarray(train.X), np.asarray(test.X), seed=config.seed)
        if not np.isfinite(test_xy).all():
            raise ValueError('Nonfinite held-out projection')
        with (directory / 'train_fitted_umap.pkl').open('wb') as handle:
            pickle.dump(reducer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        train.obsm['X_umap'] = train_xy
        train.obs['split'] = pd.Categorical(['train'] * len(train))
        test.obsm['X_umap'] = test_xy
        test.obs['split'] = pd.Categorical(['test'] * len(test))
        train.write_h5ad(directory / 'train_evaluated_adata.h5ad')
        test.write_h5ad(directory / 'test_evaluated_adata.h5ad')
        np.savez_compressed(directory / 'holdout_umap.npz', train_coordinates=train_xy, test_coordinates=test_xy,
                            train_obs_ids=np.asarray(train.obs_names, str), test_obs_ids=np.asarray(test.obs_names, str))
        write_json(directory / 'holdout_umap_metadata.json', {'fit_population': 'train', 'transform_population': 'test',
                   'training_obs_ids_sha256': identity(train.obs_names), 'test_obs_ids_sha256': identity(test.obs_names),
                   'seed': config.seed, 'metric': 'cosine', 'n_neighbors': 15, 'min_dist': .3,
                   'fitted_reducer_sha256': file_sha256(directory / 'train_fitted_umap.pkl')})
        for result in query_metrics(test.X, train.X, test.obs, train.obs, search):
            metrics.append({'trial': trial, **result})
        for column, field in enumerate(('species', 'labels2')):
            ax = axes[row, column]
            ax.scatter(train_xy[:, 0], train_xy[:, 1], s=3, c='#9ca3af', alpha=.3, rasterized=True)
            categories = sorted(test.obs[field].astype(str).unique())
            palette = plt.get_cmap('turbo', max(2, len(categories)))
            for index, category in enumerate(categories):
                mask = test.obs[field].astype(str).to_numpy() == category
                ax.scatter(test_xy[mask, 0], test_xy[mask, 1], s=9, color=palette(index), alpha=.85,
                           label=category, rasterized=True)
            field_title = 'species' if field == 'species' else 'cell type'
            ax.set(title=f'{trial}: held-out {field_title}', xlabel='UMAP 1', ylabel='UMAP 2')
            if field == 'species':
                ax.legend(markerscale=2, fontsize=8)
        plot_row, plot_col = row // 2 + 1, row % 2 + 1
        interactive.add_trace(go.Scattergl(x=train_xy[:, 0], y=train_xy[:, 1], mode='markers', name=f'{trial}: train',
                              marker={'size': 2, 'color': '#aaaaaa', 'opacity': .15}, hoverinfo='skip'), row=plot_row, col=plot_col)
        for species in sorted(test.obs.species.astype(str).unique()):
            mask = test.obs.species.astype(str).to_numpy() == species
            custom = np.column_stack([np.asarray(test.obs_names, str)[mask], test.obs.labels2.astype(str).to_numpy()[mask]])
            interactive.add_trace(go.Scattergl(x=test_xy[mask, 0], y=test_xy[mask, 1], mode='markers', name=f'{trial}: {species}',
                                  marker={'size': 5, 'opacity': .85, 'color': species_colors[species]}, customdata=custom,
                                  hovertemplate='test: %{customdata[0]}<br>cell type: %{customdata[1]}<extra>%{fullData.name}</extra>'),
                                  row=plot_row, col=plot_col)
    figure.suptitle(f'{config.test_fraction:.0%} held out per species · UMAP fitted on training cells only', fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, .98))
    for extension in ('png', 'pdf'):
        figure.savefig(assets / f'heldout_umaps.{extension}', dpi=180, bbox_inches='tight')
    plt.close(figure)
    interactive.update_layout(height=950, title='Held-out cells projected onto training-fitted UMAPs')
    interactive.write_html(assets / 'heldout_umaps.html', include_plotlyjs=True)
    frame = pd.DataFrame(metrics)
    frame.to_csv(config.root / 'heldout_metrics.csv', index=False)
    write_json(config.root / 'holdout_report.json', {'training_only_umap_fit': True,
               'query_population': 'test', 'reference_population': 'train',
               'test_cells': len(truth), 'trials': list(config.trials),
               'metrics': 'heldout_metrics.csv', 'figures': 'report_assets/heldout_umaps',
               'labels_used_for_test_inference_or_selection': False})
    from .holdout_diagnostics import diagnostics
    diagnostics(config)
    return frame

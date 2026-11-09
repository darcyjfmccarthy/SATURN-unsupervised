"""Post-hoc generalization diagnostics from finalized train/test embeddings."""

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from jax_saturn.data.cache import file_sha256
from jax_saturn.data.holdout import write_json
from jax_saturn.data.neighbors import NeighborSearch
from jax_saturn.data.work_cache import WorkCache, array_identity
from .cell_holdout import TRIALS

METRICS = ('label_vote_accuracy', 'label_same_neighbor_fraction', 'species_mixing_fraction')
TITLES = ('Cell-type vote accuracy', 'Same-type neighbours', 'Cross-species neighbours')
COLORS = dict(zip(TRIALS, ('#2563eb', '#d97706', '#059669', '#9333ea')))


def neighbor_scores(query_obs, reference_obs, neighbors, distances=None):
    """Labels enter only after neighbours have been computed in frozen space."""
    neighbors = np.asarray(neighbors)
    if (neighbors.ndim != 2 or len(neighbors) != len(query_obs) or not neighbors.shape[1]
            or not np.issubdtype(neighbors.dtype, np.integer)
            or np.any(neighbors < 0) or np.any(neighbors >= len(reference_obs))):
        raise ValueError('Neighbour indices do not align with query/reference observations')
    reference_labels = reference_obs.labels2.astype(str).to_numpy()
    query_labels = query_obs.labels2.astype(str).to_numpy()
    classes, codes = np.unique(reference_labels, return_inverse=True)
    votes = np.zeros((len(neighbors), len(classes)), np.int16)
    np.add.at(votes, (np.arange(len(neighbors))[:, None], codes[neighbors]), 1)
    # Lexicographic class order resolves equal vote counts deterministically.
    predicted = classes[votes.argmax(axis=1)]
    result = pd.DataFrame({
        'obs_id': np.asarray(query_obs.index, str),
        'species': query_obs.species.astype(str).to_numpy(), 'cell_type': query_labels,
        'predicted_cell_type': predicted,
        'label_vote_accuracy': (predicted == query_labels).astype(float),
        'label_same_neighbor_fraction': np.mean(reference_labels[neighbors] == query_labels[:, None], axis=1),
        'species_mixing_fraction': np.mean(reference_obs.species.astype(str).to_numpy()[neighbors]
                                            != query_obs.species.astype(str).to_numpy()[:, None], axis=1),
        'vote_fraction': votes.max(axis=1) / neighbors.shape[1],
    })
    if distances is not None:
        distances = np.asarray(distances)
        if distances.shape != neighbors.shape or not np.isfinite(distances).all():
            raise ValueError('Neighbour distances do not align with indices')
        result['nearest_cosine_distance'] = distances[:, 0]
    return result


def summarize_populations(train, test, trial):
    rows = []
    for dimension in ('overall', 'species', 'cell_type'):
        groups = ['all'] if dimension == 'overall' else sorted(set(train[dimension]) | set(test[dimension]))
        for group in groups:
            a = train if dimension == 'overall' else train.loc[train[dimension] == group]
            b = test if dimension == 'overall' else test.loc[test[dimension] == group]
            row = {'trial': trial, 'dimension': dimension, 'group': group,
                   'train_queries': len(a), 'test_queries': len(b)}
            for metric in METRICS:
                row[f'train_{metric}'] = a[metric].mean()
                row[f'test_{metric}'] = b[metric].mean()
                row[f'gap_pp_{metric}'] = 100 * (b[metric].mean() - a[metric].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def training_neighbors(root, directory, embeddings, obs_ids):
    metadata = json.loads((directory / 'training_umap_metadata.json').read_text())
    content = array_identity({'embeddings': np.asarray(embeddings, np.float32)})
    if (metadata['fit_identity']['content'] != content
            or metadata['training_obs_ids'] != array_identity({'obs_ids': np.asarray(obs_ids, str)})):
        raise ValueError('Training observations/embeddings differ from the finalized report')
    matches = []
    for path in (root / 'cache/report_neighbors').glob('*/manifest.json'):
        identity = json.loads(path.read_text())['identity']
        if identity.get('content') == content and identity.get('k') == 15:
            matches.append(identity)
    if len(matches) != 1:
        raise ValueError('Download the matching finalized training-neighbour cache before diagnostics')
    identity = {k: v for k, v in matches[0].items() if k not in ('schema_version', 'kind')}
    def missing():
        raise ValueError('Finalized training-neighbour cache is missing')
    neighbors = WorkCache(root / 'cache').get('report_neighbors', identity, missing)['indices']
    if neighbors.shape != (len(embeddings), 15) or np.any(neighbors == np.arange(len(embeddings))[:, None]):
        raise ValueError('Training neighbours must exclude the query itself')
    return neighbors


def _save(figure, prefix):
    import matplotlib.pyplot as plt
    for extension in ('png', 'pdf'):
        figure.savefig(f'{prefix}.{extension}', dpi=180, bbox_inches='tight')
    plt.close(figure)


def summary_figure(summary, destination, trials=TRIALS):
    import matplotlib.pyplot as plt
    rows = summary.loc[summary.dimension == 'overall'].set_index('trial').loc[list(trials)]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, metric, title in zip(axes, METRICS, TITLES):
        values = 100 * rows[[f'train_{metric}', f'test_{metric}']].to_numpy()
        left = max(0, 5 * np.floor((values.min() - 5) / 5))
        right = min(100, 5 * np.ceil((values.max() + 5) / 5))
        if metric == 'species_mixing_fraction':
            left, right = 0, max(20, right)
        for y, (trial, row) in enumerate(rows.iterrows()):
            a, b = 100 * row[f'train_{metric}'], 100 * row[f'test_{metric}']
            ax.plot([a, b], [y, y], color=COLORS[trial], lw=3)
            ax.scatter(a, y, marker='o', s=60, facecolor='white', edgecolor=COLORS[trial], zorder=3)
            ax.scatter(b, y, marker='s', s=45, color=COLORS[trial], zorder=4)
            ax.annotate(f'{a:.1f} → {b:.1f}%  ({b-a:+.1f} pp)', (left, y), xytext=(0, 13),
                        textcoords='offset points', fontsize=9)
        ax.set(title=title, xlabel='Percentage (zoomed)' if left else 'Percentage',
               ylim=(len(trials) - .4, -.7), xlim=(left, right))
        ax.grid(axis='x', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_yticks(range(len(trials)), list(trials))
    figure.suptitle('Do held-out cells retain training performance?', fontsize=15, y=1.02)
    figure.text(.5, -.03, '○ training queries (self excluded)    ■ held-out queries    |    Both query the training reference, k=15\n'
                'Same-species/atlas cell holdout; differences are descriptive, not evidence about unseen donors.',
                ha='center', fontsize=9)
    figure.tight_layout()
    _save(figure, destination / 'generalization_summary')


def _heatmap(ax, values, *, title, labels, gap=False):
    import matplotlib.pyplot as plt
    values = np.asarray(values, float)
    bound = max(2., float(np.nanmax(np.abs(values)))) if gap else 100.
    display = ax.imshow(values, cmap='RdBu' if gap else 'YlGnBu', aspect='auto',
                        vmin=-bound if gap else 0, vmax=bound)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(range(len(labels)), labels, rotation=25, ha='right')
    for y, x in np.ndindex(values.shape):
        if np.isfinite(values[y, x]):
            ax.text(x, y, f'{values[y,x]:+.1f}' if gap else f'{values[y,x]:.1f}', ha='center', va='center',
                    fontsize=8, color='white' if (abs(values[y,x]) > bound * .65 if gap else values[y,x] > 65) else '#111827')
    plt.colorbar(display, ax=ax, fraction=.04, pad=.025, label='Test − train (pp)' if gap else 'Held-out (%)')


def grouped_figures(summary, destination, *, minimum_test_cells=50, trials=TRIALS):
    import matplotlib.pyplot as plt
    species = summary.loc[summary.dimension == 'species']
    names = sorted(species.group.unique())
    labels = [f'{name}\nn={int(species.loc[species.group == name, "test_queries"].iloc[0]):,}' for name in names]
    figure, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    for row, (metric, title) in enumerate(zip(METRICS[:2], TITLES[:2])):
        for col, gap in enumerate((False, True)):
            field = f'gap_pp_{metric}' if gap else f'test_{metric}'
            values = species.pivot(index='trial', columns='group', values=field).loc[list(trials), names].to_numpy()
            if not gap:
                values *= 100
            _heatmap(axes[row, col], values, title=f'{title}: ' + ('generalization gap' if gap else 'held-out score'),
                     labels=labels, gap=gap)
            axes[row, col].set_yticks(range(len(trials)), list(trials))
    figure.suptitle('Does the average hide a species-specific generalization gap?', fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, .97))
    _save(figure, destination / 'generalization_by_species')
    types = summary.loc[(summary.dimension == 'cell_type') & (summary.test_queries >= minimum_test_cells)]
    if types.empty:
        return
    ordering = types.groupby('group').gap_pp_label_vote_accuracy.min().sort_values().head(20).index
    figure, axes = plt.subplots(1, 2, figsize=(12, max(5, .28 * len(ordering) + 1.5)), sharey=True)
    for ax, gap in zip(axes, (False, True)):
        field = 'gap_pp_label_vote_accuracy' if gap else 'test_label_vote_accuracy'
        values = types.pivot(index='group', columns='trial', values=field).loc[ordering, list(trials)].to_numpy()
        if not gap:
            values *= 100
        _heatmap(ax, values, title='Cell-type vote accuracy: ' + ('test − train' if gap else 'held-out'),
                 labels=list(trials), gap=gap)
    counts = types.drop_duplicates('group').set_index('group').test_queries
    axes[0].set_yticks(range(len(ordering)), [f'{name}  (n={counts[name]:,})' for name in ordering], fontsize=8)
    figure.suptitle('Cell types with the largest observed generalization gaps', fontsize=15)
    figure.text(.5, .01, f'Up to 20 types ranked by the largest drop in any method; ≥{minimum_test_cells} test cells.\n'
                'Types are pooled across species; all types remain in the CSV and interactive explorer.',
                ha='center', fontsize=9)
    figure.tight_layout(rect=(0, .025, 1, .97))
    _save(figure, destination / 'generalization_by_cell_type')


def interactive_errors(data, destination):
    """Filter actual test errors on existing coordinates; no UMAP is refitted."""
    from plotly.offline import get_plotlyjs
    payload = json.dumps(data, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    page = '''<!doctype html><html><head><meta charset="utf-8"><title>Held-out generalization</title>
<script>__PLOTLY__</script><style>body{font:15px system-ui;margin:24px;color:#172033}h1{font-size:23px}
label{display:inline-block;margin:6px 14px 6px 0}select{padding:6px;max-width:260px}#note{max-width:1000px;color:#526077}</style></head><body>
<h1>Where do held-out cells fail?</h1><p id="note">Filter species and cell types, then inspect wrong cell-type votes or low neighbour agreement.
The grey background shows training density. Each method has its own training-fitted UMAP; positions across methods are not directly comparable.</p>
<label>Method <select id="method"></select></label><label>Species <select id="species"></select></label>
<label>True cell type <select id="celltype"></select></label><label>Show <select id="filter">
<option value="all">All held-out cells</option><option value="wrong">Wrong cell-type vote</option><option value="low">Less than 50% same-type neighbours</option></select></label>
<label>Colour <select id="colour"><option value="agreement">Same-type neighbours</option><option value="correct">Cell-type vote correct / wrong</option></select></label>
<p id="count"></p><div id="plot"></div><script>
const data=__DATA__, $=id=>document.getElementById(id);
function options(id,values,all){const old=$(id).value;$(id).replaceChildren();if(all){let o=new Option('All','all');$(id).add(o)}
for(const value of values)$(id).add(new Option(value,value));if([...$(id).options].some(o=>o.value===old))$(id).value=old}
options('method',Object.keys(data),false);
function populate(){const d=data[$('method').value];options('species',[...new Set(d.species)].sort(),true);options('celltype',[...new Set(d.cell_type)].sort(),true);draw()}
function draw(){const d=data[$('method').value], selected=[];for(let i=0;i<d.x.length;i++){
if($('species').value!=='all'&&d.species[i]!==$('species').value)continue;
if($('celltype').value!=='all'&&d.cell_type[i]!==$('celltype').value)continue;
if($('filter').value==='wrong'&&d.correct[i])continue;if($('filter').value==='low'&&d.agreement[i]>=.5)continue;selected.push(i)}
const pick=key=>selected.map(i=>d[key][i]), custom=selected.map(i=>[d.ids[i],d.species[i],d.cell_type[i],d.predicted[i],(100*d.agreement[i]).toFixed(1)]);
let marker=$('colour').value==='agreement'?{size:5,color:pick('agreement').map(v=>100*v),colorscale:'Viridis',cmin:0,cmax:100,colorbar:{title:'Same-type<br>neighbours (%)'},opacity:.85}:
{size:5,color:pick('correct').map(v=>v?'#059669':'#dc2626'),opacity:.85};
Plotly.react('plot',[{type:'heatmap',x:d.density_x,y:d.density_y,z:d.density,colorscale:[[0,'white'],[1,'#9ca3af']],showscale:false,hoverinfo:'skip',opacity:.5},
{type:'scattergl',mode:'markers',x:pick('x'),y:pick('y'),marker,customdata:custom,
hovertemplate:'%{customdata[0]}<br>Species: %{customdata[1]}<br>True: %{customdata[2]}<br>Vote: %{customdata[3]}<br>Same-type neighbours: %{customdata[4]}%<extra></extra>'}],
{height:740,margin:{t:20},xaxis:{title:'UMAP 1',range:d.xrange},yaxis:{title:'UMAP 2',range:d.yrange},showlegend:false,uirevision:$('method').value},{responsive:true});
$('count').textContent=`Showing ${selected.length.toLocaleString()} / ${d.x.length.toLocaleString()} held-out cells. Green = correct and red = wrong in the vote view.`}
$('method').onchange=populate;for(const id of ['species','celltype','filter','colour'])$(id).onchange=draw;populate();
</script></body></html>'''
    (destination / 'heldout_errors.html').write_text(page.replace('__PLOTLY__', get_plotlyjs()).replace('__DATA__', payload))


def write_readme(summary, destination, trials=TRIALS):
    rows = summary.loc[summary.dimension == 'overall'].set_index('trial').loc[list(trials)]
    lines = ['# Do held-out cells generalize well?', '',
        'Cell-type prediction uses a majority vote among the 15 nearest training cells in the frozen embedding. '
        'Training queries exclude themselves; held-out queries use the same training reference.', '',
        '| Method | Training accuracy | Held-out accuracy | Test − train | Held-out cross-species neighbours |',
        '| --- | ---: | ---: | ---: | ---: |']
    for trial, row in rows.iterrows():
        lines.append(f'| {trial} | {100 * row.train_label_vote_accuracy:.1f}% | '
                     f'{100 * row.test_label_vote_accuracy:.1f}% | {row.gap_pp_label_vote_accuracy:+.1f} pp | '
                     f'{100 * row.test_species_mixing_fraction:.1f}% |')
    lines += ['', 'Small train/test gaps indicate little deterioration relative to training cells. '
        'Assess absolute accuracy and species mixing alongside the gap: preserving cell types does not by itself establish cross-species integration.', '',
        '- [Overall comparison](generalization_summary.png)',
        '- [Species breakdown](generalization_by_species.png)',
        '- [Largest observed cell-type gaps](generalization_by_cell_type.png): up to 20 types with at least 50 test cells, ranked by the largest drop in any method.',
        '- [Interactive errors](heldout_errors.html): filter by method, species, cell type and prediction errors.',
        '- [All group scores](generalization.csv) and [all held-out cell scores](test_cells.csv.gz).', '',
        'These are descriptive results for one cell split within the same five species and input atlases. '
        'They do not test unseen donors. Species/cell-type composition can contribute to differences, '
        'and cells are not independent biological replicates. Models, checkpoint choices and UMAPs were frozen before these diagnostics.', '']
    (destination / 'README.md').write_text('\n'.join(lines))


def diagnostics(config):
    """Regenerate from completed artifacts, without training or model selection."""
    import matplotlib
    matplotlib.use('Agg')
    root = config.root
    from jax_saturn.experiments.hmmff import trial_complete
    for trial in config.trials[1:]:
        if not trial_complete(config.run / trial, trial, epochs=config.metric_epochs):
            raise ValueError('Generalization diagnostics require finalized models')
    baseline = root / f'foundation/baseline/metric_orbax/checkpoints/epoch_{config.metric_epochs:04d}/metadata.json'
    if not baseline.is_file():
        raise ValueError('Baseline has not completed the requested epochs')
    destination = root / 'diagnostics'
    destination.mkdir(exist_ok=True)
    search = NeighborSearch(config.neighbor_backend, config.query_block, config.reference_block,
                            'gpu' if config.device in ('gpu', 'cuda') else 'cpu')
    cloud = pd.read_csv(root / 'heldout_metrics.csv').set_index(['trial', 'species'])
    comparison = pd.read_csv(root / 'run/comparison.csv').set_index('trial')
    summaries, cells, interactive, provenance = [], [], {}, {}
    for trial in config.trials:
        print(f'{trial}: checking finalized embeddings and cached training neighbours', flush=True)
        directory = config.run / trial
        train = ad.read_h5ad(directory / 'train_evaluated_adata.h5ad')
        test = ad.read_h5ad(directory / 'test_evaluated_adata.h5ad')
        if not train.obs_names.is_unique or not test.obs_names.is_unique or len(train.obs_names.intersection(test.obs_names)):
            raise ValueError('Training and test IDs must be unique and disjoint')
        train_x, test_x = np.asarray(train.X, np.float32), np.asarray(test.X, np.float32)
        train_indices = training_neighbors(root, directory, train_x, train.obs_names)
        train_scores = neighbor_scores(train.obs, train.obs, train_indices)
        for metric in METRICS[1:]:
            if not np.isclose(train_scores[metric].mean(), comparison.loc[trial, metric], atol=1e-7):
                raise ValueError(f'{trial}: cached training scores differ from original report')
        identity = {'reference': array_identity({'embeddings': train_x, 'obs_ids': np.asarray(train.obs_names, str)}),
                    'query': array_identity({'embeddings': test_x, 'obs_ids': np.asarray(test.obs_names, str)}),
                    'search': search.identity(), 'k': 15}
        def compute():
            print(f'{trial}: querying held-out cells against the frozen training reference', flush=True)
            distances, indices = search.search(test_x, train_x, 15)
            return {'indices': indices, 'distances': distances}
        cached = WorkCache(destination / 'cache').get('test_reference_neighbors', identity, compute)
        test_scores = neighbor_scores(test.obs, train.obs, cached['indices'], cached['distances'])
        differences = {metric: float(test_scores[metric].mean() - cloud.loc[(trial, 'all'), metric]) for metric in METRICS[1:]}
        if max(abs(v) for v in differences.values()) > .0005:
            raise ValueError(f'{trial}: recomputed test metrics differ by more than 0.05 percentage points: {differences}')
        summary = summarize_populations(train_scores, test_scores, trial)
        summaries.append(summary)
        test_scores.insert(0, 'trial', trial)
        cells.append(test_scores)
        xy = np.asarray(test.obsm['X_umap'])
        background = np.asarray(train.obsm['X_umap'])
        hist, xedges, yedges = np.histogram2d(background[:, 0], background[:, 1], bins=100)
        interactive[trial] = {'ids': test_scores.obs_id.tolist(), 'species': test_scores.species.tolist(),
            'cell_type': test_scores.cell_type.tolist(), 'predicted': test_scores.predicted_cell_type.tolist(),
            'correct': test_scores.label_vote_accuracy.astype(bool).tolist(),
            'agreement': test_scores.label_same_neighbor_fraction.tolist(), 'x': xy[:, 0].tolist(), 'y': xy[:, 1].tolist(),
            'density': np.log1p(hist.T).tolist(), 'density_x': ((xedges[:-1] + xedges[1:]) / 2).tolist(),
            'density_y': ((yedges[:-1] + yedges[1:]) / 2).tolist(),
            'xrange': [float(min(xedges[0], xy[:, 0].min())), float(max(xedges[-1], xy[:, 0].max()))],
            'yrange': [float(min(yedges[0], xy[:, 1].min())), float(max(yedges[-1], xy[:, 1].max()))]}
        provenance[trial] = {**identity, 'test_metric_differences_from_cloud': differences,
                             'selected_epoch': int(comparison.loc[trial, 'selected_epoch'])}
        pd.concat(summaries, ignore_index=True).to_csv(destination / 'generalization.csv', index=False)
    summary = pd.concat(summaries, ignore_index=True)
    pd.concat(cells, ignore_index=True).to_csv(destination / 'test_cells.csv.gz', index=False)
    summary_figure(summary, destination, config.trials)
    grouped_figures(summary, destination, trials=config.trials)
    interactive_errors(interactive, destination)
    write_readme(summary, destination, config.trials)
    write_json(destination / 'summary.json', {'question': 'Do held-out cells retain training performance?',
        'training_query_policy': 'Finalized training queries against training references, excluding self',
        'test_query_policy': 'All held-out queries against training references', 'k': 15,
        'label_vote_policy': 'Majority among 15 neighbours; lexicographic ties',
        'models_refitted': False, 'umaps_refitted': False, 'selection_uses_test_labels': False,
        'scope': 'Cell holdout within the same five species and input atlases; not an unseen-donor test',
        'diagnostics_source_sha256': file_sha256(Path(__file__)),
        'original_execution_result_sha256': file_sha256(root / 'execution_result.json')
            if (root / 'execution_result.json').exists() else None, 'trials': provenance})
    print(f'Generalization diagnostics saved to {destination}', flush=True)
    return summary

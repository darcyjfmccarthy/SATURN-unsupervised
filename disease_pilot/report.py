"""Disease/source/donor views of frozen, training-fitted representations."""

import json
from pathlib import Path
import pickle

import anndata as ad
import numpy as np
import pandas as pd

from .cloud import audit_split
from .preparation import sha256, write_json
from jax_saturn.data.work_cache import array_identity
from jax_saturn.experiments.holdout_report import fit_project_umap

FIELDS = ('condition', 'source', 'donor_id', 'cellType', 'organ', 'qc_total_counts', 'qc_mito_fraction')
GROUP_COLORS = {'atlas_reference': '#9ca3af', 'study_WT': '#2563eb', 'study_KO': '#d97706'}


def pretrained_coordinates(config):
    destination = config.root / 'pretraining_view'
    destination.mkdir(exist_ok=True)
    train = ad.read_h5ad(config.foundation / 'baseline/saturn_results/adata_pretrain.h5ad')
    with np.load(config.root / 'heldout/pretrained.npz', allow_pickle=False) as values:
        test_x, test_ids = values['embeddings'], values['obs_ids'].astype(str)
    identity = {'train': array_identity({'embeddings': np.asarray(train.X, np.float32),
                                        'obs_ids': np.asarray(train.obs_names, str)}),
                'test': array_identity({'embeddings': test_x, 'obs_ids': test_ids}), 'seed': config.seed,
                'policy': 'train-fit-test-transform-cosine15-min_dist0.3'}
    path = destination / 'coordinates.npz'
    if path.exists():
        record = json.loads((destination / 'provenance.json').read_text())
        if (record['identity'] != identity or record['coordinates_sha256'] != sha256(path)
                or record['reducer_sha256'] != sha256(destination / 'train_fitted_umap.pkl')):
            raise ValueError('Pretraining UMAP identity changed')
        with np.load(path, allow_pickle=False) as values:
            train_xy, test_xy = values['train_coordinates'], values['test_coordinates']
            if (not np.array_equal(values['train_obs_ids'], np.asarray(train.obs_names, str))
                    or not np.array_equal(values['test_obs_ids'], test_ids)):
                raise ValueError('Pretraining UMAP observation order changed')
    else:
        reducer, train_xy, test_xy = fit_project_umap(np.asarray(train.X), test_x, seed=config.seed)
        with (destination / 'train_fitted_umap.pkl').open('wb') as handle:
            pickle.dump(reducer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        np.savez_compressed(path, train_coordinates=train_xy, test_coordinates=test_xy,
                            train_obs_ids=np.asarray(train.obs_names, str), test_obs_ids=test_ids)
        write_json(destination / 'provenance.json', {'identity': identity, 'fit_population': 'train',
                   'transform_population': 'test', 'coordinates_sha256': sha256(path),
                   'reducer_sha256': sha256(destination / 'train_fitted_umap.pkl')})
    return pd.DataFrame(np.concatenate([train_xy, test_xy]),
                        index=[*train.obs_names.astype(str), *test_ids], columns=['x', 'y'])


def representation_coordinates(config):
    coordinates = {'pretraining': pretrained_coordinates(config)}
    for trial in config.trials:
        with np.load(config.run / trial / 'holdout_umap.npz', allow_pickle=False) as values:
            coordinates[trial] = pd.DataFrame(
                np.concatenate([values['train_coordinates'], values['test_coordinates']]),
                index=np.concatenate([values['train_obs_ids'], values['test_obs_ids']]).astype(str),
                columns=['x', 'y'])
    return coordinates


def subset_masks(metadata):
    skin = metadata.organ.eq('skin')
    return {'all_mouse': np.ones(len(metadata), bool), 'skin_context': skin.to_numpy(),
            'skin_keratinocytes': (skin & metadata.cellType.eq('keratinocyte')).to_numpy(),
            'study_fibroblasts': (metadata.source.eq('external_study') & metadata.cellType.eq('fibroblast')).to_numpy()}


def draw_field(ax, xy, metadata, field):
    values = metadata[field]
    if field.startswith('qc_'):
        numeric = pd.to_numeric(values, errors='coerce').to_numpy(float)
        if field == 'qc_total_counts':
            numeric = np.log10(1 + numeric)
        finite = np.isfinite(numeric)
        if (~finite).any():
            ax.scatter(*xy[~finite].T, s=4, c='#cbd5e1', alpha=.5, label='Unavailable', rasterized=True)
        if finite.any():
            points = ax.scatter(*xy[finite].T, s=5, c=numeric[finite], cmap='viridis', alpha=.7, rasterized=True)
            ax.figure.colorbar(points, ax=ax, fraction=.04, label='log10(1 + raw counts)' if field == 'qc_total_counts' else 'Raw mitochondrial fraction')
    else:
        values = values.fillna('unavailable').astype(str).to_numpy()
        categories = sorted(set(values))
        import matplotlib.pyplot as plt
        palette = plt.get_cmap('tab20')
        for index, category in enumerate(categories):
            mask = values == category
            ax.scatter(*xy[mask].T, s=5, c=[GROUP_COLORS.get(category, palette(index % 20))],
                       alpha=.7, label=category, rasterized=True)
        ax.legend(fontsize=6, markerscale=2, loc='best', ncol=2 if len(categories) > 8 else 1)
    ax.set(xlabel='UMAP 1', ylabel='UMAP 2')


def static_views(metadata, coordinates, destination):
    import matplotlib.pyplot as plt
    products = []
    for subset, mask in subset_masks(metadata).items():
        fields = FIELDS if subset == 'all_mouse' else ('cellType',) if subset == 'skin_context' else ('disease_group', 'donor_id')
        figure, axes = plt.subplots(len(fields), len(coordinates), figsize=(6 * len(coordinates), 4.5 * len(fields)), squeeze=False)
        for col, (name, xy) in enumerate(coordinates.items()):
            for row, field in enumerate(fields):
                ax = axes[row, col]
                if mask.any():
                    draw_field(ax, xy[mask], metadata.loc[mask], field)
                else:
                    ax.text(.5, .5, 'No cells in this subset', ha='center', transform=ax.transAxes)
                ax.set_title(f'{name}: {field} · n={int(mask.sum()):,}')
        title = subset.replace('_', ' ')
        if subset == 'study_fibroblasts':
            title += ' · no fibroblast reference in the original mouse skin subset'
        figure.suptitle(title + ' · each method has its own training-fitted coordinates')
        figure.tight_layout(rect=(0, 0, 1, .98))
        prefix = 'disease_overview' if subset == 'all_mouse' else subset
        for extension in ('png', 'pdf'):
            file = destination / f'{prefix}.{extension}'
            figure.savefig(file, dpi=150, bbox_inches='tight')
            products.append(file)
        plt.close(figure)
    return products


def interactive_view(metadata, coordinates, destination):
    from plotly.offline import get_plotlyjs
    data = {'ids': metadata.index.tolist(), 'fields': {},
            'coordinates': {key: value.tolist() for key, value in coordinates.items()},
            'subsets': {key: np.flatnonzero(value).tolist() for key, value in subset_masks(metadata).items()}}
    for field in (*FIELDS, 'split', 'disease_group'):
        values = metadata[field]
        data['fields'][field] = [None if pd.isna(v) else float(v) if field.startswith('qc_') else str(v) for v in values]
    payload = json.dumps(data, allow_nan=False, separators=(',', ':')).replace('<', '\\u003c')
    page = '''<!doctype html><html><head><meta charset="utf-8"><title>RDEB disease views</title>
<script>__PLOTLY__</script><style>body{font:15px system-ui;margin:24px;color:#172033}label{display:inline-block;margin:8px}select{padding:5px}p{max-width:1100px}</style></head><body>
<h1>RDEB: source, condition and donor in frozen SATURN representations</h1>
<p>UMAP fits training cells and transforms held-out cells. Each representation has its own coordinate system.
Atlas condition is reference, not a verified healthy control. Missing atlas donor/QC values are unavailable.
The study has two WT and two knockout animals; this cell split does not test unseen animals.</p>
<label>Representation <select id="method"></select></label><label>Subset <select id="subset"></select></label>
<label>Colour <select id="field"></select></label><label>Split <select id="split"><option value="all">All</option><option>train</option><option>test</option></select></label>
<label>Donor <select id="donor"></select></label><p id="count"></p><div id="plot"></div><script>
const data=__DATA__, $=id=>document.getElementById(id);
function options(id,values){for(const value of values)$(id).add(new Option(value,value))}
options('method',Object.keys(data.coordinates));options('subset',Object.keys(data.subsets));
options('field',['disease_group',...Object.keys(data.fields)]);options('donor',['all',...new Set(data.fields.donor_id)]);
function draw(){const method=$('method').value,field=$('field').value,xy=data.coordinates[method],f=data.fields;
const ids=data.subsets[$('subset').value].filter(i=>($('split').value==='all'||f.split[i]===$('split').value)&&($('donor').value==='all'||f.donor_id[i]===$('donor').value));
const trace=indices=>({type:'scattergl',mode:'markers',x:indices.map(i=>xy[i][0]),y:indices.map(i=>xy[i][1]),
customdata:indices.map(i=>[data.ids[i],f.source[i],f.condition[i],f.donor_id[i],f.cellType[i],f.split[i]]),
hovertemplate:'%{customdata[0]}<br>%{customdata[1]} · %{customdata[2]}<br>Donor: %{customdata[3]}<br>%{customdata[4]} · %{customdata[5]}<extra>%{fullData.name}</extra>'});
let traces=[];if(field.startsWith('qc_')){const finite=ids.filter(i=>f[field][i]!==null),missing=ids.filter(i=>f[field][i]===null);
if(missing.length)traces.push({...trace(missing),name:'Unavailable',marker:{size:4,color:'#cbd5e1',opacity:.5}});
traces.push({...trace(finite),name:field,marker:{size:5,opacity:.8,color:finite.map(i=>field==='qc_total_counts'?Math.log10(1+f[field][i]):f[field][i]),colorscale:'Viridis',showscale:true}})
}else{for(const value of [...new Set(ids.map(i=>f[field][i]??'unavailable'))].sort()){
const indices=ids.filter(i=>(f[field][i]??'unavailable')===value),colors={atlas_reference:'#9ca3af',study_WT:'#2563eb',study_KO:'#d97706'};
traces.push({...trace(indices),name:value,marker:{size:5,opacity:.8,...(colors[value]?{color:colors[value]}:{})}})}}
Plotly.react('plot',traces,{height:780,xaxis:{title:'UMAP 1'},yaxis:{title:'UMAP 2'},uirevision:method},{responsive:true});
$('count').textContent=`Showing ${ids.length.toLocaleString()} mouse cells. Library-size colours use log10(1 + raw counts).`}
for(const id of ['method','subset','field','split','donor'])$(id).onchange=draw;draw();
</script></body></html>'''
    path = destination / 'disease_explorer.html'
    path.write_text(page.replace('__PLOTLY__', get_plotlyjs()).replace('__DATA__', payload))
    return path


def disease_report(config, metadata_path):
    metadata = audit_split(config, metadata_path)
    metadata['disease_group'] = np.where(metadata.source.eq('atlas'), 'atlas_reference',
                                        np.where(metadata.condition.eq('healthy'), 'study_WT', 'study_KO'))
    coordinates = {}
    expected = pd.read_csv(config.root / 'split/split.csv', dtype=str).obs_id
    for name, xy in representation_coordinates(config).items():
        if not xy.index.is_unique or set(xy.index) != set(expected):
            raise ValueError(f'{name}: coordinate IDs do not cover the declared split')
        coordinates[name] = xy.loc[metadata.index].to_numpy(np.float32)
        if not np.isfinite(coordinates[name]).all():
            raise ValueError(f'{name}: nonfinite disease coordinates')
    destination = config.root / 'report_assets'
    destination.mkdir(exist_ok=True)
    metadata_file = destination / 'disease_metadata.csv.gz'
    metadata.to_csv(metadata_file)
    coordinates_file = destination / 'disease_coordinates.npz'
    np.savez_compressed(coordinates_file, obs_ids=metadata.index.to_numpy(str), **coordinates)
    products = [metadata_file, coordinates_file, *static_views(metadata, coordinates, destination),
                interactive_view(metadata, coordinates, destination)]
    write_json(config.root / 'disease_report.json', {
        'metadata_sha256': sha256(metadata_path), 'representations': list(coordinates), 'trials': list(config.trials),
        'mouse_cells': len(metadata), 'fit_population': 'train', 'transform_population': 'test',
        'labels_used_for_selection': False, 'donor_holdout': False,
        'groups': metadata.groupby(['source', 'condition', 'donor_id'], observed=True).size().rename('cells').reset_index().to_dict('records'),
        'coordinate_sha256': {name: array_identity({'xy': values}) for name, values in coordinates.items()},
        'files': {str(p.relative_to(config.root)): sha256(p) for p in products},
        'limitations': ['Normalized expression; count likelihood is a compatibility surrogate',
                        'Two WT and two knockout animals; cell holdout does not establish unseen-animal transfer',
                        'No original mouse skin fibroblast reference; atlas donor/assay metadata unavailable']})
    return metadata


def verify_disease_report(config, metadata_path):
    record = json.loads((config.root / 'disease_report.json').read_text())
    if (record['metadata_sha256'] != sha256(metadata_path) or record['trials'] != list(config.trials)
            or record['representations'] != ['pretraining', *config.trials]):
        raise ValueError('Disease report configuration/metadata changed')
    for name, expected in record['files'].items():
        if sha256(config.root / name) != expected:
            raise ValueError(f'Disease report output changed: {name}')
    metadata = audit_split(config, metadata_path)
    with np.load(config.root / 'report_assets/disease_coordinates.npz', allow_pickle=False) as values:
        if (not np.array_equal(values['obs_ids'], metadata.index.to_numpy(str))
                or len(metadata) != record['mouse_cells']):
            raise ValueError('Disease report observation alignment changed')
    return record

"""Reproducible cell splits and fitted, label-independent count transforms."""

import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from .cache import file_sha256
from .manifest import load_manifest

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def identity(values):
    return hashlib.sha256(json.dumps(list(map(str, values)), separators=(',', ':')).encode()).hexdigest()


def portable(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def make_split(observations, *, seed=0, test_fraction=.1):
    """Species and stable IDs alone determine membership, never expression/labels."""
    if not 0 < test_fraction < 1 or seed < 0:
        raise ValueError('Invalid split fraction or seed')
    frames = []
    seen = set()
    for code, species in enumerate(sorted(observations)):
        ids = pd.Index(observations[species]).astype(str)
        if not ids.is_unique or seen.intersection(ids) or any(not x for x in ids):
            raise ValueError('Observation IDs must be nonempty and globally unique')
        seen.update(ids)
        count = int(np.ceil(test_fraction * len(ids)))
        if count < 1 or count >= len(ids):
            raise ValueError(f'{species}: split must leave training and test observations')
        ordered = np.sort(ids.to_numpy())
        rng = np.random.default_rng(np.random.SeedSequence([seed, code]))
        test_ids = set(ordered[rng.permutation(len(ordered))[:count]])
        frames.append(pd.DataFrame({'species': species, 'obs_id': ids,
                                    'split': ['test' if x in test_ids else 'train' for x in ids]}))
    if not frames:
        raise ValueError('No species to split')
    return pd.concat(frames, ignore_index=True)


def validate_split(split, observations, *, test_fraction=.1):
    if list(split.columns) != ['species', 'obs_id', 'split'] or split.obs_id.duplicated().any():
        raise ValueError('Invalid split columns or repeated observation IDs')
    if set(split['split']) != {'train', 'test'} or set(split.species) != set(observations):
        raise ValueError('Invalid split populations')
    for species, ids in observations.items():
        group = split[split.species == species]
        if set(group.obs_id) != set(map(str, ids)):
            raise ValueError(f'Split does not cover source observations: {species}')
        if int((group['split'] == 'test').sum()) != int(np.ceil(test_fraction * len(ids))):
            raise ValueError(f'Incorrect test count: {species}')


def prepare_holdout(manifest, destination, *, seed=0, test_fraction=.1):
    """Materialize disjoint raw atlases before the training preprocessor is called."""
    destination = Path(destination).resolve()
    rows = load_manifest(manifest, check_paths=True)
    observations, inputs = {}, []
    for row in rows:
        atlas = ad.read_h5ad(row.path, backed='r')
        if not atlas.var_names.is_unique:
            raise ValueError(f'Nonunique genes: {row.species}')
        observations[row.species] = list(atlas.obs_names.astype(str))
        inputs.append({'species': row.species, 'atlas_sha256': file_sha256(row.path),
                       'protein_sha256': file_sha256(row.embedding_path),
                       'obs_ids_sha256': identity(atlas.obs_names), 'gene_ids_sha256': identity(atlas.var_names)})
        atlas.file.close()
    split = make_split(observations, seed=seed, test_fraction=test_fraction)
    validate_split(split, observations, test_fraction=test_fraction)
    destination.mkdir(parents=True, exist_ok=True)
    previous_path = destination / 'split_record.json'
    if previous_path.exists():
        previous = json.loads(previous_path.read_text())
        for relative, checksum in previous['derived_atlases'].items():
            if file_sha256(destination / relative) != checksum:
                raise ValueError(f'Derived atlas changed: {relative}')
    record = {'schema_version': 1, 'seed': seed, 'test_fraction': test_fraction,
              'manifest_sha256': file_sha256(manifest), 'inputs': inputs}
    record_path = destination / 'source_inputs.json'
    if record_path.exists() and json.loads(record_path.read_text()) != record:
        raise ValueError('Source inputs or split settings changed; use a new experiment directory')
    split_path = destination / 'split.csv'
    if split_path.exists():
        existing = pd.read_csv(split_path, dtype=str, keep_default_na=False)
        if not existing.equals(split):
            raise ValueError('Existing split differs from the deterministic experiment split')
    else:
        split.to_csv(split_path, index=False)
    manifests = {'train': [], 'test': []}
    composition = []
    for row in rows:
        atlas = ad.read_h5ad(row.path)
        for name in manifests:
            ids = split.loc[(split.species == row.species) & (split['split'] == name), 'obs_id']
            subset = atlas[ids].copy()
            path = destination / 'atlases' / name / f'{row.species}.h5ad'
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                subset.write_h5ad(path)
            else:
                saved = ad.read_h5ad(path, backed='r')
                if identity(saved.obs_names) != identity(ids) or identity(saved.var_names) != identity(atlas.var_names):
                    raise ValueError(f'Saved subset identity changed: {path}')
                saved.file.close()
            manifests[name].append({'species': row.species, 'path': portable(path),
                                    'embedding_path': portable(row.embedding_path), 'in_label_col': row.in_label_col})
            for label, count in subset.obs[row.in_label_col].astype(str).value_counts().items():
                composition.append({'species': row.species, 'split': name, 'cell_type': label, 'cells': int(count)})
        del atlas
    for name, entries in manifests.items():
        pd.DataFrame(entries).to_csv(destination / f'{name}.csv', index=False)
    pd.DataFrame(composition).to_csv(destination / 'split_composition.csv', index=False)
    record['split_sha256'] = file_sha256(split_path)
    # Preserve the original input record independently of derived file timestamps.
    write_json(record_path, {k: v for k, v in record.items() if k != 'split_sha256'})
    derived = {str(p.relative_to(destination)): file_sha256(p) for p in sorted((destination / 'atlases').rglob('*.h5ad'))}
    write_json(destination / 'split_record.json', {**record, 'derived_atlases': derived,
               'counts': split.groupby(['species', 'split']).size().unstack(fill_value=0).to_dict('index'),
               'claim': 'strict inductive cell holdout; protein embeddings are externally pretrained'})
    return split


def save_fitted_preprocessing(data, path, *, hv_genes):
    state = {'schema_version': 1, 'species_names': list(data.species_names),
             'genes': {name: [gene.removeprefix(name + '_') for gene in genes]
                       for name, genes in zip(data.species_names, data.gene_names)},
             'hv_genes': hv_genes, 'source_manifest_sha256': data.source_manifest_sha256,
             'training_obs_ids_sha256': identity(data.obs.index),
             'embedding_cache_sha256': data.embedding_cache_sha256,
             'policy': 'protein-overlap-lowercase; training-only seurat_v3 HVGs after normalize_total(1e4); model receives original selected counts and applies log1p'}
    path = Path(path)
    if path.exists() and json.loads(path.read_text()) != state:
        raise ValueError('Fitted preprocessing changed; use a new foundation')
    write_json(path, state)
    return state


def frozen_counts(atlas, species, state):
    """Apply fitted feature order. No labels, normalization fitting or HVG selection."""
    if state.get('schema_version') != 1 or species not in state['species_names']:
        raise ValueError('Incompatible frozen preprocessing species/schema')
    if not atlas.obs_names.is_unique or not atlas.var_names.is_unique:
        raise ValueError('Duplicate cell or gene IDs')
    genes = state['genes'][species]
    missing = pd.Index(genes).difference(atlas.var_names)
    if len(missing):
        raise ValueError(f'{species}: missing required fitted genes: {list(missing[:10])}')
    source = atlas.X.data if hasattr(atlas.X, 'tocsr') else np.asarray(atlas.X)
    if not np.isfinite(source).all() or np.any(source < 0):
        raise ValueError('Expression must contain finite nonnegative counts')
    values = atlas[:, genes].X
    values = values.toarray() if hasattr(values, 'toarray') else np.asarray(values)
    return np.asarray(values, np.float32)

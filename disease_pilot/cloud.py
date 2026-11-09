"""Portable, checksum-verified inputs for the first disease cell-holdout run."""

import json
from pathlib import Path
import shutil

import anndata as ad
import numpy as np
import pandas as pd

from .preparation import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
AUDITS = ('evaluation_metadata.csv.gz', 'preparation_record.json', 'notebook_execution.json',
          'resolved_metadata.json', 'sample_covariates.csv', 'external_qc.csv.gz',
          'gene_mapping.csv.gz', 'protein_coverage.csv', 'atlas_audit.csv',
          'count_preparation.json', 'condition_by_technical_batch.csv')


def data_path(path, root=ROOT):
    path = Path(path)
    if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] != 'data':
        raise ValueError(f'Bundle paths must be repository-relative under data/: {path}')
    return Path(root) / path


def copy_verified(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(destination) != sha256(source):
            raise ValueError(f'Existing input differs; choose a new bundle destination: {destination}')
    else:
        shutil.copy2(source, destination)
    if sha256(destination) != sha256(source):
        raise ValueError(f'Input copy checksum differs: {destination}')


def prepare_bundle(prepared, destination, manifest, *, root=ROOT):
    root, prepared = Path(root).resolve(), Path(prepared).resolve()
    destination, manifest = Path(destination).resolve(), Path(manifest).resolve()
    data_path(destination.relative_to(root), root)
    data_path(manifest.relative_to(root), root)
    record = json.loads((prepared / 'preparation_record.json').read_text())
    execution = json.loads((prepared / 'notebook_execution.json').read_text())
    version = record['identity']['config']['dataset_version_id']
    raw = root / 'data/disease_pilot/downloads' / f'{version}.h5ad'
    if sha256(raw) != execution['source_download_sha256']:
        raise ValueError('Pinned raw study download changed')
    if sha256(Path(__file__).with_name('preparation.py')) != record['identity']['preparation_source_sha256']:
        raise ValueError('Preparation implementation changed; revalidate preparation in a fresh destination')
    for name, expected in record['outputs_sha256'].items():
        if sha256(prepared / name) != expected:
            raise ValueError(f'Prepared output checksum differs: {name}')
    # Validate the original atlas/protein identities too, before any new fitting.
    for entry in record['identity']['inputs']:
        for path_key, hash_key in (('atlas_path', 'atlas_sha256'), ('embedding_path', 'embedding_sha256')):
            if sha256(entry[path_key]) != entry[hash_key]:
                raise ValueError(f'Original preparation input changed: {entry[path_key]}')
    rows = pd.read_csv(prepared / 'exploration.csv')
    products, counts = [], {}
    for index, row in rows.iterrows():
        source = Path(row.path)
        protein = (root / row.embedding_path).resolve()
        target = destination / source.name
        copy_verified(source, target)
        atlas = ad.read_h5ad(target, backed='r')
        if not atlas.obs_names.is_unique or not atlas.var_names.is_unique:
            raise ValueError('Nonunique prepared observation/gene IDs')
        for start in range(0, atlas.n_obs, 1024):
            values = atlas.X[start:start + 1024]
            data = values.data if hasattr(values, 'tocsr') else np.asarray(values)
            if not np.isfinite(data).all() or np.any(data < 0):
                raise ValueError('Nonfinite or negative model expression')
            np.testing.assert_allclose(np.asarray(values.sum(axis=1)).ravel(), 10000, rtol=5e-5)
        counts[row.species] = {'cells': atlas.n_obs, 'genes': atlas.n_vars}
        atlas.file.close()
        rows.loc[index, 'path'] = str(target.relative_to(root))
        rows.loc[index, 'embedding_path'] = str(protein.relative_to(root))
        products.extend([target, protein])
    manifest.parent.mkdir(parents=True, exist_ok=True)
    encoded = rows.to_csv(index=False)
    if manifest.exists() and manifest.read_text() != encoded:
        raise ValueError('Manifest already exists with different inputs')
    manifest.write_text(encoded)
    products.append(manifest)
    for name in AUDITS:
        copy_verified(prepared / name, destination / name)
        products.append(destination / name)
    receipt = {'schema_version': 1, 'population_source': 'full exploration.csv, not donor-fold training inputs',
               'claim': '90/10 cell holdout within five species; donors can occur in both splits',
               'expression_scale': 'per-cell sum 10000 over retained genes; normalized expression, not raw counts',
               'counts': counts, 'preparation_record_sha256': sha256(prepared / 'preparation_record.json'),
               'files': [{'path': str(p.relative_to(root)), 'size': p.stat().st_size, 'sha256': sha256(p)}
                         for p in sorted(set(products))]}
    write_json(destination / 'bundle_record.json', receipt)
    verify_bundle(destination / 'bundle_record.json', root=root)
    return receipt


def verify_bundle(path, *, root=ROOT):
    record = json.loads(Path(path).read_text())
    if record['schema_version'] != 1:
        raise ValueError('Unsupported input bundle schema')
    for entry in record['files']:
        file = data_path(entry['path'], root)
        if file.stat().st_size != entry['size'] or sha256(file) != entry['sha256']:
            raise ValueError(f'Input bundle checksum differs: {entry["path"]}')
    return record


def load_metadata(path):
    metadata = pd.read_csv(path, index_col=0, low_memory=False)
    metadata.index = metadata.index.astype(str)
    metadata.index.name = 'obs_id'
    required = {'source', 'condition', 'donor_id', 'cellType', 'organ', 'qc_total_counts', 'qc_mito_fraction'}
    if not metadata.index.is_unique or not required.issubset(metadata):
        raise ValueError('Disease metadata requires unique observation IDs and source/donor/condition/QC fields')
    if not metadata.source.isin(['atlas', 'external_study']).all():
        raise ValueError('Unexpected mouse metadata source')
    return metadata


def audit_split(config, metadata_path=None):
    import os
    metadata_path = metadata_path or ROOT / os.environ['DISEASE_METADATA']
    metadata = load_metadata(metadata_path)
    split = pd.read_csv(config.root / 'split/split.csv', dtype=str, keep_default_na=False)
    mouse = split.loc[split.species == 'm_musculus'].set_index('obs_id')
    if set(mouse.index) != set(metadata.index):
        raise ValueError('Disease metadata must cover exactly the augmented mouse observations')
    joined = mouse.join(metadata.drop(columns='species', errors='ignore'), validate='one_to_one')
    fields = ['split', 'source', 'condition', 'donor_id', 'organ', 'cellType']
    composition = joined.groupby(fields, observed=True, dropna=False).size().rename('cells').reset_index()
    composition.to_csv(config.root / 'split/disease_composition.csv', index=False)
    return joined

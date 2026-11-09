"""Configuration and population checks shared by the holdout notebook and CLI."""

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from jax_saturn.data.cache import file_sha256
from jax_saturn.data.holdout import ROOT, identity, portable, prepare_holdout, write_json
from jax_saturn.data.manifest import load_manifest

TRIALS = ('baseline', 'infonce', 'mmd', 'ot')


def parse_trials(value):
    trials = tuple(value.split(',')) if isinstance(value, str) else tuple(value)
    if (not trials or trials[0] != 'baseline' or len(trials) < 2
            or len(set(trials)) != len(trials) or set(trials) - set(TRIALS)):
        raise ValueError('Trials must be unique supported names, beginning with baseline and a label-free method')
    return trials


@dataclass(frozen=True)
class HoldoutConfig:
    manifest: str = 'data/human_monkey_mouse_frog_fish.csv'
    output: str = 'out/human_monkey_mouse_frog_fish_jax_cell_holdout'
    seed: int = 0
    test_fraction: float = .1
    hv_genes: int = 2000
    num_macrogenes: int = 200
    hidden_dim: int = 256
    model_dim: int = 256
    batch_size: int = 512
    pretrain_epochs: int = 20
    metric_epochs: int = 30
    device: str = 'gpu'
    neighbor_backend: str = 'jax'
    query_block: int = 1024
    reference_block: int = 8192
    trials: tuple = TRIALS

    def __post_init__(self):
        object.__setattr__(self, 'trials', parse_trials(self.trials))

    @classmethod
    def from_env(cls):
        aliases = {'manifest': 'IN_DATA', 'output': 'RUN_ROOT', 'batch_size': 'METRIC_BATCH_SIZE',
                   'test_fraction': 'TEST_FRACTION', 'device': 'DEVICE', 'neighbor_backend': 'NEIGHBOR_BACKEND'}
        defaults = cls()
        values = {name: type(value)(os.environ.get(aliases.get(name, name.upper()), value))
                  for name, value in asdict(defaults).items() if name != 'trials'}
        return cls(**values, trials=parse_trials(os.environ.get('SATURN_TRIALS', TRIALS)))

    @property
    def root(self):
        path = Path(self.output).expanduser()
        return (ROOT / path).resolve()

    @property
    def source(self):
        return (ROOT / self.manifest).resolve()

    @property
    def foundation(self):
        return self.root / 'foundation'

    @property
    def run(self):
        return self.root / 'run'

    @property
    def preprocessing(self):
        return self.foundation / 'shared/preprocessing.json'

    def scientific_record(self):
        record = asdict(self)
        for key in ('device', 'neighbor_backend', 'query_block', 'reference_block'):
            record.pop(key)
        # Keep historical four-method experiment records byte-shape compatible.
        if self.trials == TRIALS:
            record.pop('trials')
        else:
            record['trials'] = list(self.trials)
        record.update(manifest=portable(self.source), output=portable(self.root), mixed_precision='fp32')
        return record

    def execution_env(self):
        aliases = {'manifest': 'IN_DATA', 'output': 'RUN_ROOT', 'batch_size': 'METRIC_BATCH_SIZE',
                   'test_fraction': 'TEST_FRACTION', 'device': 'DEVICE', 'neighbor_backend': 'NEIGHBOR_BACKEND'}
        return {**os.environ, **{aliases.get(name, name.upper()): str(value)
                for name, value in asdict(self).items() if name != 'trials'},
                'SATURN_TRIALS': ','.join(self.trials)}


def prepare(config):
    if min(config.hv_genes, config.num_macrogenes, config.hidden_dim, config.model_dim,
           config.batch_size, config.pretrain_epochs, config.metric_epochs) < 1:
        raise ValueError('Dimensions, batches and epochs must be positive')
    if config.root == ROOT or not config.root.is_relative_to(ROOT / 'out'):
        # External temporary output directories are supported by local fixture tests.
        if not str(config.root).startswith('/tmp/'):
            raise ValueError('Experiment output must be beneath out/ or a temporary fixture directory')
    record_path = config.root / 'experiment_config.json'
    record = config.scientific_record()
    if record_path.exists() and json.loads(record_path.read_text()) != record:
        raise ValueError('Experiment configuration changed; choose a fresh output directory')
    split = prepare_holdout(config.source, config.root / 'split', seed=config.seed, test_fraction=config.test_fraction)
    write_json(record_path, record)
    return split


def preflight(config, *, gpu=False):
    """Validate actual source/subset bytes and exclusion, not just manifest paths."""
    split_root = config.root / 'split'
    if os.environ.get('DISEASE_INPUT_BUNDLE'):
        from disease_pilot.cloud import verify_bundle, audit_split
        verify_bundle(ROOT / os.environ['DISEASE_INPUT_BUNDLE'])
        audit_split(config)
    saved_config = json.loads((config.root / 'experiment_config.json').read_text())
    if saved_config != config.scientific_record():
        raise ValueError('Scientific settings differ from the prepared experiment')
    record = json.loads((split_root / 'split_record.json').read_text())
    if file_sha256(config.source) != record['manifest_sha256']:
        raise ValueError('Source manifest changed')
    sources = load_manifest(config.source, check_paths=True)
    if [r.species for r in sources] != sorted(record['counts']):
        raise ValueError('Species population changed')
    for row, expected in zip(sources, record['inputs']):
        if file_sha256(row.path) != expected['atlas_sha256'] or file_sha256(row.embedding_path) != expected['protein_sha256']:
            raise ValueError(f'Source input contents changed: {row.species}')
    for path, checksum in record['derived_atlases'].items():
        if file_sha256(split_root / path) != checksum:
            raise ValueError(f'Derived atlas changed: {path}')
    if file_sha256(split_root / 'split.csv') != record['split_sha256']:
        raise ValueError('Split CSV changed')
    split = pd.read_csv(split_root / 'split.csv', dtype=str, keep_default_na=False)
    from jax_saturn.data.holdout import make_split, validate_split
    observations = {}
    for row in sources:
        atlas = ad.read_h5ad(row.path, backed='r')
        observations[row.species] = list(map(str, atlas.obs_names))
        atlas.file.close()
    validate_split(split, observations, test_fraction=config.test_fraction)
    if not split.equals(make_split(observations, seed=config.seed, test_fraction=config.test_fraction)):
        raise ValueError('Split is not the declared deterministic species split')
    for population in ('train', 'test'):
        for row in load_manifest(split_root / f'{population}.csv', check_paths=True):
            atlas = ad.read_h5ad(row.path, backed='r')
            expected = split.loc[(split.species == row.species) & (split['split'] == population), 'obs_id']
            if list(atlas.obs_names.astype(str)) != list(expected):
                raise ValueError(f'{population} observations differ from saved split: {row.species}')
            atlas.file.close()
    train_ids = split.loc[split['split'] == 'train', 'obs_id'].to_numpy(str)
    artifact = config.foundation / 'shared/label_free_artifact.npz'
    if artifact.exists():
        with np.load(artifact, allow_pickle=False) as values:
            if set(values.files) != {'embeddings', 'macrogenes', 'species', 'obs_ids'} or not np.array_equal(values['obs_ids'], train_ids):
                raise ValueError('Training artifact must contain exactly training observations and four label-free fields')
    if config.preprocessing.exists():
        state = json.loads(config.preprocessing.read_text())
        if state['training_obs_ids_sha256'] != identity(train_ids):
            raise ValueError('Preprocessing fit population differs from training split')
    if gpu:
        import jax
        import jax.numpy as jnp
        from jax_saturn.data.neighbors import NeighborSearch
        if jax.__version__ != '0.6.2':
            raise ValueError('Expected JAX 0.6.2')
        device = jax.devices('gpu')[0]
        with jax.default_device(device):
            (jnp.ones((4, 4)) @ jnp.ones((4, 4))).block_until_ready()
        NeighborSearch(config.neighbor_backend, config.query_block, config.reference_block, 'gpu').validate_device()
    result = {'counts': record['counts'], 'split_sha256': record['split_sha256'],
              'training_cells': len(train_ids), 'test_cells': int((split['split'] == 'test').sum()),
              'architecture': [config.num_macrogenes, config.hidden_dim, config.model_dim],
              'strict_exclusion_verified': True, 'gpu_execution_verified': gpu}
    write_json(config.root / 'preflight.json', result)
    if gpu:
        # Later population audits run without initializing CUDA. Keep the
        # successful execution receipt instead of replacing it with gpu=False.
        write_json(config.root / 'gpu_execution_preflight.json', result)
    return result

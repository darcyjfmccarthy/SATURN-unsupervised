"""Leakage, frozen feature/inference alignment, projection and deadline boundaries."""

from dataclasses import replace
from datetime import datetime, timezone
import importlib.util
import json
import io
import os
import contextlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import anndata as ad
import jax
import numpy as np
import pandas as pd

from jax_saturn.data.holdout import frozen_counts, make_split, save_fitted_preprocessing, validate_split
from jax_saturn.data.preparation import prepare_centroids, prepare_data
from jax_saturn.experiments.cell_holdout import HoldoutConfig, prepare, preflight
from jax_saturn.experiments.holdout_report import fit_project_umap, query_metrics
from jax_saturn.verification.test_pretrain import write_atlas_fixture

ROOT = Path(__file__).resolve().parents[2]


def runner():
    import sys
    sys.path.insert(0, str(ROOT / 'scripts'))
    spec = importlib.util.spec_from_file_location('holdout_runner_test', ROOT / 'scripts/run_cell_holdout_jax.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CellHoldoutTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def config(self):
        source = self.root / 'source'
        source.mkdir()
        manifest = write_atlas_fixture(source)
        return HoldoutConfig(manifest=str(manifest), output=str(self.root / 'run'), hv_genes=32,
            num_macrogenes=4, hidden_dim=9, model_dim=5, batch_size=16,
            pretrain_epochs=1, metric_epochs=1, device='cpu', neighbor_backend='sklearn')

    def test_split_deterministic_species_rounding_and_coverage(self):
        observations = {'b': [f'b{i}' for i in range(11)], 'a': [f'a{i}' for i in range(20)]}
        split = make_split(observations)
        validate_split(split, observations)
        self.assertEqual(split.groupby(['species','split']).size().to_dict(),
                         {('a','test'):2, ('a','train'):18, ('b','test'):2, ('b','train'):9})
        reordered = make_split({name: list(reversed(ids)) for name, ids in reversed(list(observations.items()))})
        self.assertEqual(split.set_index('obs_id')['split'].to_dict(), reordered.set_index('obs_id')['split'].to_dict())
        with self.assertRaises(ValueError):
            make_split({'a':['same','a'], 'b':['same','b']})
        with self.assertRaises(ValueError):
            make_split({'a':['only']})

    def test_test_counterfactual_does_not_change_fitted_training_inputs(self):
        config = self.config()
        split = prepare(config)
        preflight(config)
        data = prepare_data(config.root / 'split/train.csv', cache_dir=self.root / 'cache', hv_genes=32)
        centroids = prepare_centroids(data, self.root / 'centroids.npz', hv_genes=32, num_macrogenes=4)
        state = save_fitted_preprocessing(data, config.preprocessing, hv_genes=32)
        self.assertEqual(state['species_names'], list(data.species_names))
        for path in (self.root / 'source').glob('*.h5ad'):
            atlas = ad.read_h5ad(path)
            test_ids = set(split.loc[split['split'] == 'test', 'obs_id'])
            mask = atlas.obs_names.isin(test_ids)
            values = atlas.X.toarray()
            values[mask] = 10000
            atlas.X = values
            labels = atlas.obs.cellType.astype(str).to_numpy()
            labels[mask] = 'test_labels_changed'
            atlas.obs['cellType'] = labels
            atlas.write_h5ad(path)
        with self.assertRaisesRegex(ValueError, 'Source input contents changed'):
            preflight(config)
        other = replace(config, output=str(self.root / 'counterfactual'))
        second_split = prepare(other)
        self.assertTrue(split.equals(second_split))
        again = prepare_data(other.root / 'split/train.csv', cache_dir=self.root / 'cache2', hv_genes=32)
        second_centroids = prepare_centroids(again, self.root / 'centroids2.npz', hv_genes=32, num_macrogenes=4)
        self.assertEqual(data.gene_names, again.gene_names)
        for first, second in zip(data.values, again.values):
            np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(centroids['scores'], second_centroids['scores'])
        self.assertTrue(data.obs.equals(again.obs))
        from jax_saturn.models.saturn import SaturnPretrainModule, init_pretrain_params
        module = SaturnPretrainModule(data.species_names, data.gene_counts, 4, 9, 5)
        first = init_pretrain_params(module, jax.random.key(0), centroids['scores'])
        second = init_pretrain_params(module, jax.random.key(0), second_centroids['scores'])
        for x, y in zip(jax.tree_util.tree_leaves(first), jax.tree_util.tree_leaves(second)):
            np.testing.assert_array_equal(x, y)

    def test_frozen_counts_reorders_without_label_or_fitting_access(self):
        atlas = ad.AnnData(np.array([[2,3,4],[5,6,7]], np.float32),
                          obs=pd.DataFrame(index=['x','y']), var=pd.DataFrame(index=['extra','g2','g1']))
        state = {'schema_version':1, 'species_names':['a'], 'genes':{'a':['g1','g2']}}
        with patch('scanpy.pp.highly_variable_genes', side_effect=AssertionError('no fitting')):
            np.testing.assert_array_equal(frozen_counts(atlas, 'a', state), [[4,3],[7,6]])
        with self.assertRaisesRegex(ValueError, 'missing required'):
            frozen_counts(atlas[:, :2], 'a', state)
        with self.assertRaises(ValueError):
            frozen_counts(atlas, 'unseen', state)
        atlas.X[0,0] = -1
        with self.assertRaises(ValueError):
            frozen_counts(atlas, 'a', state)

    def test_umap_fits_training_only_and_transforms_test(self):
        train, test = np.ones((10,3), np.float32), np.zeros((2,3), np.float32)
        reducer = Mock()
        reducer.fit_transform.return_value = np.ones((10,2))
        reducer.transform.return_value = np.zeros((2,2))
        fit_project_umap(train, test, reducer=reducer)
        reducer.fit_transform.assert_called_once_with(train)
        reducer.transform.assert_called_once_with(test)

    def test_query_reference_population_is_training_only(self):
        from jax_saturn.data.neighbors import NeighborSearch
        reference = np.array([[1,0],[0,1]], np.float32)
        queries = np.array([[1,0]], np.float32)
        qobs = pd.DataFrame({'species':['a'], 'labels2':['T']})
        robs = pd.DataFrame({'species':['b','a'], 'labels2':['T','B']})
        result = query_metrics(queries, reference, qobs, robs, NeighborSearch('sklearn'), k=1)[0]
        self.assertEqual(result['species_mixing_fraction'], 1)
        self.assertEqual(result['label_same_neighbor_fraction'], 1)
        self.assertEqual(result['query_population'], 'test')
        self.assertEqual(result['reference_population'], 'train')

    def test_deadline_gate_includes_foundation_and_never_launches_when_expired(self):
        module = runner()
        config = self.config()
        config.root.mkdir()
        with patch.object(module, 'epoch_at', return_value=0):
            with self.assertRaises(module.TimeBudgetError):
                module.timing_gate(config, 0)
        record = json.loads((config.root / 'timing_gate.json').read_text())
        self.assertEqual(set(record['stages']), {'pretrain','baseline','infonce','mmd','ot'})
        with patch.object(module, 'run_process') as process:
            with self.assertRaises(ValueError):
                module.run(config, start='2020-01-01T00:00:00Z')
            process.assert_not_called()

    def test_original_window_cannot_be_replaced_with_a_new_start(self):
        module = runner()
        config = self.config()
        config.root.mkdir()
        (config.root / 'execution_options.json').write_text(json.dumps({'vm_start_time': 'older-window'}))
        with patch.object(module, 'preflight'), patch.object(module, 'run_process') as process:
            with self.assertRaisesRegex(ValueError, 'original VM deadline'):
                module.run(config, start=datetime.now(timezone.utc).isoformat())
            process.assert_not_called()

    def test_provider_termination_time_can_only_shorten_the_deadline(self):
        module = runner()
        start = '2026-10-19T11:00:00Z'
        self.assertEqual(module.effective_deadline(start, '2026-10-19T17:00:10Z'), module.parse_start(start) + 21600)
        self.assertEqual(module.effective_deadline(start, '2026-10-19T16:59:50Z'), module.parse_start(start) + 21590)

    def test_completed_checkpoint_with_interrupted_exports_is_not_complete(self):
        module = runner()
        config = self.config()
        results = config.foundation / 'baseline/saturn_results'
        results.mkdir(parents=True)
        with patch.object(module, 'epoch_at', return_value=1):
            self.assertFalse(module.foundation_complete(config, 'baseline'))
            (results / 'metric_summary.json').write_text(json.dumps({'epoch':1}))
            pd.DataFrame({'epoch':[1]}).to_csv(results / 'metric_history.csv', index=False)
            (results / 'final_adata.h5ad').touch()
            self.assertTrue(module.foundation_complete(config, 'baseline'))

    def test_resume_regenerates_final_exports_without_repeating_epochs(self):
        module = runner()
        config = self.config()
        artifact = config.foundation / 'shared/label_free_artifact.npz'
        artifact.parent.mkdir(parents=True)
        artifact.touch()
        calls = []
        with patch.object(module, 'preflight'), patch.object(module, 'epoch_at', return_value=1), \
                patch.object(module, 'foundation_complete', return_value=True), \
                patch.object(module, 'stage_baseline'), patch.object(module, 'foundation_inventory', return_value={}), \
                patch('jax_saturn.experiments.hmmff.trial_complete', return_value=False), \
                patch.object(module, 'run_process', side_effect=lambda command, **kwargs: calls.append(list(map(str,command)))):
            module.run(config, start=datetime.now(timezone.utc).isoformat(), skip_notebook=True)
        training = [command for command in calls if '--objective' in command]
        self.assertEqual(len(training), 3)
        for command in training:
            self.assertEqual(command[command.index('--epochs') + 1], '1')
            self.assertTrue(command[command.index('--resume') + 1].endswith('epoch_0001'))

    def test_inventory_excludes_uncommitted_and_interrupted_checkpoint_files(self):
        module = runner()
        config = self.config()
        for relative in ('shared/pretrain_orbax/checkpoints/epoch_0001/params/data',
                         'shared/pretrain_orbax/checkpoints/epoch_0002/params/data',
                         'shared/pretrain_orbax/interrupted/old/checkpoints/epoch_0001/params/data'):
            path = config.foundation / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('fixture')
        (config.foundation / 'shared/pretrain_orbax/checkpoints/epoch_0001/metadata.json').write_text('{}')
        inventory = module.foundation_inventory(config)
        self.assertEqual(len(inventory), 2)
        self.assertFalse(any('interrupted' in path or 'epoch_0002' in path for path in inventory))

    def test_notebook_reads_historical_dimensions_and_restores_fresh_defaults(self):
        notebook = json.loads((ROOT / 'notebooks/jax/human_monkey_mouse_frog_fish.ipynb').read_text())
        source = ''.join(notebook['cells'][2]['source'])
        metadata = self.root / 'shared/pretrain_orbax/checkpoints/epoch_0020/metadata.json'
        metadata.parent.mkdir(parents=True)
        metadata.write_text(json.dumps({'hyperparameters': {'num_macrogenes':300, 'hidden_dim':384, 'embed_dim':384}}))
        for reuse, expected in [('1', (300,384,384)), ('0', (200,256,256))]:
            with patch.dict(os.environ, {'HMMFF_REUSE_FOUNDATION':reuse, 'HMMFF_OUTPUT_DIR':str(self.root)}), \
                    patch('jax.devices', return_value=['fixture_gpu']), contextlib.redirect_stdout(io.StringIO()):
                scope = {}
                exec(compile(source, 'notebook-setup', 'exec'), scope)
            self.assertEqual((scope['NUM_MACROGENES'], scope['HIDDEN_DIM'], scope['MODEL_DIM']), expected)


if __name__ == '__main__':
    unittest.main()

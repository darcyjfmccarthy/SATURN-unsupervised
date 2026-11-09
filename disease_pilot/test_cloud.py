"""Selected-method execution, portable staging and strict metadata alignment."""

from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from disease_pilot.cloud import audit_split, data_path, load_metadata, verify_bundle
from disease_pilot.preparation import sha256
from disease_pilot.report import interactive_view, subset_masks
from jax_saturn.experiments.cell_holdout import HoldoutConfig, TRIALS, parse_trials
from jax_saturn.verification.test_cell_holdout import runner

ROOT = Path(__file__).resolve().parents[1]


class DiseaseCloudTests(unittest.TestCase):
    def test_selected_methods_and_historical_config_compatibility(self):
        default = HoldoutConfig()
        self.assertEqual(default.trials, TRIALS)
        self.assertNotIn('trials', default.scientific_record())
        with patch.dict(os.environ, {'SATURN_TRIALS': 'baseline,infonce'}):
            config = HoldoutConfig.from_env()
        self.assertEqual(config.trials, ('baseline', 'infonce'))
        self.assertEqual(config.scientific_record()['trials'], ['baseline', 'infonce'])
        self.assertEqual(config.execution_env()['SATURN_TRIALS'], 'baseline,infonce')
        for value in ('baseline', 'infonce,baseline', 'baseline,infonce,infonce', 'baseline,unknown'):
            with self.assertRaises(ValueError):
                parse_trials(value)
        self.assertEqual(set(runner().stage_paths(config)), {'pretrain', 'baseline', 'infonce'})

    def test_timing_gate_reserves_only_selected_stages(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = replace(HoldoutConfig(), output=temporary, trials=('baseline', 'infonce'))
            module = runner()
            import time
            with patch.object(module, 'epoch_at', return_value=0):
                record = module.timing_gate(config, time.time() + 21600)
            self.assertEqual(set(record['stages']), {'pretrain', 'baseline', 'infonce'})
            self.assertTrue(record['passes'])

    def test_missing_frozen_artifact_is_not_silently_rebuilt(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = replace(HoldoutConfig(), output=temporary, trials=('baseline', 'infonce'),
                             pretrain_epochs=1, metric_epochs=1)
            (config.root / 'foundation_manifest.json').write_text('{}')
            module = runner()
            from datetime import datetime, timezone
            with patch.object(module, 'preflight'), patch.object(module, 'epoch_at', return_value=1), \
                    patch.object(module, 'foundation_complete', return_value=True), \
                    patch.object(module, 'run_process') as process:
                with self.assertRaisesRegex(ValueError, 'restore the verified backup'):
                    module.run(config, start=datetime.now(timezone.utc).isoformat(), skip_notebook=True)
                process.assert_not_called()

    def test_bundle_hashes_and_safe_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'data/metadata.csv'
            path.parent.mkdir()
            path.write_text('obs_id,condition\na,healthy\n')
            record = root / 'bundle.json'
            record.write_text(json.dumps({'schema_version': 1, 'files': [
                {'path': 'data/metadata.csv', 'size': path.stat().st_size, 'sha256': sha256(path)}]}))
            verify_bundle(record, root=root)
            path.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                verify_bundle(record, root=root)
            for value in ('/tmp/file', '../data/file', 'out/file'):
                with self.assertRaises(ValueError):
                    data_path(value, root)

    def metadata(self):
        return pd.DataFrame({'source': ['atlas', 'external_study', 'external_study'],
            'condition': ['atlas_reference', 'healthy', 'disease'], 'donor_id': ['unknown', 'WT1', 'KO1'],
            'cellType': ['keratinocyte', 'keratinocyte', 'fibroblast'], 'organ': ['skin'] * 3,
            'qc_total_counts': [np.nan, 1000., 2000.], 'qc_mito_fraction': [np.nan, .01, .03]},
            index=['atlas:a', 'study:b', 'study:c'])

    def test_metadata_alignment_rejects_duplicates_missing_and_extra_mouse_cells(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'split').mkdir()
            metadata = self.metadata()
            path = root / 'metadata.csv.gz'
            metadata.to_csv(path)
            split = pd.DataFrame({'species': ['m_musculus'] * 3, 'obs_id': metadata.index,
                                  'split': ['train', 'train', 'test']})
            split.to_csv(root / 'split/split.csv', index=False)
            config = replace(HoldoutConfig(), output=str(root))
            joined = audit_split(config, path)
            self.assertEqual(joined.index.tolist(), metadata.index.tolist())
            self.assertTrue(pd.isna(joined.loc['atlas:a', 'qc_total_counts']))
            metadata.iloc[:2].to_csv(path)
            with self.assertRaisesRegex(ValueError, 'exactly'):
                audit_split(config, path)
            pd.concat([metadata, metadata.iloc[:1]]).to_csv(path)
            with self.assertRaisesRegex(ValueError, 'unique'):
                load_metadata(path)

    def test_staging_includes_declared_metadata_without_parent_tree_upload(self):
        spec = importlib.util.spec_from_file_location('stage_disease_test', ROOT / 'scripts/stage_saturn_inputs.py')
        stage = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stage)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'data').mkdir()
            manifest = root / 'data/source.csv'
            pd.DataFrame([{'species': s, 'path': f'data/{s}.h5ad', 'embedding_path': f'data/{s}.pt'}
                          for s in ('a', 'b')]).to_csv(manifest, index=False)
            for name in ('a.h5ad', 'a.pt', 'b.h5ad', 'b.pt', 'metadata.csv'):
                (root / 'data' / name).touch()
            bundle = root / 'data/bundle.json'
            bundle.write_text(json.dumps({'files': [{'path': 'data/metadata.csv'}, {'path': 'data/a.h5ad'}]}))
            with patch.object(stage, 'ROOT', root):
                commands = list(stage.commands(manifest, 'test', 'upload', bundle))
            self.assertEqual(len(commands), 7)
            self.assertTrue(any(c[-1] == 'gs://test/inputs/metadata.csv' for c in commands))
            self.assertTrue(all(c[:3] == ['gcloud', 'storage', 'cp'] for c in commands))

    def test_interactive_view_preserves_missing_qc_and_safe_embedded_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            metadata = self.metadata()
            metadata['split'] = ['train', 'train', 'test']
            metadata['disease_group'] = ['atlas_reference', 'study_WT', 'study_KO']
            self.assertEqual(subset_masks(metadata)['skin_keratinocytes'].sum(), 2)
            path = interactive_view(metadata, {'baseline': np.zeros((3, 2))}, Path(temporary))
            page = path.read_text()
            payload = page.rsplit('<script>', 1)[1].split(', $=id=>', 1)[0]
            self.assertNotIn('NaN', payload)
            self.assertIn('"qc_total_counts":[null,1000.0,2000.0]', page)


if __name__ == '__main__':
    unittest.main()

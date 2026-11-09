"""Generalization scores, reference alignment and frozen cache provenance."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from jax_saturn.data.work_cache import WorkCache, array_identity
from jax_saturn.experiments.holdout_diagnostics import neighbor_scores, summarize_populations, training_neighbors


class HoldoutDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        self.reference = pd.DataFrame({'species': ['human', 'human', 'mouse', 'mouse'],
            'labels2': ['A', 'B', 'A', 'B']}, index=['a', 'b', 'c', 'd'])
        self.query = pd.DataFrame({'species': ['human', 'mouse', 'human'],
            'labels2': ['A', 'B', 'unseen']}, index=['x', 'y', 'z'])

    def test_reference_votes_and_absent_test_type(self):
        indices = np.array([[0, 2, 1], [1, 3, 2], [0, 2, 1]])
        scores = neighbor_scores(self.query, self.reference, indices, np.ones((3, 3)) * .2)
        self.assertEqual(scores.predicted_cell_type.tolist(), ['A', 'B', 'A'])
        np.testing.assert_allclose(scores.label_vote_accuracy, [1, 1, 0])
        np.testing.assert_allclose(scores.label_same_neighbor_fraction, [2/3, 2/3, 0])
        np.testing.assert_allclose(scores.species_mixing_fraction, [1/3, 1/3, 1/3])
        self.assertEqual(scores.obs_id.tolist(), ['x', 'y', 'z'])
        tied = neighbor_scores(self.query.iloc[:1], self.reference, np.array([[1, 0]]))
        self.assertEqual(tied.predicted_cell_type.iloc[0], 'A')
        with self.assertRaisesRegex(ValueError, 'align'):
            neighbor_scores(self.query, self.reference, np.array([[8], [1], [2]]))

    def test_group_counts_and_population_weighting(self):
        training = neighbor_scores(self.reference, self.reference, np.array([[1, 2], [0, 3], [0, 1], [1, 2]]))
        test = neighbor_scores(self.query, self.reference, np.array([[0, 2], [1, 3], [0, 2]]))
        summary = summarize_populations(training, test, 'baseline')
        overall = summary.loc[summary.dimension == 'overall'].iloc[0]
        by_species = summary.loc[summary.dimension == 'species']
        self.assertEqual(by_species.train_queries.sum(), len(training))
        self.assertEqual(by_species.test_queries.sum(), len(test))
        weighted = np.average(by_species.test_label_vote_accuracy, weights=by_species.test_queries)
        self.assertAlmostEqual(overall.test_label_vote_accuracy, weighted)
        self.assertAlmostEqual(overall.gap_pp_label_vote_accuracy,
                               100 * (test.label_vote_accuracy.mean() - training.label_vote_accuracy.mean()))
        unseen = summary.loc[(summary.dimension == 'cell_type') & (summary.group == 'unseen')].iloc[0]
        self.assertEqual(unseen.train_queries, 0)
        self.assertTrue(np.isnan(unseen.gap_pp_label_vote_accuracy))

    def test_cached_training_neighbors_require_exact_embedding_and_id_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / 'run/baseline'
            directory.mkdir(parents=True)
            embeddings = np.arange(60, dtype=np.float32).reshape(20, 3)
            ids = np.asarray([f'cell_{i}' for i in range(20)])
            content = array_identity({'embeddings': embeddings})
            (directory / 'training_umap_metadata.json').write_text(json.dumps({
                'fit_identity': {'content': content}, 'training_obs_ids': array_identity({'obs_ids': ids})}))
            indices = (np.arange(20)[:, None] + np.arange(1, 16)) % 20
            WorkCache(root / 'cache').get('report_neighbors', {'content': content, 'k': 15}, lambda: {'indices': indices})
            np.testing.assert_array_equal(training_neighbors(root, directory, embeddings, ids), indices)
            with self.assertRaisesRegex(ValueError, 'observations/embeddings'):
                training_neighbors(root, directory, embeddings, ids[::-1])
            path = next((root / 'cache/report_neighbors').glob('*/values.npz'))
            with path.open('ab') as handle:
                handle.write(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                training_neighbors(root, directory, embeddings, ids)


if __name__ == '__main__':
    unittest.main()

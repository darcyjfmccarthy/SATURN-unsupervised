"""Exact search, metrics, seed preservation, cache integrity and staged execution."""

from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import jax
import numpy as np

from jax_saturn.data import graphs
from jax_saturn.data.neighbors import NeighborSearch
from jax_saturn.data.work_cache import WorkCache
from jax_saturn.training.label_agnostic import build_graphs, LabelConfig, train_label_free
from jax_saturn.experiments.hmmff import trial_complete, validate_foundation
from label_agnostic import objectives as reference
from label_agnostic.metrics import embedding_benchmark_metrics
from jax_saturn.verification import test_label_training as fixture_module

DEVICE=os.environ.get('SATURN_TEST_NEIGHBOR_DEVICE','cpu')
ROOT=Path(__file__).resolve().parents[2]


class NeighborTests(unittest.TestCase):
    def test_non_tied_and_block_boundaries(self):
        x=np.random.default_rng(99).normal(size=(73,19)).astype(np.float32)
        expected=NeighborSearch('sklearn').search(x,x,12,self_indices=np.arange(len(x)))
        for qb,rb in ((8,13),(16,32),(1024,8192)):
            with self.subTest(blocks=(qb,rb)):
                actual=NeighborSearch('jax',qb,rb,DEVICE).search(x,x,12,self_indices=np.arange(len(x)))
                np.testing.assert_array_equal(actual[1],expected[1])
                np.testing.assert_allclose(actual[0],expected[0],atol=1e-5,rtol=0)

    def test_ties_zero_duplicates_and_self(self):
        x=np.asarray([[1,0],[1,0],[0,0],[0,1],[0,-1],[-1,0]],np.float32)
        results=[]
        for qb,rb in ((2,2),(7,3),(1024,8192)):
            search=NeighborSearch('jax',qb,rb,DEVICE)
            distances,indices=search.search(x,x,5,self_indices=np.arange(6))
            self.assertFalse(np.any(indices==np.arange(6)[:,None]))
            np.testing.assert_array_equal(indices[2],[0,1,3,4,5])
            np.testing.assert_array_equal(distances[2],np.ones(5))
            np.testing.assert_array_equal(indices[0],[1,2,3,4,5])
            results.append((distances,indices))
        for result in results[1:]:
            for a,b in zip(results[0],result):np.testing.assert_array_equal(a,b)
        d,i=NeighborSearch('jax',3,2,DEVICE).search(x[:1],x[:1],0,self_indices=[0])
        self.assertEqual(i.shape,(1,0))
        # Cross-species queries do not remove a coincident reference row.
        _,i=NeighborSearch('jax',2,2,DEVICE).search(x[:1],x[:2],2)
        np.testing.assert_array_equal(i,[[0,1]])

    def test_explicit_gpu_does_not_fallback(self):
        with patch('jax.devices',side_effect=RuntimeError('CUDA unavailable')):
            with self.assertRaisesRegex(RuntimeError,'CUDA unavailable'):
                NeighborSearch('jax',device='gpu').validate_device()

    def test_seeded_negatives_unchanged_and_small_species(self):
        rng=np.random.default_rng(4)
        x=rng.normal(size=(123,17)).astype(np.float32)
        species=np.asarray(['a']*60+['b']*60+['c']*2+['d'])
        for seed in (0,1,998):
            got=graphs.build_preservation_graph(x,species,neighbor_k=5,negative_k=7,seed=seed)
            want=reference.build_preservation_graph(x,species,neighbor_k=5,negative_k=7,seed=seed)
            for a,b in zip(got,want):np.testing.assert_array_equal(a,b)
        gpu=graphs.build_preservation_graph(x,species,neighbor_k=5,negative_k=7,seed=0,
            search=NeighborSearch('jax',16,32,DEVICE))
        for a,b in zip(gpu,graphs.build_preservation_graph(x,species,neighbor_k=5,negative_k=7,seed=0)):
            np.testing.assert_allclose(a,b,atol=1e-5,rtol=0)

    def test_metrics_and_reciprocal_agreement(self):
        rng=np.random.default_rng(12)
        x=rng.normal(size=(137,23)).astype(np.float32)
        macros=rng.normal(size=(137,11)).astype(np.float32)
        species=np.asarray(['a']*47+['b']*45+['c']*45)
        labels=np.asarray([str(i%5) for i in range(len(x))])
        search=NeighborSearch('jax',16,32,DEVICE)
        got=graphs.build_cross_species_positives(x,macros,species,search=search)
        want=graphs.build_cross_species_positives(x,macros,species)
        for a,b in zip(got,want):np.testing.assert_array_equal(a,b)
        cpu=embedding_benchmark_metrics(x,species,labels)
        gpu=embedding_benchmark_metrics(x,species,labels,search=search)
        for key in cpu:self.assertLessEqual(abs(cpu[key]-gpu[key]),1e-4,key)
        teacher=graphs.build_preservation_graph(x,species)[3]
        self.assertLessEqual(abs(graphs.topology_recall_at_50(x,species,teacher)-
            graphs.topology_recall_at_50(x,species,teacher,search=search)),1e-4)
        self.assertEqual(graphs.species_mixing_fraction(x,species),graphs.species_mixing_fraction(x,species,search=search))

    def test_atomic_cache_and_corruption_rejection(self):
        with tempfile.TemporaryDirectory() as root:
            cache=WorkCache(root)
            calls=[]
            def build():calls.append(1);return {'v':np.arange(10)}
            a=cache.get('fixture',{'seed':0},build)
            b=cache.get('fixture',{'seed':0},build)
            np.testing.assert_array_equal(a['v'],b['v'])
            self.assertEqual(len(calls),1)
            cache.get('fixture',{'seed':1},build)
            self.assertEqual(len(calls),2)
            path=Path(cache.events[0]['path'])
            (path/'values.npz').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'Corrupt/incomplete'):
                cache.get('fixture',{'seed':0},build)
            (path/'manifest.json').unlink()
            with self.assertRaisesRegex(ValueError,'Corrupt/incomplete'):
                cache.get('fixture',{'seed':0},build)

    def test_cached_and_uncached_graphs(self):
        rng=np.random.default_rng(12)
        artifact={'embeddings':rng.normal(size=(40,5)).astype(np.float32),
            'macrogenes':rng.normal(size=(40,4)).astype(np.float32),
            'species':np.asarray(['a']*20+['b']*20),'obs_ids':np.asarray([str(i) for i in range(40)])}
        with tempfile.TemporaryDirectory() as root:
            cache=WorkCache(root)
            search=NeighborSearch('jax',16,32,DEVICE)
            values=[]
            for objective in ('infonce','mmd','ot','infonce'):
                values.append(build_graphs(artifact,objective,LabelConfig(model_dim=5,hidden_dim=7),search=search,cache=cache))
            for item in values[1:]:
                np.testing.assert_array_equal(values[0][3],item[3])
                self.assertEqual(values[0][4],item[4])
            self.assertEqual(sum(e['kind']=='preservation' and not e['hit'] for e in cache.events),1)
            self.assertEqual(sum(e['kind']=='reciprocal' and not e['hit'] for e in cache.events),1)

    def test_foundation_failure_never_calls_completed_trainers(self):
        with tempfile.TemporaryDirectory() as root, \
             patch('jax_saturn.training.pretrain.train_pretrain',side_effect=AssertionError('must not train')), \
             patch('jax_saturn.training.baseline.train_baseline',side_effect=AssertionError('must not train')):
            with self.assertRaises(FileNotFoundError):validate_foundation(root)


class ExactResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse fixture creation, not its tests.
        cls.fixture=fixture_module.LabelTrainingTests
        cls.fixture.setUpClass()
        cls.addClassCleanup(cls.fixture.doClassCleanups)

    def test_two_epoch_resume_matches_uninterrupted_all_objectives(self):
        f=self.fixture
        search=NeighborSearch('jax',8,8,DEVICE)
        for objective in ('infonce','mmd','ot'):
            with self.subTest(objective=objective), jax.default_device(jax.devices(DEVICE)[0]):
                root=f.root/('exact_'+objective)
                config=replace(f.config,epochs=4)
                full,history,_=train_label_free(f.artifact,f.checkpoint,root/'full',objective=objective,
                    config=config,neighbor_search=search,cache_dir=root/'cache')
                train_label_free(f.artifact,f.checkpoint,root/'staged',objective=objective,
                    config=replace(config,epochs=2),neighbor_search=search,cache_dir=root/'cache')
                self.assertFalse(trial_complete(root/'staged',objective,epochs=30))
                self.assertTrue(trial_complete(root/'staged',objective,epochs=2))
                resumed,resumed_history,_=train_label_free(f.artifact,f.checkpoint,root/'staged',objective=objective,
                    config=config,resume=root/'staged/checkpoints/epoch_0002',neighbor_search=search,cache_dir=root/'cache')
                self.assertEqual(history,resumed_history)
                for name in ('params','opt_state','best_params','rng'):
                    for a,b in zip(jax.tree_util.tree_leaves(getattr(full,name)),jax.tree_util.tree_leaves(getattr(resumed,name))):
                        if name=='rng':a,b=jax.random.key_data(a),jax.random.key_data(b)
                        np.testing.assert_array_equal(a,b)
                for name in ('preservation_weight','best_epoch','best_step','best_mixing','best_recall','diagnostics'):
                    self.assertEqual(getattr(full,name),getattr(resumed,name))
                self.assertTrue(trial_complete(root/'staged',objective,epochs=4))
                self.assertFalse(trial_complete(root/'staged',objective,epochs=30))

if __name__=='__main__':unittest.main()

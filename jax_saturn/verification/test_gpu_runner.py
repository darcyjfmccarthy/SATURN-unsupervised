"""One production launch, checkpoint reuse, timing gate and backup boundaries."""

from contextlib import ExitStack
from datetime import datetime,timezone
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jax_saturn.experiments.hmmff import OBJECTIVES

ROOT=Path(__file__).resolve().parents[2]


def script(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class RunnerTests(unittest.TestCase):
    def exercise(self,*,passes=True,already_complete=False):
        runner=script('run_hmmff_gpu')
        epochs={name:30 if already_complete else 0 for name in OBJECTIVES};calls=[]
        def epoch(path):return epochs.get(Path(path).name,0)
        def child(command,**kwargs):
            command=list(map(str,command));calls.append(command)
            if '--objective' in command:
                objective=command[command.index('--objective')+1]
                epochs[objective]=int(command[command.index('--epochs')+1])
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            output=Path(tmp)/'new';foundation=Path(tmp)/'foundation'
            foundation.mkdir();output.mkdir()
            record={'files':[],'cells':148727,'species':['d_rerio','h_sapiens','m_murinus','m_musculus','x_laevis'],'architecture':[300,384,384]}
            # Existing complete checkpoints trigger additional preflight metadata
            # checks, which are tested by real foundation/trainer integration.
            if already_complete:
                from jax_saturn.distributed.checkpoint import directory_sha256
                for objective in OBJECTIVES:
                    checkpoint=output/objective/'checkpoints/epoch_0030'
                    checkpoint.mkdir(parents=True)
                    (checkpoint/'metadata.json').write_text(json.dumps({'hyperparameters':{
                        'label_free_artifact_sha256':'artifact','pretrain_checkpoint_sha256':'pretrain'}}))
            replacements={
                'validate_foundation':lambda path:record,'stage_foundation':lambda *a:None,
                'validate_report_inputs':lambda path:{'cells':148727,'files':[]},
                'committed_epoch':epoch,'trial_complete':lambda path,objective:epochs[objective]==30,
                'validate_trial_resume':lambda *a:None,'run_process':child,'inventory':lambda path:[],
                'verify_outputs':lambda output:{'scientific_acceptance':False},
                'remaining_estimate':lambda *a,**k:{'required_seconds':10 if passes else 999999},
            }
            for name,value in replacements.items():stack.enter_context(patch.object(runner,name,value))
            stack.enter_context(patch('jax_saturn.data.neighbors.NeighborSearch.validate_device'))
            stack.enter_context(patch('jax.devices',return_value=['fixture_gpu']))
            stack.enter_context(patch('jax_saturn.distributed.checkpoint.load_pretrain_metric_params'))
            stack.enter_context(patch('jax_saturn.distributed.checkpoint.directory_sha256',return_value='pretrain'))
            stack.enter_context(patch('jax_saturn.data.cache.file_sha256',return_value='artifact'))
            baseline=stack.enter_context(patch('jax_saturn.training.baseline.train_baseline',side_effect=AssertionError('must skip')))
            pretrain=stack.enter_context(patch('jax_saturn.training.pretrain.train_pretrain',side_effect=AssertionError('must skip')))
            # This script is imported lazily by the runner from its scripts dir.
            import sys
            stack.enter_context(patch.dict(sys.modules,{'package_saturn_source':script('package_saturn_source')}))
            stack.enter_context(patch('sys.argv',['run_hmmff_gpu.py','run','--foundation',str(foundation),'--output',str(output),
                '--cache',str(Path(tmp)/'cache'),'--vm-start-time',datetime.now(timezone.utc).isoformat()]))
            stack.enter_context(patch('sys.stdout',new_callable=io.StringIO))
            status=runner.main()
            pretrain.assert_not_called();baseline.assert_not_called()
            return status,calls,epochs

    def test_one_launch_runs_two_real_epochs_then_resumes(self):
        status,calls,epochs=self.exercise()
        self.assertEqual(status,0)
        training=[c for c in calls if '--objective' in c]
        self.assertEqual([int(c[c.index('--epochs')+1]) for c in training],[2,2,2,30,30,30])
        self.assertEqual([c[c.index('--objective')+1] for c in training],list(OBJECTIVES)*2)
        for command in training[3:]:
            self.assertIn('--resume',command)
            self.assertTrue(command[command.index('--resume')+1].endswith('epoch_0002'))
        self.assertEqual(epochs,dict.fromkeys(OBJECTIVES,30))
        self.assertFalse(any('gcloud' in item for command in calls for item in command))

    def test_gate_preserves_production_epochs_and_stops_before_more_training(self):
        status,calls,epochs=self.exercise(passes=False)
        self.assertEqual(status,3);self.assertEqual(len(calls),3)
        self.assertEqual(epochs,dict.fromkeys(OBJECTIVES,2))

    def test_completed_epochs_are_not_repeated(self):
        status,calls,epochs=self.exercise(already_complete=True)
        self.assertEqual(status,0)
        self.assertEqual(len(calls),1)
        self.assertIn('execute_saturn_notebook.py',calls[0][1])

    def test_timing_gate_counts_remaining_epochs_and_compilation(self):
        runner=script('run_hmmff_gpu')
        with patch.object(runner,'committed_epoch',return_value=2),patch.object(runner,'read_events',return_value=[
            {'event':'epoch_complete','epoch':1,'seconds':130.,'first_update_seconds':100.,'invocation_first_epoch':True},
            {'event':'epoch_complete','epoch':2,'seconds':30.,'first_update_seconds':1.,'invocation_first_epoch':False}]):
            result=runner.remaining_estimate(Path('/fixture'),setup_reserve=100,final_reserve=200)
            self.assertEqual(result['remaining_epoch_seconds'],28*30*3)
            self.assertEqual(result['remaining_compilation_seconds'],300)
            self.assertEqual(result['required_seconds'],1.3*(100+200+300+28*30*3))

    def test_backup_skips_uncommitted_and_verifies_download(self):
        module=script('backup_saturn_run')
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source';dest=Path(tmp)/'snapshot'
            good=source/'infonce/checkpoints/epoch_0002';bad=source/'infonce/checkpoints/epoch_0003'
            good.mkdir(parents=True);bad.mkdir(parents=True)
            (good/'metadata.json').write_text('{}');(good/'params').write_text('saved')
            (bad/'params').write_text('partial')
            (source/'executed.ipynb').write_text('partial notebook')
            records=module.snapshot(source,dest)
            self.assertFalse(any('epoch_0003' in row['path'] for row in records))
            self.assertTrue(any('executed.ipynb' in row['path'] for row in records))
            module.verify_download(dest)
            (dest/'executed.ipynb').write_text('corrupt')
            with self.assertRaises(ValueError):module.verify_download(dest)

if __name__=='__main__':unittest.main()

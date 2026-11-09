"""Notebook failure recovery and immutable static report caches."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import anndata as ad
import nbformat
import numpy as np
import psutil

from jax_saturn.data.cache import file_sha256
from jax_saturn.experiments.report_cache import pretrained_view

ROOT=Path(__file__).resolve().parents[2]


class NotebookExecutionTests(unittest.TestCase):
    def test_streams_partial_notebook_failure_and_cleanup(self):
        spec=importlib.util.spec_from_file_location('execute_saturn_notebook',ROOT/'scripts/execute_saturn_notebook.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source.ipynb';output=Path(tmp)/'partial.ipynb'
            nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("print('stream-forwarded',flush=True)"),
                nbformat.v4.new_code_cell("raise ValueError('expected-test-failure')")])
            nbformat.write(nb,source)
            before={p.pid for p in psutil.Process().children(recursive=True)}
            with self.assertRaisesRegex(Exception,'expected-test-failure'):module.execute(source,output)
            partial=nbformat.read(output,as_version=4)
            self.assertEqual(partial.cells[0].outputs[0].text.strip(),'stream-forwarded')
            self.assertEqual(partial.cells[1].outputs[0].output_type,'error')
            self.assertEqual({p.pid for p in psutil.Process().children(recursive=True)},before)

    def test_pretrained_cache_preserves_original_foundation(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source.h5ad';cache=Path(tmp)/'cache'
            a=ad.AnnData(np.random.default_rng(0).normal(size=(100,12)).astype(np.float32))
            a.obs['species']=['a']*50+['b']*50
            a.obs['labels2']=[str(i%4) for i in range(100)]
            a.write_h5ad(source);before=file_sha256(source)
            first,metrics=pretrained_view(source,cache_dir=cache)
            second,other=pretrained_view(source,cache_dir=cache)
            np.testing.assert_array_equal(first.obsm['X_umap'],second.obsm['X_umap'])
            self.assertEqual(metrics,other);self.assertEqual(file_sha256(source),before)

if __name__=='__main__':unittest.main()

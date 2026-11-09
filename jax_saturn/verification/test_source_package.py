"""Deployment archives retain Python data modules while excluding biological data."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class SourcePackageTests(unittest.TestCase):
    def test_archive_contains_all_runtime_python_including_data_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'source.tar.gz'
            subprocess.run([sys.executable, str(ROOT/'scripts/package_saturn_source.py'),
                            '--output', str(archive)], check=True, capture_output=True, text=True)
            with tarfile.open(archive) as bundle:
                names = set(bundle.getnames())
                python_source = {p.relative_to(ROOT).as_posix() for p in (ROOT/'jax_saturn').rglob('*.py')}
                self.assertTrue(python_source <= names, sorted(python_source - names))
                self.assertIn('jax_saturn/data/neighbors.py', names)
                self.assertIn('jax_saturn/data/work_cache.py', names)
                for name in ('disease_pilot/cloud.py', 'disease_pilot/report.py',
                             'experiments/hmmff_rdeb_skin_cell_holdout.env',
                             'notebooks/jax/human_monkey_mouse_frog_fish_rdeb_skin_holdout.ipynb'):
                    self.assertIn(name, names)
                self.assertFalse(any(Path(name).parts[0] in {'data','out','dist'} for name in names))
                record = json.load(bundle.extractfile('source_manifest.json'))
                self.assertEqual(names, {row['path'] for row in record['files']} | {'source_manifest.json'})
                for member in bundle.getmembers():
                    self.assertFalse(Path(member.name).is_absolute())
                    self.assertNotIn('..', Path(member.name).parts)
                    self.assertTrue(member.isfile() or member.isdir())
                installation = root/'installed'
                bundle.extractall(installation)
            self.assertFalse((installation/'.git').exists())
            subprocess.run([sys.executable,str(installation/'scripts/package_saturn_source.py'),
                            '--verify-tree',str(installation/'source_manifest.json')],
                           check=True,capture_output=True,text=True,cwd=installation)

    def test_old_incomplete_manifest_is_rejected(self):
        spec=importlib.util.spec_from_file_location('source_package',ROOT/'scripts/package_saturn_source.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source_manifest.json'
            path.write_text(json.dumps({'files':[{'path':'scripts/run_hmmff_gpu.py'}]}))
            with self.assertRaisesRegex(ValueError,'missing required runtime files'):
                module.verify_tree(path)


if __name__ == '__main__':
    unittest.main()

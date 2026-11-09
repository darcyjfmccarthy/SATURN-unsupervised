#!/usr/bin/env python3
"""Check documented shell syntax, new CLI flags and notebook Python cells locally."""

import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    docs=[ROOT/'GCP_RUNBOOK.md', ROOT/'jax_saturn/EXPERIMENT_WORKFLOW.md',ROOT/'notebooks/jax/GCP_FIVE_SPECIES_RUNBOOK.md', ROOT/'notebooks/local/DISEASE_EXPERIMENT.md', ROOT/'notebooks/local/README.md']
    count=python_count=0
    for path in docs:
        for code in re.findall(r'```bash\n(.*?)```',path.read_text(),re.S):
            subprocess.run(['bash','-n'],input=code,text=True,check=True,capture_output=True);count+=1
            for source in re.findall(r"<<['\"]?PY['\"]?\s*\n(.*?)^PY\s*$",code,re.S|re.M):
                compile(source,str(path),'exec');python_count+=1
        for source in re.findall(r'```python\n(.*?)```',path.read_text(),re.S):
            compile(source,str(path),'exec');python_count+=1
        for target in re.findall(r'\]\(([^)#]+)(?:#[^)]*)?\)',path.read_text()):
            if not target.startswith(('http:','https:')) and not (path.parent/target).exists():raise ValueError(f'Broken link in {path}: {target}')
    notebooks = ['human_monkey_mouse_frog_fish.ipynb', 'human_monkey_mouse_frog_fish_holdout.ipynb', 'human_monkey_mouse_frog_fish_rdeb_skin_holdout.ipynb']
    cell_count = 0
    for name in notebooks:
        nb=json.loads((ROOT/'notebooks/jax'/name).read_text())
        for i,cell in enumerate(nb['cells']):
            if cell['cell_type']=='code':compile(''.join(cell['source']),f'{name} cell {i}','exec')
        cell_count += len(nb['cells'])
    subprocess.run(['bash', '-n', str(ROOT/'scripts/bootstrap_cell_holdout_vm.sh')], check=True)
    scripts=('run_hmmff_gpu.py','train_label_agnostic_jax.py','evaluate_label_agnostic_benchmark.py',
                   'execute_saturn_notebook.py','package_saturn_source.py','record_saturn_inputs.py',
                   'stage_saturn_inputs.py','backup_saturn_run.py','validate_gpu_exact_real.py',
                   'preflight_saturn_jax.py','train_saturn_jax.py','prepare_label_agnostic_artifacts.py',
                   'run_cell_holdout_jax.py','prepare_disease_cloud_inputs.py','validate_disease_holdout.py')
    for script in scripts:
        result=subprocess.run([sys.executable,str(ROOT/'scripts'/script),'--help'],check=True,capture_output=True,text=True)
        if 'usage:' not in result.stdout:raise AssertionError(script)
    print(f'PASS: {count} shell examples, {python_count} Python examples, {cell_count} notebook cells, {len(scripts)} CLIs and guide links.')

if __name__=='__main__':main()

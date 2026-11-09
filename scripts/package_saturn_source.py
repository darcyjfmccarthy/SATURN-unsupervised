#!/usr/bin/env python3
"""Package working source, including new files, without data, runs or credentials."""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[1]
REQUIRED_SOURCE = {
    'jax_saturn/data/__init__.py', 'jax_saturn/data/batching.py',
    'jax_saturn/data/cache.py', 'jax_saturn/data/graphs.py',
    'jax_saturn/data/manifest.py', 'jax_saturn/data/neighbors.py',
    'jax_saturn/data/preparation.py', 'jax_saturn/data/work_cache.py',
    'jax_saturn/experiments/hmmff.py', 'jax_saturn/training/label_agnostic.py',
    'scripts/run_hmmff_gpu.py', 'scripts/train_label_agnostic_jax.py',
    'jax_saturn/data/holdout.py', 'jax_saturn/experiments/cell_holdout.py',
    'jax_saturn/experiments/holdout_report.py', 'scripts/run_cell_holdout_jax.py',
    'jax_saturn/experiments/holdout_diagnostics.py',
    'scripts/bootstrap_cell_holdout_vm.sh', 'experiments/hmmff_cell_holdout.env',
    'notebooks/jax/human_monkey_mouse_frog_fish_holdout.ipynb',
}


def require_runtime_files(paths):
    missing = REQUIRED_SOURCE - set(paths)
    if missing:
        raise ValueError(f'Source package is missing required runtime files: {sorted(missing)}. Rebuild with the corrected packager.')


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def source_files():
    result=subprocess.run(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT,check=True,capture_output=True)
    root_excluded={'data','out','dist'}
    excluded={'.git','.agents','.codex','.aws','.cache','.ssh','.gcloud','.config'}
    binary={'.h5ad','.pt','.npz','.pkl','.pyc','.pem','.key'}
    paths=[]
    for name in sorted(set(result.stdout.decode().split('\0'))- {''}):
        path=Path(name)
        if path.parts[0] in root_excluded:continue
        if any(part in excluded or part.startswith('.venv') or part=='__pycache__' for part in path.parts):continue
        if path.suffix in binary or path.name=='source_manifest.json':continue
        if path.name.startswith('.env') or any(word in path.name.lower() for word in ('credential','service-account','service_account','private_key')):continue
        actual=ROOT/path
        if actual.is_symlink():raise ValueError(f'Unexpected source symlink: {path}')
        if actual.is_file():paths.append(path)
    require_runtime_files(p.as_posix() for p in paths)
    return paths


def manifest():
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,check=True,capture_output=True,text=True).stdout.strip()
    return {'schema_version':1,'git_parent':commit,'includes_working_tree_edits_and_new_files':True,
        'files':[{'path':p.as_posix(),'size':(ROOT/p).stat().st_size,'sha256':digest(ROOT/p)} for p in source_files()]}


def verify_tree(path):
    record=json.loads(Path(path).read_text())
    require_runtime_files(row['path'] for row in record['files'])
    for row in record['files']:
        relative=Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe source manifest path')
        file=ROOT/relative
        if file.stat().st_size!=row['size'] or digest(file)!=row['sha256']:raise ValueError(f'Source checksum differs: {relative}')
    print(f'Verified {len(record["files"])} source files.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'dist/saturn-gpu-exact.tar.gz')
    parser.add_argument('--verify-tree',type=Path)
    args=parser.parse_args()
    if args.verify_tree:
        verify_tree(args.verify_tree);return
    record=manifest();encoded=(json.dumps(record,indent=2,sort_keys=True)+'\n').encode()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as compressed,tarfile.open(fileobj=compressed,mode='w') as archive:
        for row in record['files']:
            path=ROOT/row['path'];info=archive.gettarinfo(str(path),arcname=row['path'])
            info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0
            with path.open('rb') as handle:archive.addfile(info,handle)
        info=tarfile.TarInfo('source_manifest.json');info.size=len(encoded);info.mtime=0;info.mode=0o644
        archive.addfile(info,io.BytesIO(encoded))
    args.output.with_name('source_manifest.json').write_bytes(encoded)
    args.output.with_name(args.output.name+'.sha256').write_text(digest(args.output)+'  '+args.output.name+'\n')
    print(f'{args.output}: {len(record["files"])} source files, SHA256 {digest(args.output)}')

if __name__=='__main__':main()

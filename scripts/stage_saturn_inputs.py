#!/usr/bin/env python3
"""Upload/download only the atlas and embedding files named in a species CSV."""

import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def commands(manifest,bucket,direction,bundle_record=None):
    rows=list(csv.DictReader(manifest.open()))
    if not rows or len({r['species'] for r in rows})!=len(rows):raise ValueError('Species must be unique')
    paths=[Path(r[key]) for r in rows for key in ('path','embedding_path')]
    if len(set(paths))!=2*len(rows):raise ValueError('Each species must name distinct input files')
    transfer_paths = [manifest.relative_to(ROOT), *paths]
    if bundle_record is not None:
        transfer_paths += [bundle_record.relative_to(ROOT)]
        transfer_paths += [Path(entry['path']) for entry in json.loads(bundle_record.read_text())['files']]
    transfer_paths = list(dict.fromkeys(transfer_paths))
    for path in transfer_paths:
        if path.is_absolute() or '..' in path.parts or path.parts[0]!='data':raise ValueError(f'Unsafe input path: {path}')
    bucket=bucket.removeprefix('gs://').rstrip('/')
    for path in transfer_paths:
        remote=f'gs://{bucket}/inputs/{path.relative_to("data").as_posix()}'
        local=ROOT/path
        if direction=='upload':
            if not local.is_file():raise FileNotFoundError(local)
            yield ['gcloud','storage','cp',str(local),remote]
        else:
            local.parent.mkdir(parents=True,exist_ok=True)
            yield ['gcloud','storage','cp',remote,str(local)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True,help='Local CSV defining exactly the required species')
    parser.add_argument('--bucket',required=True)
    parser.add_argument('--direction',choices=['upload','download'],required=True)
    parser.add_argument('--execute',action='store_true',help='Otherwise print commands only')
    parser.add_argument('--bundle-record', type=Path, help='Also stage the explicitly inventoried report/provenance files')
    args=parser.parse_args()
    env={**os.environ,'CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_ENABLED':'False'}
    for command in commands(args.manifest.resolve(),args.bucket,args.direction,
                            args.bundle_record.resolve() if args.bundle_record else None):
        print(shlex.join(command),flush=True)
        if args.execute:subprocess.run(command,check=True,cwd=ROOT,env=env)

if __name__=='__main__':main()

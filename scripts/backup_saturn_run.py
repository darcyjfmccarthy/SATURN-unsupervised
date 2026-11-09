#!/usr/bin/env python3
"""Snapshot committed checkpoints and partial notebooks; upload every five minutes."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from jax_saturn.data.cache import file_sha256
from jax_saturn.experiments.hmmff import atomic_json


def snapshot(source,destination):
    source,destination=Path(source),Path(destination)
    records=[]
    for path in sorted(source.rglob('*')):
        if not path.is_file():continue
        relative=path.relative_to(source)
        if any(part.startswith('.') or part.endswith(('.tmp','.staging')) for part in relative.parts):continue
        # Never advertise an Orbax directory until its metadata commit marker
        # exists. Components of a published checkpoint are immutable.
        committed=True;immutable=False
        for i,part in enumerate(relative.parts):
            if part.startswith('epoch_') and i and relative.parts[i-1] in {'checkpoints','selected_orbax'}:
                checkpoint=source.joinpath(*relative.parts[:i+1])
                committed=(checkpoint/'metadata.json').is_file()
                immutable=committed
        if not committed:continue
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if immutable:
            try:os.link(path,target)
            except OSError:shutil.copy2(path,target)
        else:shutil.copy2(path,target)
        checksum=file_sha256(target)
        if not immutable and checksum!=file_sha256(path):
            raise RuntimeError(f'File changed during snapshot; retry next cycle: {relative}')
        records.append({'path':relative.as_posix(),'size':target.stat().st_size,'sha256':checksum})
    atomic_json(destination/'backup_manifest.json',{'schema_version':1,'snapshot_time_unix':time.time(),'files':records})
    return records


def backup(run_dir,prefix):
    env={**os.environ,'CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_ENABLED':'False'}
    with tempfile.TemporaryDirectory(prefix='saturn-backup-') as tmp:
        records=snapshot(run_dir,Path(tmp))
        command=['gcloud','storage','rsync','--recursive','--checksums-only',tmp,prefix]
        subprocess.run(command,check=True,env=env)
        verification=subprocess.run([*command,'--dry-run'],check=True,env=env,text=True,capture_output=True)
        output=verification.stdout+verification.stderr
        print(output,end='',flush=True)
        if re.search(r'would\s+(copy|delete|update)|copying\s|deleting\s',output,re.IGNORECASE):
            raise RuntimeError('Bucket checksum verification reports differences; retry backup')
        print(json.dumps({'event':'backup_verified','files':len(records),'time_unix':time.time(),'prefix':prefix}),flush=True)


def verify_download(root):
    root=Path(root)
    records=json.loads((root/'backup_manifest.json').read_text())['files']
    for row in records:
        relative=Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe backup path')
        path=root/relative
        if path.stat().st_size!=row['size'] or file_sha256(path)!=row['sha256']:
            raise ValueError(f'Download checksum differs: {path}')
    print(f'Verified downloaded backup: {len(records)} files.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--bucket-prefix')
    parser.add_argument('--loop',action='store_true')
    parser.add_argument('--verify-download',action='store_true')
    args=parser.parse_args()
    if args.verify_download:verify_download(args.run_dir);return
    if not args.bucket_prefix or not args.bucket_prefix.startswith('gs://'):parser.error('--bucket-prefix must be gs://...')
    while True:
        started=time.monotonic()
        try:backup(args.run_dir,args.bucket_prefix)
        except Exception as error:
            print(f'BACKUP FAILED: {error}',file=sys.stderr,flush=True)
            if not args.loop:raise
        if not args.loop:break
        # Five-minute start cadence. A long transfer starts the next cycle as
        # soon as it finishes; backups never run concurrently.
        time.sleep(max(1,300-(time.monotonic()-started)))

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Record actual atlas/protein file contents and ordered observation identities."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from jax_saturn.data.cache import file_sha256
from jax_saturn.experiments.hmmff import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--split',type=Path,help='CSV with species, obs_id, split; use stable source observation IDs')
    args=parser.parse_args()
    import anndata as ad
    rows=list(csv.DictReader(args.manifest.open()))
    if not rows or len({r['species'] for r in rows})!=len(rows):raise ValueError('Manifest species must be nonempty and unique')
    records=[]
    for row in rows:
        atlas_path=ROOT/row['path'];embedding_path=ROOT/row['embedding_path']
        atlas=ad.read_h5ad(atlas_path,backed='r')
        obs=list(map(str,atlas.obs_names))
        if not atlas.obs_names.is_unique:raise ValueError(f'Nonunique observation IDs in {atlas_path}')
        records.append({'species':row['species'],'atlas':{'path':row['path'],'sha256':file_sha256(atlas_path),'size':atlas_path.stat().st_size},
            'protein_embeddings':{'path':row['embedding_path'],'sha256':file_sha256(embedding_path),'size':embedding_path.stat().st_size},
            'cells':atlas.n_obs,'genes':atlas.n_vars,'ordered_observation_ids_sha256':hashlib.sha256(json.dumps(obs,separators=(',',':')).encode()).hexdigest(),
            'ordered_gene_ids_sha256':hashlib.sha256(json.dumps(list(map(str,atlas.var_names)),separators=(',',':')).encode()).hexdigest()})
        atlas.file.close()
    split=None
    if args.split:
        entries=list(csv.DictReader(args.split.open()))
        if not entries or not {'species','obs_id','split'}.issubset(entries[0]):raise ValueError('Split requires species,obs_id,split columns')
        pairs=[(r['species'],r['obs_id']) for r in entries]
        if len(set(pairs))!=len(pairs):raise ValueError('Split observation IDs repeat')
        split={'path':str(args.split),'sha256':file_sha256(args.split),'rows':len(entries),'counts':{name:sum(r['split']==name for r in entries) for name in sorted({r['split'] for r in entries})}}
    atomic_json(args.output,{'schema_version':1,'manifest':str(args.manifest),'manifest_sha256':file_sha256(args.manifest),
        'inputs':records,'split':split,'scope':'Content hashes captured now; no claim about missing historical fit-time hashes'})
    print(f'Recorded actual input hashes for {len(records)} species: {args.output}')

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Required local real-data accuracy checks; never launches cloud training."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from jax_saturn.data.neighbors import NeighborSearch,normalize
from jax_saturn.data.cache import file_sha256
from jax_saturn.contracts.validation import load_npz
from jax_saturn.experiments.hmmff import FOUNDATION,atomic_json
from jax_saturn.data.graphs import build_preservation_graph,topology_recall_at_50
from label_agnostic.metrics import embedding_benchmark_metrics


def compare(query,reference,k,search,*,self_indices=None):
    cpu=NeighborSearch('sklearn').search(query,reference,k,self_indices=self_indices)
    gpu=search.search(query,reference,k,self_indices=self_indices)
    distance_error=float(np.max(np.abs(cpu[0]-gpu[0])))
    if distance_error>1e-5:raise AssertionError(f'Distance error {distance_error}')
    differing=np.flatnonzero(np.any(cpu[1]!=gpu[1],axis=1))
    genuine=[];ties=[]
    q,r=normalize(query),normalize(reference)
    for row in differing:
        only_cpu=np.setdiff1d(cpu[1][row],gpu[1][row])
        only_gpu=np.setdiff1d(gpu[1][row],cpu[1][row])
        if not len(only_cpu):
            # Internal order differences must also be near equal distances.
            mismatch=cpu[1][row]!=gpu[1][row]
            differences=np.abs((1-r[cpu[1][row][mismatch]]@q[row])-(1-r[gpu[1][row][mismatch]]@q[row]))
        else:
            differences=np.abs(np.sort(1-r[only_cpu]@q[row])-np.sort(1-r[only_gpu]@q[row]))
        (ties if np.max(differences,initial=0)<=1e-5 else genuine).append(int(row))
    if genuine:raise AssertionError(f'Non-tied neighbor disagreements: {genuine}')
    return {'queries':len(query),'complete_reference_rows':len(reference),'k':k,'max_distance_error':distance_error,
            'exact_index_rows':len(query)-len(differing),'tie_or_roundoff_rows':ties,'non_tied_mismatch_rows':genuine,
            'tie_classification_distance_tolerance':1e-5}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact',type=Path,default=ROOT/FOUNDATION/'shared/label_free_artifact.npz')
    parser.add_argument('--output',type=Path,default=ROOT/'out/gpu_exact_validation/real_data.json')
    parser.add_argument('--query-block',type=int,default=1024)
    parser.add_argument('--reference-block',type=int,default=8192)
    args=parser.parse_args()
    values=load_npz(args.artifact,'label_free')
    search=NeighborSearch('jax',args.query_block,args.reference_block,'gpu')
    rng=np.random.default_rng(0)
    x,s=values['embeddings'],values['species']
    results=[];start=time.monotonic()
    query_ids=np.sort(rng.choice(len(x),64,replace=False))
    results.append({'population':'joint',**compare(x[query_ids],x,50,search,self_indices=query_ids)})
    for species in np.unique(s):
        ids=np.flatnonzero(s==species)
        local=np.sort(rng.choice(len(ids),min(32,len(ids)),replace=False))
        results.append({'population':str(species),**compare(x[ids[local]],x[ids],50,search,self_indices=local)})
    # Representative metric comparisons on a declared validation subset, never
    # substitute these subsets for the all-cell production experiment.
    selected=np.sort(np.concatenate([rng.choice(np.flatnonzero(s==name),400,replace=False) for name in np.unique(s)]))
    subset=x[selected];species=s[selected]
    import anndata as ad
    truth=ad.read_h5ad(args.artifact.parents[1]/'baseline/saturn_results/adata_pretrain.h5ad',backed='r')
    if not np.array_equal(np.asarray(truth.obs_names,str),values['obs_ids']):raise AssertionError('Truth order differs')
    labels=np.asarray(truth.obs['labels2'],str)[selected]
    truth.file.close()
    cpu=embedding_benchmark_metrics(subset,species,labels)
    gpu=embedding_benchmark_metrics(subset,species,labels,search=search)
    differences={key:abs(cpu[key]-gpu[key]) for key in cpu}
    teacher=build_preservation_graph(subset,species)[3]
    differences['teacher_top15_recall_at_50']=abs(topology_recall_at_50(subset,species,teacher)-topology_recall_at_50(subset,species,teacher,search=search))
    if max(differences.values())>1e-4:raise AssertionError(differences)
    result={'passed':True,'artifact':str(args.artifact),'artifact_sha256':file_sha256(args.artifact),
        'population_shape':list(x.shape),'search':search.identity(),'search_checks':results,
        'metric_fixture':'400 real cells/species with original labels2; all reference rows for sampled neighbor queries',
        'metric_absolute_differences':differences,'seconds':time.monotonic()-start}
    atomic_json(args.output,result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()

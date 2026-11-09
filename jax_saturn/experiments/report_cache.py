"""Content-keyed post-hoc caches, separate from label-free training artifacts."""

import importlib.metadata

import numpy as np

from jax_saturn.data.graphs import joint_neighbors
from jax_saturn.data.work_cache import WorkCache, array_identity


def cached_neighbors(embeddings, *, k=15, search=None, cache_dir=None):
    cache=WorkCache(cache_dir)
    identity={'content':array_identity({'embeddings':np.asarray(embeddings,np.float32)}),
              'k':k,'search':search.identity() if search else None,'metric_policy':'joint-v1'}
    return cache.get('report_neighbors',identity,lambda:{'indices':joint_neighbors(embeddings,k,search=search)})['indices']


def pretrained_view(path, *, seed=0, search=None, cache_dir=None):
    """Original scanpy visualization, cached outside the unchanged source h5ad."""
    import anndata as ad
    import scanpy as sc
    from label_agnostic.metrics import embedding_benchmark_metrics
    atlas=ad.read_h5ad(path)
    identity={'content':array_identity({'embeddings':np.asarray(atlas.X,np.float32),
        'species':np.asarray(atlas.obs['species'],str),'labels':np.asarray(atlas.obs['labels2'],str)}),
        'seed':seed,'scanpy':importlib.metadata.version('scanpy'),'umap':importlib.metadata.version('umap-learn')}
    cache=WorkCache(cache_dir)
    def coordinates():
        if 'X_umap' not in atlas.obsm:
            sc.pp.neighbors(atlas,n_neighbors=15,metric='cosine',random_state=seed)
            sc.tl.umap(atlas,min_dist=.3,random_state=seed)
        return {'coordinates':np.asarray(atlas.obsm['X_umap'],np.float32)}
    atlas.obsm['X_umap']=cache.get('pretrained_umap',{**identity,'policy':'scanpy-default-pca50-cosine15-min_dist0.3-v1'},coordinates)['coordinates']
    def metrics():
        values=embedding_benchmark_metrics(np.asarray(atlas.X),np.asarray(atlas.obs['species'],str),
            np.asarray(atlas.obs['labels2'],str),k=15,search=search,
            neighbors=cached_neighbors(atlas.X,k=15,search=search,cache_dir=cache_dir))
        return {key:np.asarray(value) for key,value in values.items()}
    values=cache.get('pretrained_metrics',{**identity,'search':search.identity() if search else None,'policy':'metrics15-v2'},metrics)
    return atlas,{key:float(value) for key,value in values.items()}

"""Exhaustive blocked cosine search. JAX is imported only when requested.

GPU policy: host float32 normalization, HIGHEST float32 dot products, cosine
distance clipped to [0, 2], ties ordered by reference index. Explicit self IDs
are excluded, including for duplicate and zero vectors. No approximate search.
"""

from dataclasses import dataclass
from functools import lru_cache
import time

import numpy as np
from sklearn.neighbors import NearestNeighbors

POLICY = "cosine-fp32-highest-clip-index-ties-self-exclude-v1"
_EXECUTABLES = {}


def normalize(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


@lru_cache(maxsize=64)
def _jax_merge(k):
    import jax
    import jax.numpy as jnp

    @jax.jit
    def merge(query, reference, offset, self_ids, best_d, best_i, valid_reference):
        similarity = jnp.matmul(query, reference.T, precision=jax.lax.Precision.HIGHEST)
        distances = jnp.clip(1.0 - similarity, 0.0, 2.0)
        indices = jnp.broadcast_to(jnp.arange(reference.shape[0], dtype=jnp.int32) + offset, distances.shape)
        distances = jnp.where((indices == self_ids[:, None]) | ~valid_reference[None, :], jnp.inf, distances)
        distances = jnp.concatenate((best_d, distances), axis=1)
        indices = jnp.concatenate((best_i, indices), axis=1)
        # Exact top_k orders equal values by their lower operand position.
        # Earlier blocks have lower reference indices; best_i is already in
        # distance/index order. Thus concatenating best before the next block
        # resolves ties by reference index without perturbing distances.
        scores, positions = jax.lax.top_k(-distances, k)
        return -scores, jnp.take_along_axis(indices, positions, axis=1)
    return merge


@dataclass(frozen=True)
class NeighborSearch:
    backend: str = "sklearn"
    query_block: int = 1024
    reference_block: int = 8192
    device: str = "gpu"

    def __post_init__(self):
        if self.backend not in {"sklearn", "jax"}:
            raise ValueError("Neighbor backend must be sklearn or jax")
        if self.query_block < 1 or self.reference_block < 1:
            raise ValueError("Neighbor block sizes must be positive")
        if self.device not in {"cpu", "gpu"}:
            raise ValueError("Neighbor device must be cpu or gpu")

    def identity(self):
        from dataclasses import asdict
        import importlib.metadata
        return {**asdict(self), "numerical_policy": POLICY,
                "library_version": importlib.metadata.version("jax" if self.backend == "jax" else "scikit-learn")}

    def validate_device(self):
        if self.backend == "jax":
            import jax
            # Never turn a missing CUDA installation into an unnoticed CPU run.
            devices = jax.devices(self.device)
            if not devices:
                raise RuntimeError(f"No requested JAX {self.device} device")
            return devices[0]

    def search(self, query, reference, k, *, self_indices=None, normalized=False):
        query = np.asarray(query, dtype=np.float32)
        reference = np.asarray(reference, dtype=np.float32)
        if query.ndim != 2 or reference.ndim != 2 or query.shape[1] != reference.shape[1]:
            raise ValueError("Search requires two matrices of equal feature dimension")
        if not np.isfinite(query).all() or not np.isfinite(reference).all():
            raise ValueError("Nonfinite neighbor inputs")
        self_ids = np.full(len(query), -1, np.int32) if self_indices is None else np.asarray(self_indices, np.int32)
        if self_ids.shape != (len(query),) or np.any(self_ids < -1) or np.any(self_ids >= len(reference)):
            raise ValueError("Invalid self reference indices")
        available = len(reference) - int(np.any(self_ids >= 0))
        if not isinstance(k, (int, np.integer)) or not 0 <= k <= available:
            raise ValueError("Neighbor k exceeds the available reference population")
        if k == 0 or len(query) == 0:
            return np.empty((len(query), k), np.float32), np.empty((len(query), k), np.int64)
        if self.backend == "sklearn":
            # The reference uses sklearn's own normalization and distances.
            count = min(k + int(np.any(self_ids >= 0)), len(reference))
            model = NearestNeighbors(n_neighbors=count, metric="cosine").fit(reference)
            distances, indices = model.kneighbors(query)
            if np.any(self_ids >= 0):
                keep = indices != self_ids[:, None]
                positions = np.argsort(~keep, axis=1, kind="stable")[:, :k]
                distances = np.take_along_axis(distances, positions, axis=1)
                indices = np.take_along_axis(indices, positions, axis=1)
            return distances[:, :k].astype(np.float32), indices[:, :k]
        import jax
        import jax.numpy as jnp
        device = self.validate_device()
        if not normalized:
            query, reference = normalize(query), normalize(reference)
        result_d = np.empty((len(query), k), np.float32)
        result_i = np.empty((len(query), k), np.int64)
        merge = _jax_merge(k)
        with jax.default_device(device):
            # Reference lives on device once; only a bounded distance tile and
            # a q-block x k accumulator exist, never a cell-by-cell matrix.
            reference_device = jax.device_put(reference, device)
            for start in range(0, len(query), self.query_block):
                size = min(self.query_block, len(query) - start)
                q = jnp.asarray(np.pad(query[start:start + size], ((0, self.query_block - size), (0, 0))))
                ids = jnp.asarray(np.pad(self_ids[start:start + size], (0, self.query_block - size), constant_values=-1))
                best_d = jnp.full((self.query_block, k), jnp.inf, jnp.float32)
                best_i = jnp.full((self.query_block, k), np.iinfo(np.int32).max, jnp.int32)
                for offset in range(0, len(reference), self.reference_block):
                    width = min(self.reference_block, len(reference) - offset)
                    block = reference_device[offset:offset + width]
                    block = jnp.pad(block, ((0, self.reference_block - width), (0, 0)))
                    valid = jnp.arange(self.reference_block) < width
                    signature = (k, tuple(q.shape), tuple(block.shape), str(device))
                    if signature not in _EXECUTABLES:
                        from jax_saturn.progress import CURRENT_PROGRESS
                        progress = CURRENT_PROGRESS.get()
                        started = time.monotonic()
                        _EXECUTABLES[signature] = merge.lower(q, block, jnp.int32(offset), ids, best_d, best_i, valid).compile()
                        if progress is not None:
                            progress.event("neighbor_compilation", k=k, dimension=q.shape[1],
                                query_block=self.query_block, reference_block=self.reference_block,
                                seconds=time.monotonic() - started)
                    best_d, best_i = _EXECUTABLES[signature](q, block, jnp.int32(offset), ids, best_d, best_i, valid)
                result_d[start:start + size], result_i[start:start + size] = jax.device_get((best_d[:size], best_i[:size]))
        return result_d, result_i


def indices(query, reference, k, *, search=None, same=False):
    """Keep the historical sklearn path byte-for-byte when no backend is set."""
    if search is None:
        count = min(k + int(same), len(reference))
        result = NearestNeighbors(n_neighbors=count, metric="cosine").fit(reference).kneighbors(query, return_distance=False)
        return result[:, 1:] if same else result
    return search.search(query, reference, k, self_indices=np.arange(len(query)) if same else None)[1]

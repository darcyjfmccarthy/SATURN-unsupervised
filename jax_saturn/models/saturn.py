"""Species-sliced Flax models preserving the reference SATURN forward behavior."""

import operator
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.core import freeze, unfreeze


class PretrainOutputs(NamedTuple):
    macrogenes: jax.Array
    embedding: jax.Array
    mu: None
    log_var: None
    px_rate: jax.Array
    px_r: jax.Array
    px_drop: jax.Array


def _uniform(bound):
    def initialize(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)
    return initialize


def _layer_norm(name):
    # Torch uses epsilon=1e-5 and centered variance. Flax's defaults differ.
    return nn.LayerNorm(epsilon=1e-5, use_fast_variance=False,
                        dtype=jnp.float32, param_dtype=jnp.float32, name=name)


class FullBlock(nn.Module):
    features: int
    dropout: float = 0.1
    dtype: object = jnp.float32
    use_bias: bool = True

    @nn.compact
    def __call__(self, values, *, train=False):
        bound = 1 / np.sqrt(values.shape[-1])
        values = nn.Dense(self.features, use_bias=self.use_bias,
                          kernel_init=_uniform(bound), bias_init=_uniform(bound),
                          dtype=self.dtype, param_dtype=jnp.float32,
                          precision=jax.lax.Precision.HIGHEST, name="dense")(values)
        values = nn.relu(_layer_norm("layer_norm")(values.astype(jnp.float32)))
        return nn.Dropout(self.dropout, name="dropout")(values, deterministic=not train)


class _Blocks(nn.Module):
    features: tuple
    dropout: float
    dtype: object = jnp.float32

    @nn.compact
    def __call__(self, values, *, train=False):
        for index, features in enumerate(self.features):
            values = FullBlock(features, self.dropout, self.dtype,
                               name=f"block_{index}")(values, train=train)
        return values


class _MacrogeneWeights(nn.Module):
    num_macrogenes: int
    total_genes: int

    @nn.compact
    def __call__(self):
        return self.param("log_gene_to_macrogene", nn.initializers.zeros,
                          (self.num_macrogenes, self.total_genes), jnp.float32)


class SpeciesDropoutDecoder(nn.Module):
    species_names: tuple
    gene_counts: tuple
    hidden_dim: int
    dtype: object = jnp.float32

    def setup(self):
        # Initialize every species head, even when init sees just one species.
        def initialize(key, count):
            kernel_key, bias_key = jax.random.split(key)
            uniform = _uniform(1 / np.sqrt(self.hidden_dim))
            return {"kernel": uniform(kernel_key, (self.hidden_dim, count)),
                    "bias": uniform(bias_key, (count,))}
        self.heads = {name: self.param(name, initialize, count)
                      for name, count in zip(self.species_names, self.gene_counts)}

    def __call__(self, values, species_code):
        head = self.heads[self.species_names[species_code]]
        result = jnp.matmul(values.astype(self.dtype), head["kernel"].astype(self.dtype),
                            precision=jax.lax.Precision.HIGHEST)
        return (result + head["bias"].astype(self.dtype)).astype(jnp.float32)


class _SpeciesDispersion(nn.Module):
    species_names: tuple
    gene_counts: tuple

    def setup(self):
        self.log_dispersion = {name: self.param(name, nn.initializers.normal(stddev=1.),
                                               (count,), jnp.float32)
                               for name, count in zip(self.species_names, self.gene_counts)}

    def __call__(self, species_code):
        return jnp.exp(self.log_dispersion[self.species_names[species_code]])


class _Decoder(nn.Module):
    species_names: tuple
    gene_counts: tuple
    hidden_dim: int
    num_macrogenes: int
    dropout: float
    dtype: object = jnp.float32

    def setup(self):
        self.px_decoder = _Blocks((self.hidden_dim,), self.dropout, self.dtype)
        # The reference does not pass its configured dropout to this block.
        self.cl_scale_decoder = _Blocks((self.num_macrogenes,), 0.1, self.dtype)
        self.px_dropout_decoders = SpeciesDropoutDecoder(
            self.species_names, self.gene_counts, self.hidden_dim, self.dtype)
        self.px_rs = _SpeciesDispersion(self.species_names, self.gene_counts)

    def __call__(self, embedding, weights, species_code, library, *, train=False):
        # First parity deliberately retains the reference's spec_idx = 0.
        covariates = jnp.broadcast_to(jax.nn.one_hot(0, len(self.species_names)),
                                     (embedding.shape[0], len(self.species_names)))
        decoded = self.px_decoder(jnp.concatenate((embedding, covariates), axis=-1), train=train)
        scale = self.cl_scale_decoder(decoded, train=train)
        logits = jnp.matmul(scale.astype(self.dtype), weights.astype(self.dtype),
                            precision=jax.lax.Precision.HIGHEST).astype(jnp.float32)
        rate = jnp.exp(library) * jax.nn.softmax(logits, axis=-1)
        return rate, self.px_rs(species_code), self.px_dropout_decoders(decoded, species_code)


def _dimensions(*values):
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("Model dimensions must be positive integers")


def _species_code(code, count):
    try:
        code = operator.index(code)
    except TypeError as error:
        raise ValueError("species_code must be a static integer (compile per species)") from error
    if not 0 <= code < count:
        raise ValueError("species_code is out of range")
    return code


class SaturnPretrainModule(nn.Module):
    species_names: tuple
    gene_counts: tuple
    num_macrogenes: int = 200
    hidden_dim: int = 256
    embed_dim: int = 256
    dropout: float = 0.1
    dtype: object = jnp.float32

    def setup(self):
        _dimensions(self.num_macrogenes, self.hidden_dim, self.embed_dim, *self.gene_counts)
        if (not self.species_names or len(self.species_names) != len(self.gene_counts)
                or any(not isinstance(name, str) or not name for name in self.species_names)
                or list(self.species_names) != sorted(set(self.species_names))):
            raise ValueError("Species names must be sorted, unique and match gene counts")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.macrogene = _MacrogeneWeights(self.num_macrogenes, sum(self.gene_counts))
        self.cl_layer_norm = _layer_norm("cl_layer_norm")
        self.encoder = _Blocks((self.hidden_dim, self.embed_dim), self.dropout, self.dtype)
        self.decoder = _Decoder(self.species_names, self.gene_counts, self.hidden_dim,
                                self.num_macrogenes, self.dropout, self.dtype)
        self.p_weights_embeddings = _Blocks((256,), self.dropout, self.dtype)

    def __call__(self, values, species_code, *, train=False):
        species_code = _species_code(species_code, len(self.species_names))
        if values.ndim != 2 or values.shape[1] != self.gene_counts[species_code]:
            raise ValueError("Expression must have shape [batch, genes_for_species]")
        log_weights = self.macrogene()
        start = sum(self.gene_counts[:species_code])
        end = start + self.gene_counts[species_code]
        weights = jnp.exp(log_weights[:, start:end])
        projected = jnp.matmul(jnp.log1p(values.astype(jnp.float32)).astype(self.dtype),
                               weights.T.astype(self.dtype), precision=jax.lax.Precision.HIGHEST)
        macrogenes = nn.relu(self.cl_layer_norm(projected.astype(jnp.float32)))
        macrogenes = self._drop_macrogenes(macrogenes, train=train)
        embedding = self.encoder(macrogenes, train=train)
        library = jnp.log(jnp.sum(values.astype(jnp.float32), axis=-1, keepdims=True))
        rate, dispersion, drop = self.decoder(embedding, weights, species_code, library, train=train)
        if self.is_initializing():
            # This block is used by the global ranking penalty, not by forward.
            self.p_weights_embeddings(jnp.exp(log_weights).T, train=False)
        return PretrainOutputs(macrogenes, embedding, None, None, rate, dispersion, drop)

    @nn.compact
    def _drop_macrogenes(self, values, *, train):
        return nn.Dropout(self.dropout, name="macrogene_dropout")(values, deterministic=not train)

    def gene_weight_embeddings(self, *, train=False):
        """Learned protein-ranking embeddings over the full gene weight matrix."""
        return self.p_weights_embeddings(jnp.exp(self.macrogene()).T, train=train)


class SaturnMetricModule(nn.Module):
    input_dim: int = 200
    hidden_dim: int = 256
    embed_dim: int = 256
    dropout: float = 0.1
    dtype: object = jnp.float32

    def setup(self):
        _dimensions(self.input_dim, self.hidden_dim, self.embed_dim)
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.cl_layer_norm = _layer_norm("cl_layer_norm")
        self.encoder = _Blocks((self.hidden_dim, self.embed_dim), self.dropout, self.dtype)

    def __call__(self, macrogenes, *, train=False):
        if macrogenes.ndim != 2 or macrogenes.shape[1] != self.input_dim:
            raise ValueError("Macrogenes must have shape [batch, input_dim]")
        if self.is_initializing():
            # Preserve checkpoint fields, but reference metric forward skips this.
            self.cl_layer_norm(macrogenes)
        return self.encoder(macrogenes, train=train)


def init_pretrain_params(module, key, gene_scores):
    """Initialize all heads and install reference centroid scores as log weights."""
    scores = np.asarray(gene_scores, dtype=np.float32)
    expected = (sum(module.gene_counts), module.num_macrogenes)
    if scores.shape != expected or not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError(f"gene_scores must be finite nonnegative values with shape {expected}")
    params = unfreeze(module.init(key, jnp.ones((2, module.gene_counts[0])), 0)["params"])
    # Exact zeros from one_hot centroids are permitted, as in Torch log().
    params["macrogene"]["log_gene_to_macrogene"] = jnp.log(jnp.asarray(scores.T))
    return freeze(params)


def pretrain_apply(module, params, batch, species_code, *, train=False, rngs=None):
    return module.apply({"params": params}, batch, species_code, train=train, rngs=rngs)


def metric_apply(module, params, macrogenes, *, train=False, rngs=None):
    return module.apply({"params": params}, macrogenes, train=train, rngs=rngs)

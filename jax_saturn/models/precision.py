"""Matmul precision policy; parameters and loss arithmetic remain float32."""

import jax.numpy as jnp


def resolve_precision(value=None, *, platform=None):
    if value is None:
        value = "bf16" if platform == "tpu" else "fp32"
    if value not in {"fp32", "bf16"}:
        raise ValueError("Mixed precision must be fp32 or bf16")
    return value


def matmul_dtype(value):
    return jnp.bfloat16 if resolve_precision(value) == "bf16" else jnp.float32

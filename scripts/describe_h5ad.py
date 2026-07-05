#!/usr/bin/env python3
"""Summarize the contents of an .h5ad (AnnData) file.

Usage:
  python scripts/describe_h5ad.py path/to/file.h5ad --json summary.json

This prints a compact summary and can dump JSON for programmatic use.
"""
import argparse
import json
import os
from collections import Counter

def summarize(adata, head=5, max_cat_values=10):
    s = {}
    s["shape"] = adata.shape
    s["n_obs"] = adata.n_obs
    s["n_vars"] = adata.n_vars
    s["obs_columns"] = list(adata.obs.columns)
    s["var_columns"] = list(adata.var.columns)
    s["obsm_keys"] = list(adata.obsm.keys())
    s["varm_keys"] = list(adata.varm.keys())
    s["layers"] = list(adata.layers.keys())
    s["uns_keys"] = list(adata.uns.keys())
    s["X_dtype"] = str(adata.X.dtype) if adata.X is not None else None
    # quick samples
    try:
        s["obs_head"] = adata.obs.head(head).to_dict(orient="list")
    except Exception:
        s["obs_head"] = None
    try:
        s["var_head"] = adata.var.head(head).to_dict(orient="list")
    except Exception:
        s["var_head"] = None
    # count unique values for categorical obs columns (limited)
    categorical = {}
    for col in adata.obs.columns:
        try:
            vals = adata.obs[col].values
            if vals.dtype.kind in ("O","S","U") or len(set(vals)) < 50:
                counts = Counter(vals)
                most = counts.most_common(max_cat_values)
                categorical[col] = {str(k): int(v) for k, v in most}
        except Exception:
            continue
    s["obs_categorical_preview"] = categorical
    return s

def human_print(summary):
    print("----- h5ad summary -----")
    print(f"Shape (obs, vars): {summary.get('shape')}")
    print(f"n_obs: {summary.get('n_obs')}, n_vars: {summary.get('n_vars')}")
    print("\nobs columns:")
    for c in summary.get("obs_columns", [])[:200]:
        print("  - ", c)
    print("\nvar columns:")
    for c in summary.get("var_columns", [])[:200]:
        print("  - ", c)
    print("\nobsm keys:", summary.get("obsm_keys"))
    print("varm keys:", summary.get("varm_keys"))
    print("layers:", summary.get("layers"))
    print("uns keys (sample):", summary.get("uns_keys")[:50] if summary.get("uns_keys") else [])
    print("\nX dtype:", summary.get("X_dtype"))
    if summary.get("obs_categorical_preview"):
        print("\nCategorical preview (up to 10 most common values):")
        for col, counts in summary["obs_categorical_preview"].items():
            print(f"  {col}: {counts}")

def main():
    p = argparse.ArgumentParser(description="Summarize an .h5ad file")
    p.add_argument("path", help="Path to .h5ad file")
    p.add_argument("--json", help="Write machine-readable JSON summary to this path")
    p.add_argument("--head", type=int, default=5, help="Rows to show from obs/var")
    args = p.parse_args()

    try:
        import anndata as ad
    except Exception as e:
        print("Missing dependency: anndata (install with `pip install anndata`).")
        raise

    if not os.path.exists(args.path):
        raise FileNotFoundError(args.path)

    adata = ad.read_h5ad(args.path)
    summary = summarize(adata, head=args.head)
    human_print(summary)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote JSON summary to {args.json}")

if __name__ == "__main__":
    main()

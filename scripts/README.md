# h5ad summarizer

Small CLI to print a quick summary of an AnnData `.h5ad` file and optionally
dump a JSON summary.

Usage
```
python scripts/describe_h5ad.py data/cell_atlases/h_sapiens.h5ad --json summary.json
```

Dependencies
- anndata

Install with:
```bash
pip install anndata
# optionally: pip install scanpy  # for extra utilities
```

What it prints
- shape (n_obs, n_vars)
- lists of `obs` and `var` columns
- `obsm`, `varm`, `layers`, `uns` keys
- small preview of categorical `obs` columns

## Label-agnostic SATURN benchmark

Run the complete four-trial GPU benchmark from the repository root:

```bash
conda run -n saturn bash scripts/run_label_agnostic_benchmark.sh
```

The default output directory is `out/label_agnostic_benchmark_clean`.
Override it with `OUT_DIR=/path/to/output`. The command trains one shared
pretraining model, runs the labeled baseline plus InfoNCE, MMD, and partial-OT
fine-tuning, evaluates all trials with one frozen post-hoc protocol, and writes
an executed comparison notebook.

For a presentation-oriented, cell-by-cell version of the same workflow, open
`notebooks/human_monkey_mouse.ipynb` with a JAX-capable kernel. It now uses the
JAX backend and defaults to a fresh run into
`out/human_monkey_mouse_jax_benchmark_walkthrough`. See
[the notebook guide](../notebooks/README.md) for dependencies, native checkpoint
paths and the smoke/report notebooks. For the JAX shell pipeline use
`VALIDATION_PROFILE=full bash scripts/run_label_agnostic_benchmark_jax.sh`;
its output directory is `out/human_monkey_mouse_jax_benchmark`.

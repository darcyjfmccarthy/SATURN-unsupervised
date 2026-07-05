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
`notebooks/label_agnostic_benchmark_walkthrough.ipynb` with the `saturn`
environment. It reuses the canonical completed run by default; set
`REBUILD_FROM_SCRATCH = True` in its configuration cell to execute all stages
into `out/label_agnostic_benchmark_walkthrough`.

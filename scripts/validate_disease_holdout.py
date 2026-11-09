#!/usr/bin/env python3
"""One-epoch CPU smoke of all five species, both methods and the disease notebook."""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'out/rdeb_validation_smoke')
    parser.add_argument('--resume', action='store_true', help='Continue this validation fixture after interruption')
    args = parser.parse_args()
    import anndata as ad
    import numpy as np
    import pandas as pd
    from scipy import sparse
    from jax_saturn.data.cache import save_cache
    from jax_saturn.experiments.cell_holdout import HoldoutConfig, prepare
    output = args.output.resolve()
    source = output / 'synthetic_inputs'
    if args.resume:
        import run_cell_holdout_jax as module
        saved = json.loads((output / 'experiment_config.json').read_text())
        saved.pop('mixed_precision')
        config = HoldoutConfig(**saved, device='cpu', neighbor_backend='jax', query_block=64, reference_block=256)
        os.environ.update(config.execution_env())
        os.environ.update(DISEASE_METADATA=str(source / 'evaluation_metadata.csv.gz'), DISEASE_INPUT_BUNDLE='',
                          REPORT_NOTEBOOK='notebooks/jax/human_monkey_mouse_frog_fish_rdeb_skin_holdout.ipynb')
        start = json.loads((output / 'execution_options.json').read_text())['vm_start_time']
        code = module.run(config, start=start)
        if code:
            raise SystemExit(code)
        print('PASS: resumed five-species CPU smoke, both methods and disease notebook')
        return
    source.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(29)
    rows = []
    for species in ('d_rerio', 'h_sapiens', 'm_murinus', 'm_musculus', 'x_laevis'):
        counts = rng.poisson(rng.uniform(.5, 12, (1, 80)), (40, 80)).astype(np.float32)
        obs = pd.DataFrame({'cellType': ['keratinocyte', 'fibroblast'] * 20, 'organ': ['skin'] * 40},
                           index=[f'{species}:cell{i}' for i in range(40)])
        atlas = ad.AnnData(sparse.csr_matrix(counts), obs=obs, var=pd.DataFrame(index=[f'G{i}' for i in range(80)]))
        path = source / f'{species}.h5ad'
        atlas.write_h5ad(path)
        protein = source / f'{species}.npz'
        save_cache(protein, {'gene_symbols': np.array([f'g{i}' for i in range(80)]),
                            'embeddings': rng.normal(size=(80, 8)).astype(np.float32)},
                   {'schema_version': 1, 'species': species, 'source_path': 'synthetic-fixture',
                    'source_sha256': 'a' * 64, 'created_by': 'validate_disease_holdout', 'gene_symbol_case': 'lower'}, kind='embedding')
        rows.append({'species': species, 'path': str(path), 'embedding_path': str(protein), 'in_label_col': 'cellType'})
        if species == 'm_musculus':
            metadata = obs.copy()
    manifest = source / 'manifest.csv'
    pd.DataFrame(rows).to_csv(manifest, index=False)
    metadata['source'] = ['atlas'] * 20 + ['external_study'] * 20
    metadata['condition'] = ['atlas_reference'] * 20 + ['healthy'] * 10 + ['disease'] * 10
    metadata['donor_id'] = ['unknown'] * 20 + ['WT1'] * 5 + ['WT2'] * 5 + ['KO1'] * 5 + ['KO2'] * 5
    metadata['qc_total_counts'] = [np.nan] * 20 + [1000.] * 20
    metadata['qc_mito_fraction'] = [np.nan] * 20 + [.02] * 20
    metadata_path = source / 'evaluation_metadata.csv.gz'
    metadata.to_csv(metadata_path)
    config = HoldoutConfig(manifest=str(manifest), output=str(output), hv_genes=32, num_macrogenes=4,
                           hidden_dim=9, model_dim=5, batch_size=512, pretrain_epochs=1, metric_epochs=1,
                           device='cpu', neighbor_backend='jax', trials=('baseline', 'infonce'),
                           query_block=64, reference_block=256)
    os.environ.update(config.execution_env())
    os.environ.update(DISEASE_METADATA=str(metadata_path), DISEASE_INPUT_BUNDLE='',
                      REPORT_NOTEBOOK='notebooks/jax/human_monkey_mouse_frog_fish_rdeb_skin_holdout.ipynb')
    prepare(config)
    subprocess.run([sys.executable, ROOT / 'scripts/record_saturn_inputs.py', '--manifest', manifest,
                    '--split', output / 'split/split.csv', '--output', output / 'input_record.json'], check=True)
    spec = importlib.util.spec_from_file_location('disease_smoke_runner', ROOT / 'scripts/run_cell_holdout_jax.py')
    sys.path.insert(0, str(ROOT / 'scripts'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    code = module.run(config, start=datetime.now(timezone.utc).isoformat())
    if code:
        raise SystemExit(code)
    assert not (output / 'run/mmd').exists() and not (output / 'run/ot').exists()
    assert json.loads((output / 'execution_result.json').read_text())['execution_success']
    print('PASS: five-species CPU training, baseline + InfoNCE, frozen inference, disease views and executed notebook')


if __name__ == '__main__':
    main()

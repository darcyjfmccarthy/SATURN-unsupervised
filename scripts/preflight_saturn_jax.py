#!/usr/bin/env python3
"""Inspect pilot inputs and host array sizes without loading expression matrices."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jax_saturn.data.manifest import load_manifest


def inspect_manifest(manifest, *, base_dir=None, hv_genes=2000, embed_dim=256,
                     num_macrogenes=200, expected_species=None):
    import h5py

    if min(hv_genes, embed_dim, num_macrogenes) < 1:
        raise ValueError("Array dimensions must be positive")
    rows = load_manifest(manifest, base_dir=base_dir)
    issues, entries = [], []
    if expected_species is not None and len(rows) != expected_species:
        issues.append(f"Expected {expected_species} species, found {len(rows)}")
    for row in rows:
        entry = {"species": row.species, "atlas": str(row.path),
                 "embedding": str(row.embedding_path), "cells": None, "genes": None}
        for kind, path in (("atlas", row.path), ("embedding", row.embedding_path)):
            entry[kind + "_bytes"] = path.stat().st_size if path.is_file() else None
            if not path.is_file():
                issues.append(f"{row.species}: missing {kind}: {path}")
        if row.path.is_file():
            try:
                with h5py.File(row.path, "r") as handle:
                    matrix = handle["X"]
                    shape = matrix.shape if isinstance(matrix, h5py.Dataset) else matrix.attrs["shape"]
                    if len(shape) != 2 or any(int(size) < 1 for size in shape):
                        raise ValueError("X must have a nonempty two-dimensional shape")
                    entry["cells"], entry["genes"] = map(int, shape)
            except (OSError, KeyError, TypeError, ValueError) as error:
                issues.append(f"{row.species}: cannot inspect atlas: {error}")
        entries.append(entry)
    known = [entry for entry in entries if entry["cells"] is not None]
    cells = sum(entry["cells"] for entry in known)
    return {
        "schema_version": 1, "manifest": str(Path(manifest).resolve()),
        "ready_for_input_loading": not issues,
        "scope": "Path availability and atlas shapes only; embedding contents, gene overlap, labels and training compatibility are not validated.",
        "species_count": len(entries), "inspected_atlas_count": len(known),
        "total_cells_in_inspected_atlases": cells,
        "dimensions": {"hv_genes": hv_genes, "embed_dim": embed_dim,
                       "num_macrogenes": num_macrogenes},
        "host_array_bytes": {
            "dense_hv_expression_float32": sum(entry["cells"] * min(entry["genes"], hv_genes) * 4 for entry in known),
            "one_embedding_bank_float32": cells * embed_dim * 4,
            "one_macrogene_export_float32": cells * num_macrogenes * 4,
        },
        "memory_scope": "Array sizes for inspected atlases, assuming all requested HV genes survive filtering. Excludes raw matrices, copies, graphs, parameters, optimizer, gradients, activations and compilation; not a device-fit estimate.",
        "issues": issues, "species": entries,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/datatable.csv")
    parser.add_argument("--base-dir", type=Path)
    parser.add_argument("--expected-species", type=int)
    parser.add_argument("--hv-genes", type=int, default=2000)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--num-macrogenes", type=int, default=200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.expected_species is not None and args.expected_species < 1:
        parser.error("--expected-species must be positive")
    try:
        report = inspect_manifest(args.manifest, base_dir=args.base_dir,
                                  hv_genes=args.hv_genes, embed_dim=args.embed_dim,
                                  num_macrogenes=args.num_macrogenes,
                                  expected_species=args.expected_species)
    except ValueError as error:
        parser.error(str(error))
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized, end="")
    return 0 if report["ready_for_input_loading"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

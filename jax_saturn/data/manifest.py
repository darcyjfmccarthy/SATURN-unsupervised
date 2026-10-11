"""Normalize the two repository manifest formats into canonical species order."""

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManifestRow:
    species: str
    path: Path
    embedding_path: Path
    in_label_col: str
    source_row_index: int


def load_manifest(path, *, base_dir=None, check_paths=False):
    """Read HMM or datatable CSV; source indices are zero-based data row indices.

    Relative data paths use the repository root by default, matching the existing
    training scripts. External manifests can supply an explicit base_dir.
    """
    root = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parents[2]
    root = root.resolve()
    rows = []
    seen = set()
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if len(set(columns)) != len(columns):
            raise ValueError("Manifest contains duplicate columns")
        if "species" not in columns:
            raise ValueError("Manifest requires species")
        for aliases in (("path", "atlas"), ("embedding_path", "gene_embeddings")):
            if not set(aliases).intersection(columns):
                raise ValueError(f"Manifest requires {' or '.join(aliases)}")
        for index, row in enumerate(reader):
            if None in row:
                raise ValueError(f"Manifest row {index} has extra fields")

            def value(*names, default=None):
                values = [(row.get(name) or "").strip() for name in names]
                present = [item for item in values if item]
                if len(set(present)) > 1:
                    raise ValueError(f"Manifest row {index}: conflicting aliases {names}")
                if present:
                    return present[0]
                if default is not None:
                    return default
                raise ValueError(f"Manifest row {index}: missing {names[0]}")

            species = value("species")
            if species in seen:
                raise ValueError(f"Duplicate manifest species: {species}")
            seen.add(species)

            def resolve(raw):
                target = Path(raw).expanduser()
                target = (target if target.is_absolute() else root / target).resolve()
                if check_paths and not target.is_file():
                    raise FileNotFoundError(target)
                return target

            rows.append(ManifestRow(
                species=species,
                path=resolve(value("path", "atlas")),
                embedding_path=resolve(value("embedding_path", "gene_embeddings")),
                in_label_col=value("in_label_col", default="cellType"),
                source_row_index=index,
            ))
    if not rows:
        raise ValueError("Manifest has no data rows")
    return sorted(rows, key=lambda row: row.species)

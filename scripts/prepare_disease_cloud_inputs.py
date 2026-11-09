#!/usr/bin/env python3
"""Copy audited disease preparation into a portable data/ cloud input bundle."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from disease_pilot.cloud import prepare_bundle, verify_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, default=ROOT / 'out/disease_pilot/rdeb_skin/prep_seed0')
    parser.add_argument('--destination', type=Path, default=ROOT / 'data/disease_pilot/rdeb_skin/cell_holdout_seed0')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/manifests/hmmff_rdeb_skin_cell_holdout.csv')
    parser.add_argument('--verify-bundle', type=Path)
    args = parser.parse_args()
    record = verify_bundle(args.verify_bundle) if args.verify_bundle else prepare_bundle(args.prepared, args.destination, args.manifest)
    print(f'Verified portable disease bundle: {len(record["files"])} files; counts: {record["counts"]}')


if __name__ == '__main__':
    main()

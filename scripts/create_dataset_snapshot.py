"""Create or verify the read-only frozen dataset snapshot."""

from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.provenance import create_dataset_snapshot, verify_dataset_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--manifest", default="artifacts/manifests/dataset_manifest.csv")
    parser.add_argument("--snapshot", default="artifacts/manifests/dataset_snapshot.json")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    if args.verify:
        payload = verify_dataset_snapshot(
            args.data_root, args.manifest, args.snapshot, project_root=PROJECT_ROOT,
        )
        print("Dataset snapshot verified without modification.")
    else:
        payload = create_dataset_snapshot(
            args.data_root, args.manifest, args.snapshot, project_root=PROJECT_ROOT,
        )
        print("Dataset snapshot created. Dataset files were read only.")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


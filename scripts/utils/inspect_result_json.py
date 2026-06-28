#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Inspect one result.json file.")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    with args.path.open() as f:
        data = json.load(f)

    keys = [
        "dataset", "protocol", "subset", "imputation_label", "test_person", "val_person",
        "epochs", "batch_size", "patience", "elapsed_seconds", "finished_at",
    ]
    for key in keys:
        print(f"{key}: {data.get(key)}")
    print("metrics:")
    print(json.dumps(data.get("metrics", {}), indent=2))


if __name__ == "__main__":
    main()

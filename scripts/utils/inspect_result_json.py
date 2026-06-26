#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from pprint import pprint


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspeciona um result.json de um fold.")
    parser.add_argument("json_path", type=Path)
    args = parser.parse_args()

    with args.json_path.open() as f:
        data = json.load(f)

    keys = ["status", "dataset", "protocol", "subset", "imputation_label", "test_person", "val_person", "elapsed_seconds", "finished_at"]
    print("Resumo:")
    for key in keys:
        print(f"{key}: {data.get(key)}")

    print("\nmetrics:")
    pprint(data.get("metrics", {}))

    print("\ntrainer_result keys:")
    trainer = data.get("trainer_result", {}) or {}
    print(sorted(trainer.keys()))


if __name__ == "__main__":
    main()

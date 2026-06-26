#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from collections import Counter


def main() -> None:
    parser = argparse.ArgumentParser(description="Conta result.json por subset/imputação.")
    parser.add_argument("results_dir", type=Path, help="Ex: experiments/ksl_nested_lopo_resume_grid")
    args = parser.parse_args()

    files = sorted((args.results_dir / "runs").glob("*/*/*/result.json"))
    print(f"Total result.json: {len(files)}")

    counter = Counter()
    for path in files:
        rel = path.relative_to(args.results_dir / "runs")
        parts = rel.parts
        if len(parts) >= 3:
            counter[(parts[0], parts[1])] += 1

    for (subset, imputation), count in sorted(counter.items()):
        print(f"{subset:10s} {imputation:20s} {count:6d}")


if __name__ == "__main__":
    main()

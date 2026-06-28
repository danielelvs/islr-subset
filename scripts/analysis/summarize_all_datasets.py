#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Combine summary_outer CSVs from multiple datasets.")
    parser.add_argument("--reports-root", type=Path, default=Path("reports"))
    parser.add_argument("--output", type=Path, default=Path("reports/all_datasets_summary_outer.csv"))
    args = parser.parse_args()

    files = sorted(args.reports_root.glob("*/**/*_summary_outer.csv"))
    if not files:
        raise FileNotFoundError(f"Nenhum *_summary_outer.csv encontrado em {args.reports_root}")

    frames = []
    for path in files:
        df = pd.read_csv(path)
        df["source_file"] = str(path)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Gerado: {args.output}")
    print(out[[c for c in ["dataset", "subset", "imputation_label", "f1_mean", "f1_std"] if c in out.columns]].to_string(index=False))


if __name__ == "__main__":
    main()

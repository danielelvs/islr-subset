#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULTS = {
    "ksl": Path("reports/ksl/ksl_nested_lopo_summary_outer.csv"),
    "include50": Path("reports/include50/include50_nested_lopo_summary_outer.csv"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Combina os summaries finais dos datasets em uma tabela única.")
    parser.add_argument("--output", type=Path, default=Path("reports/summary_all_datasets.csv"))
    args = parser.parse_args()

    frames = []
    for dataset, path in DEFAULTS.items():
        if not path.exists():
            print(f"Aviso: não encontrei {path}; pulando {dataset}.")
            continue
        df = pd.read_csv(path)
        if "dataset" not in df.columns:
            df.insert(0, "dataset", dataset)
        frames.append(df)

    if not frames:
        raise FileNotFoundError("Nenhum summary encontrado em reports/ksl ou reports/include50.")

    out = pd.concat(frames, ignore_index=True, sort=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Gerado: {args.output}")

    cols = [c for c in ["dataset", "subset", "imputation_label", "accuracy_mean", "f1_mean", "f1_std", "f1_count"] if c in out.columns]
    print(out[cols].to_string(index=False))


if __name__ == "__main__":
    main()

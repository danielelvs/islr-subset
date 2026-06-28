#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

META_ALIASES = {
    "person": ["person", "signer", "interpreter", "participant_id"],
    "category": ["category", "sign_id", "label", "class"],
    "video_name": ["video_name", "sequence_id", "path", "file", "filename"],
    "frame": ["frame", "frame_id"],
}


def find_any(columns, names):
    for name in names:
        if name in columns:
            return name
    return None


def main():
    parser = argparse.ArgumentParser(description="Validate an ISLR landmark CSV before batch training.")
    parser.add_argument("--data-csv", required=True, type=Path)
    parser.add_argument("--nrows", type=int, default=5000)
    args = parser.parse_args()

    if not args.data_csv.exists():
        raise FileNotFoundError(args.data_csv)

    df = pd.read_csv(args.data_csv, nrows=args.nrows)
    print(f"Arquivo: {args.data_csv}")
    print(f"Amostra: {len(df):,} linhas | {len(df.columns):,} colunas")

    for canonical, aliases in META_ALIASES.items():
        found = find_any(df.columns, aliases)
        print(f"{canonical:10s}: {'OK -> ' + found if found else 'NÃO ENCONTRADO'}")

    landmark_cols = [c for c in df.columns if c.endswith(("_x", "_y", "_z")) or c.endswith((".x", ".y", ".z"))]
    print(f"landmarks : {len(landmark_cols)} colunas com sufixo de coordenada detectadas")

    if landmark_cols:
        print("Exemplos:", landmark_cols[:12])


if __name__ == "__main__":
    main()

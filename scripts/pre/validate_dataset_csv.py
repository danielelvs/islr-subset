#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

DEFAULT_METADATA_COLUMNS = [
    "category", "video_name", "frame", "person", "missing_hand", "missing_face", "sign", "sign_id", "interpreter", "signer",
]
REQUIRED_ANY_PERSON = ["person", "signer", "interpreter", "participant_id"]
REQUIRED_ANY_VIDEO = ["video_name", "category", "path", "sequence_id"]
REQUIRED_ANY_SIGN = ["sign", "sign_id", "label", "class", "gloss"]


def axis_columns(columns: list[str]) -> list[str]:
    return [c for c in columns if re.search(r"[_.:][xyz]$", c)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Valida rapidamente um CSV de landmarks para os batches.")
    parser.add_argument("csv", type=Path, help="CSV de landmarks, ex: data/interim/ksl/ksl_mediapipe.csv")
    parser.add_argument("--nrows", type=int, default=5000, help="Linhas a ler para inspeção rápida. Use 0 para ler tudo.")
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(args.csv)

    df = pd.read_csv(args.csv, nrows=None if args.nrows == 0 else args.nrows)
    cols = list(df.columns)
    axes = axis_columns(cols)

    print(f"Arquivo: {args.csv}")
    print(f"Linhas lidas: {len(df):,}")
    print(f"Colunas: {len(cols):,}")
    print(f"Colunas de coordenadas detectadas: {len(axes):,}")

    for label, options in [("pessoa", REQUIRED_ANY_PERSON), ("vídeo", REQUIRED_ANY_VIDEO), ("classe/sinal", REQUIRED_ANY_SIGN)]:
        found = [c for c in options if c in cols]
        print(f"Campo de {label}: {'OK ' + str(found) if found else 'NÃO ENCONTRADO'}")

    metadata_found = [c for c in DEFAULT_METADATA_COLUMNS if c in cols]
    print(f"Metadados conhecidos encontrados ({len(metadata_found)}): {metadata_found}")

    for c in ["person", "signer", "interpreter", "video_name", "category", "sign", "sign_id"]:
        if c in df.columns:
            print(f"unique({c}) = {df[c].nunique(dropna=True)}")

    nan_rate = df[axes].isna().mean().mean() if axes else float("nan")
    print(f"Taxa média de NaN nas coordenadas lidas: {nan_rate:.4f}")

    if not axes:
        raise SystemExit("ERRO: não encontrei colunas terminando em _x/_y/_z, .x/.y/.z ou :x/:y/:z.")

    print("Validação básica concluída.")


if __name__ == "__main__":
    main()

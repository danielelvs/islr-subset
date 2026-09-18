#!/usr/bin/env python3
"""Stream one dataset out of a published combined ``frames.csv`` bundle."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import pandas as pd

DATASETS = ("include50", "ksl", "minds", "ufop")
REQUIRED_COLUMNS = {
    "dataset", "class_id", "sample_id", "sequence_id", "signer_id",
    "frame_id", "split",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract one dataset from a combined publication bundle."
    )
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunksize", type=int, default=10_000)
    args = parser.parse_args()

    frames_path = args.bundle_dir / "frames.csv"
    if not frames_path.exists():
        raise FileNotFoundError(frames_path)
    if args.output.exists():
        raise FileExistsError(
            f"Output already exists: {args.output}. Choose another path."
        )
    if args.chunksize < 1:
        parser.error("--chunksize must be at least 1")

    header = pd.read_csv(frames_path, nrows=0).columns
    missing = REQUIRED_COLUMNS - set(header)
    if missing:
        raise ValueError(f"frames.csv is missing canonical columns: {sorted(missing)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    total = 0
    try:
        with tempfile.NamedTemporaryFile(
            dir=args.output.parent,
            prefix=args.output.name + ".",
            suffix=".partial",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)

        first = True
        for chunk in pd.read_csv(
            frames_path,
            chunksize=args.chunksize,
            dtype={"dataset": "string"},
            keep_default_na=True,
        ):
            selected = chunk[chunk["dataset"].eq(args.dataset)]
            if selected.empty:
                continue
            selected.to_csv(
                temporary_path,
                mode="w" if first else "a",
                header=first,
                index=False,
            )
            first = False
            total += len(selected)

        if first:
            raise ValueError(f"No rows found for dataset={args.dataset!r}")
        os.replace(temporary_path, args.output)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    print(f"Saved {total:,} frames for {args.dataset} to {args.output}")


if __name__ == "__main__":
    main()

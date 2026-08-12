#!/usr/bin/env python3
"""Extract frame-level landmarks from a raw video dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from datasets.base_dataset import BaseVideoDataset
from extraction.base_extractor import BaseExtractor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract landmarks from a supported video dataset."
    )
    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        choices=["minds", "ufop", "vlibrasil"],
    )
    parser.add_argument(
        "-e",
        "--extractor",
        default="mediapipe",
        choices=["mediapipe", "openpose"],
    )
    parser.add_argument("-i", "--input-dir", default="data/raw")
    parser.add_argument("-o", "--output-dir", default="data/interim")
    parser.add_argument("-c", "--chunk-size", type=int, default=10_000)
    args = parser.parse_args()

    dataset = BaseVideoDataset.create(args.dataset, args.input_dir)
    extractor = BaseExtractor.create(args.extractor)
    processor = dataset.get_processor(extractor)
    videos = dataset.prepare_data()
    if not videos:
        raise RuntimeError(f"No videos were found for dataset '{args.dataset}'.")

    output_directory = Path(args.output_dir) / args.dataset
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{args.dataset}_{args.extractor}.csv"
    if output_path.exists():
        raise FileExistsError(
            f"{output_path} already exists. Remove it before running extraction again."
        )

    processor.process_all(videos, str(output_path), chunk_size=args.chunk_size)
    print(f"\nExtraction completed: {output_path}")


if __name__ == "__main__":
    main()

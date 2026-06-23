#!/usr/bin/env python
"""
Step 1 — Extract landmarks from raw videos.

Usage:
    python scripts/01_extract_landmarks.py -d minds -e mediapipe
    python scripts/01_extract_landmarks.py -d ufop  -e mediapipe -i data/raw -o data/interim
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datasets.base_dataset import BaseVideoDataset
from extraction.base_extractor import BaseExtractor


def main():
    parser = argparse.ArgumentParser(description="Landmark extraction from video datasets")
    parser.add_argument("-d", "--dataset", required=True, choices=["minds", "ufop", "vlibras"], help="Dataset name")
    parser.add_argument("-e", "--extractor", default="mediapipe", choices=["mediapipe", "openpose"], help="Extractor backend")
    parser.add_argument("-i", "--input_dir", default="data/raw", help="Raw dataset root")
    parser.add_argument("-o", "--output_dir", default="data/interim", help="Output directory for CSVs")
    parser.add_argument("-c", "--chunk_size", type=int, default=10_000, help="Rows per CSV chunk")
    args = parser.parse_args()

    dataset   = BaseVideoDataset.create(args.dataset, args.input_dir)
    extractor = BaseExtractor.create(args.extractor)
    processor = dataset.get_processor(extractor)
    videos    = dataset.prepare_data()

    os.makedirs(os.path.join(args.output_dir, args.dataset), exist_ok=True)
    out = os.path.join(args.output_dir, args.dataset, f"{args.dataset}_{args.extractor}.csv")

    if os.path.exists(out):
        raise FileExistsError(f"{out} already exists. Remove it before re-running.")

    processor.process_all(videos, out, chunk_size=args.chunk_size)
    print(f"\nExtraction complete → {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Measure landmark extraction latency for MediaPipe or OpenPose."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from datasets.base_dataset import BaseVideoDataset
from extraction.base_extractor import BaseExtractor


def frame_count(video_path: str) -> int:
    capture = cv2.VideoCapture(video_path)
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return count


def representative_sample(videos: list[tuple], sample_size: int) -> list[tuple]:
    """Select exactly ``sample_size`` videos across the duration distribution."""

    if sample_size < 1:
        raise ValueError("sample_size must be at least 1")
    ordered = sorted(videos, key=lambda video: frame_count(video[0]))
    if len(ordered) <= sample_size:
        return ordered
    indices = np.linspace(0, len(ordered) - 1, sample_size, dtype=int)
    return [ordered[index] for index in indices]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark landmark extraction on a representative video sample."
    )
    parser.add_argument("-d", "--dataset", required=True, choices=["minds", "ufop"])
    parser.add_argument(
        "-e",
        "--extractor",
        default="mediapipe",
        choices=["mediapipe", "openpose"],
    )
    parser.add_argument("-i", "--input-dir", default="data/raw")
    parser.add_argument("-n", "--sample-size", type=int, default=5)
    parser.add_argument("-o", "--output-dir", default="reports/tables")
    args = parser.parse_args()

    dataset = BaseVideoDataset.create(args.dataset, args.input_dir)
    extractor = BaseExtractor.create(args.extractor)
    videos = dataset.prepare_data()
    sample = representative_sample(videos, args.sample_size)
    if not sample:
        raise RuntimeError("No videos were available for the benchmark.")

    video_durations: list[float] = []
    total_frames = 0
    for video_path, *_ in tqdm(sample, desc="Benchmarking"):
        start = time.perf_counter()
        processed_frames = sum(
            1 for _ in extractor.get_video_landmarks(video_path)
        )
        elapsed = time.perf_counter() - start
        video_durations.append(elapsed)
        total_frames += processed_frames

    total_seconds = sum(video_durations)
    average_seconds = total_seconds / len(sample)
    frames_per_second = total_frames / total_seconds if total_seconds else 0.0

    print(f"\nExtractor:          {args.extractor}")
    print(f"Sampled videos:     {len(sample)}")
    print(f"Processed frames:   {total_frames}")
    print(f"Average time/video: {average_seconds:.4f} s")
    print(f"Overall FPS:        {frames_per_second:.2f}")

    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"latency_{args.dataset}_{args.extractor}.json"
    output = {
        "dataset": args.dataset,
        "extractor": args.extractor,
        "sampled_videos": len(sample),
        "processed_frames": total_frames,
        "total_seconds": total_seconds,
        "average_time_per_video_seconds": average_seconds,
        "frames_per_second": frames_per_second,
        "video_durations_seconds": video_durations,
    }
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

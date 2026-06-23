#!/usr/bin/env python
"""
Step 5 — Measure extraction latency (FPS) for MediaPipe vs OpenPose.

Usage:
    python scripts/05_latency_benchmark.py -d minds -e mediapipe -i data/raw
    python scripts/05_latency_benchmark.py -d ufop  -e openpose  -i data/raw
"""

import argparse
import json
import os
import sys
from time import time

import cv2
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datasets.base_dataset import BaseVideoDataset
from extraction.base_extractor import BaseExtractor


def video_info(path: str):
    cap = cv2.VideoCapture(path)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()
    return frames, fps


def representative_sample(videos, n: int = 5):
    sorted_vids = sorted(videos, key=lambda v: video_info(v[0])[0])
    if len(sorted_vids) <= n:
        return sorted_vids
    mid = len(sorted_vids) // 2
    return sorted_vids[:n] + sorted_vids[mid - n//2: mid + n//2 + n % 2] + sorted_vids[-n:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True, choices=["minds", "ufop"])
    parser.add_argument("-e", "--extractor", default="mediapipe", choices=["mediapipe", "openpose"])
    parser.add_argument("-i", "--input_dir", default="data/raw")
    parser.add_argument("-n", "--n_videos", type=int, default=5)
    parser.add_argument("-o", "--output_dir", default="reports/tables")
    args = parser.parse_args()

    dataset   = BaseVideoDataset.create(args.dataset, args.input_dir)
    extractor = BaseExtractor.create(args.extractor)
    videos    = dataset.prepare_data()
    sample    = representative_sample(videos, args.n_videos)

    total_time, total_frames = 0.0, 0
    for video_path, *_ in tqdm(sample, desc="Benchmarking"):
        frames, _ = video_info(video_path)
        t0 = time()
        for _ in extractor.get_video_landmarks(video_path):
            pass
        total_time   += time() - t0
        total_frames += frames

    avg_time = total_time / len(sample)
    fps      = total_frames / total_time

    print(f"\nExtractor:      {args.extractor}")
    print(f"Videos sampled:   {len(sample)}")
    print(f"Avg time/video:   {avg_time:.4f}s")
    print(f"Overall FPS:      {fps:.2f}")

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, f"latency_{args.dataset}_{args.extractor}.json")
    with open(out, "w") as f:
        json.dump({"extractor": args.extractor, "fps": fps, "avg_time_s": avg_time}, f)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()

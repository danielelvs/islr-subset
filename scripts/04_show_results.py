#!/usr/bin/env python
"""
Step 4 — Aggregate and display results from a training run.

Usage:
    python scripts/04_show_results.py -r 63 -d minds
"""

import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--ref", required=True)
    parser.add_argument("-d", "--dataset", required=True)
    parser.add_argument("--output_dir", default="experiments")
    args = parser.parse_args()

    results_path = os.path.join(args.output_dir, args.ref, args.dataset)
    if not os.path.isdir(results_path):
        print(f"No results found at {results_path}")
        sys.exit(1)

    results = []
    for fname in os.listdir(results_path):
        if fname.endswith(".json"):
            with open(os.path.join(results_path, fname)) as f:
                results.append(json.load(f))

    if not results:
        print("No JSON result files found.")
        sys.exit(1)

    first = results[0]
    print(f"\n{args.dataset} #{args.ref}")
    print(f"  model:         {first.get('model', 'resnet18')}")
    print(f"  image_method:  {first.get('image_method', 'Skeleton-DML')}")
    print(f"  optimizer:     {first.get('optimizer')}")
    print(f"  epochs:        {first.get('epochs')}")
    print(f"  folds:         {len(results)}")
    print()

    def metric_stats(fn, **kw):
        vals = [fn(r["true_labels"], r["predicted_labels"], **kw) for r in results]
        return np.mean(vals), np.std(vals)

    for name, fn, kw in [
        ("Accuracy",  accuracy_score,  {}),
        ("Precision", precision_score, {"average": "macro", "zero_division": 0}),
        ("Recall",    recall_score,    {"average": "macro", "zero_division": 0}),
        ("F1",        f1_score,        {"average": "macro", "zero_division": 0}),
    ]:
        m, s = metric_stats(fn, **kw)
        print(f"  {name:<10} {m:.4f} ± {s:.4f}")


if __name__ == "__main__":
    main()

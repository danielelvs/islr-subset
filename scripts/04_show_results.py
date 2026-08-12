#!/usr/bin/env python3
"""Compatibility entry point for the maintained result summarizer.

This numbered script keeps the original step-based workflow while delegating
all aggregation logic to ``scripts/batch/summarize_dataset_results.py``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARIZER = PROJECT_ROOT / "scripts" / "batch" / "summarize_dataset_results.py"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize experiment results.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--reports-dir", required=True, type=Path)
    parser.add_argument("--expected-runs-per-condition", type=int, default=None)
    args = parser.parse_args()

    command = [
        sys.executable,
        str(SUMMARIZER),
        "--dataset",
        args.dataset,
        "--results-dir",
        str(args.results_dir),
        "--reports-dir",
        str(args.reports_dir),
    ]
    if args.expected_runs_per_condition is not None:
        command.extend(
            [
                "--expected-runs-per-condition",
                str(args.expected_runs_per_condition),
            ]
        )

    print("Running:", " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()

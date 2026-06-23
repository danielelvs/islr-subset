#!/usr/bin/env python
"""
Step 4 — Agrega resultados de um experimento e salva tabela CSV.

Uso:
    python scripts/04_show_results.py -r 63 -d minds
    python scripts/04_show_results.py -r 63 -d minds --save-table
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


METRICS = [
    ("Accuracy",  accuracy_score,  {}),
    ("Precision", precision_score, {"average": "macro", "zero_division": 0}),
    ("Recall",    recall_score,    {"average": "macro", "zero_division": 0}),
    ("F1",        f1_score,        {"average": "macro", "zero_division": 0}),
]


def compute(fn, y_true, y_pred, **kw):
    return fn(y_true, y_pred, **kw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--ref",        required=True)
    parser.add_argument("-d", "--dataset",    required=True)
    parser.add_argument("--output_dir",       default="experiments")
    parser.add_argument("--table_dir",        default="reports/tables")
    parser.add_argument("--save-table",       action="store_true",
                        help="Salva CSV em reports/tables/")
    args = parser.parse_args()

    results_path = os.path.join(args.output_dir, args.ref, args.dataset)
    if not os.path.isdir(results_path):
        print(f"Nenhum resultado em {results_path}")
        sys.exit(1)

    results = []
    for fname in sorted(os.listdir(results_path)):
        if fname.endswith(".json"):
            with open(os.path.join(results_path, fname)) as f:
                results.append(json.load(f))

    if not results:
        print("Nenhum arquivo JSON encontrado.")
        sys.exit(1)

    first = results[0]

    # ── Terminal ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*55}")
    print(f"  {args.dataset} #{args.ref}  ({len(results)} folds)")
    print(f"  model:        {first.get('model', 'resnet18')}")
    print(f"  image_method: {first.get('image_method', 'Skeleton-DML')}")
    print(f"  optimizer:    {first.get('optimizer')}")
    print(f"  epochs:       {first.get('epochs')}")
    print(f"{'═'*55}")

    rows = []
    for name, fn, kw in METRICS:
        vals = [compute(fn, r["true_labels"], r["predicted_labels"], **kw) for r in results]
        mean, std = np.mean(vals), np.std(vals)
        print(f"  {name:<12} {mean:.4f} ± {std:.4f}   (per fold: {[round(v,4) for v in vals]})")
        rows.append({
            "metric": name,
            "mean":   round(mean, 4),
            "std":    round(std, 4),
            "folds":  [round(v, 4) for v in vals],
        })

    print(f"{'═'*55}\n")

    # ── Salva tabela ──────────────────────────────────────────────────────────
    if args.save_table:
        os.makedirs(args.table_dir, exist_ok=True)
        out_path = os.path.join(
            args.table_dir,
            f"results_{args.dataset}_ref{args.ref}.csv"
        )
        df = pd.DataFrame([
            {
                "dataset":      args.dataset,
                "ref":          args.ref,
                "model":        first.get("model", "resnet18"),
                "image_method": first.get("image_method", "Skeleton-DML"),
                "folds":        len(results),
                **{r["metric"]: f"{r['mean']:.4f}" for r in rows},
                **{f"{r['metric']}_std": f"{r['std']:.4f}" for r in rows},
            }
        ])
        df.to_csv(out_path, index=False)
        print(f"Tabela salva → {out_path}")


if __name__ == "__main__":
    main()

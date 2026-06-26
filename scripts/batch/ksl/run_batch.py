#!/usr/bin/env python3
"""
Batch runner for KSL nested LOPO experiments.

Runs one subset + one imputation setting at a time, with resume by fold.
Expected project layout:

.
├── data/interim/ksl/ksl_mediapipe.csv
├── experiments/
├── scripts/batch/ksl/run_batch.py
└── src/

Example:
python scripts/batch/ksl/run_batch.py \
  --dataset ksl \
  --data-csv data/interim/ksl/ksl_mediapipe.csv \
  --results-dir experiments/ksl_nested_lopo_resume_grid \
  --subset 2nd \
  --imputation true \
  --max-runs 10 \
  --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

# Ensure local src is importable even after moving this file under scripts/batch/<dataset>/.
def find_project_root() -> Path:
    candidates = [Path.cwd(), *Path(__file__).resolve().parents]
    for candidate in candidates:
        if (candidate / "src").exists():
            return candidate
    raise FileNotFoundError(
        "Não encontrei a raiz do projeto. Rode este script a partir da pasta do projeto "
        "ou garanta que exista uma pasta src/."
    )


PROJECT_ROOT = find_project_root()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Imports from project. These require the local src fixes already discussed:
# - preprocessing/landmark_subsets.py without MediaPipe import
# - subset_laines() with 68 landmarks
from preprocessing.landmark_subsets import SUBSETS as LANDMARK_SUBSETS, indices_to_columns
from preprocessing.imputation import impute_by_video
from training.trainer import Trainer

# Register model and representation subclasses used by Trainer/BaseModel/BaseRepresentation.
import models.resnet18  # noqa: F401
import representations.skeleton_dml  # noqa: F401


DEFAULT_METADATA_COLUMNS = [
    "category",
    "video_name",
    "frame",
    "person",
    "missing_hand",
    "missing_face",
    "sign",
    "sign_id",
    "interpreter",
    "signer",
]


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "sim", "s", "with", "com"}:
        return True
    if value in {"false", "0", "no", "n", "nao", "não", "without", "sem"}:
        return False
    raise argparse.ArgumentTypeError(f"Valor booleano inválido: {value}")


def parse_max_runs(value: str | None) -> int | None:
    if value is None:
        return None
    value = str(value).strip().lower()
    if value in {"none", "null", "all", "tudo", ""}:
        return None
    return int(value)


def sanitize_id(value: Any) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text)
    return text.strip("-") or "unknown"


def impute_label(impute: bool) -> str:
    return "with_imputation" if impute else "without_imputation"


def strip_axis(column_name: str) -> str:
    return re.sub(r"_[xyz]$", "", column_name)


def src_base_name_for_index(idx: int) -> str:
    cols = indices_to_columns([idx])
    if not cols:
        raise ValueError(f"Índice de landmark inválido ou sem colunas: {idx}")
    return strip_axis(cols[0])


def numeric_aliases_for_index(idx: int) -> list[str]:
    if idx < 468:
        return [f"face_{idx}"]
    if 468 <= idx < 489:
        h = idx - 468
        return [f"hand_0_{h}", f"hand_0_{h:02d}", f"left_hand_{h}", f"left_hand_{h:02d}"]
    if 489 <= idx < 522:
        p = idx - 489
        return [f"pose_{p}", f"pose_{p:02d}", f"body_{p}", f"body_{p:02d}"]
    if 522 <= idx < 543:
        h = idx - 522
        return [f"hand_1_{h}", f"hand_1_{h:02d}", f"right_hand_{h}", f"right_hand_{h:02d}"]
    raise ValueError(f"Índice fora do intervalo MediaPipe Holistic: {idx}")


def expected_base_names_for_index(idx: int) -> list[str]:
    base_names: list[str] = []
    src_base = src_base_name_for_index(idx)
    base_names.append(src_base)

    if src_base.startswith("hand_0_"):
        base_names.append(src_base.replace("hand_0_", "left_hand_", 1))
    if src_base.startswith("hand_1_"):
        base_names.append(src_base.replace("hand_1_", "right_hand_", 1))
    if src_base.startswith("pose_"):
        base_names.append(src_base.replace("pose_", "body_", 1))

    base_names.extend(numeric_aliases_for_index(idx))
    return list(dict.fromkeys(base_names))


def find_existing_column(df: pd.DataFrame, base_names: list[str], axis: str) -> str | None:
    candidates: list[str] = []
    for base in base_names:
        candidates.extend([f"{base}_{axis}", f"{base}.{axis}", f"{base}:{axis}"])
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def canonical_base_name(idx: int) -> str:
    # Stable numeric output names. LopoDataset will still detect x/y/z columns.
    if idx < 468:
        return f"face_{idx}"
    if 468 <= idx < 489:
        return f"hand_0_{idx - 468:02d}"
    if 489 <= idx < 522:
        return f"pose_{idx - 489:02d}"
    if 522 <= idx < 543:
        return f"hand_1_{idx - 522:02d}"
    raise ValueError(f"Índice fora do intervalo MediaPipe Holistic: {idx}")


def normalize_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_candidates = {
        "interpreter": "person",
        "signer": "person",
        "frame_id": "frame",
        "sign_id": "category",
        "sequence_id": "video_name",
        "path": "video_name",
    }
    for old, new in rename_candidates.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    required = ["person", "category", "video_name"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV sem colunas obrigatórias {missing}. Colunas encontradas: {list(df.columns[:30])}..."
        )

    if "frame" not in df.columns:
        # fallback: preserve input order within each video
        df["frame"] = df.groupby("video_name").cumcount()

    # Keep person as original value when possible, but make category consecutive int.
    categories = sorted(df["category"].dropna().unique())
    df["category"] = df["category"].map({c: i for i, c in enumerate(categories)})
    return df


def subset_indices(subset_name: str) -> list[int]:
    if subset_name not in LANDMARK_SUBSETS:
        raise ValueError(f"Subset desconhecido: {subset_name}. Disponíveis: {list(LANDMARK_SUBSETS)}")
    return LANDMARK_SUBSETS[subset_name]()


def prepare_subset_dataframe(raw_df: pd.DataFrame, subset: str, use_imputation: bool) -> pd.DataFrame:
    df = normalize_metadata(raw_df)
    indices = subset_indices(subset)

    keep_meta = [c for c in DEFAULT_METADATA_COLUMNS if c in df.columns]
    for required in ["category", "video_name", "frame", "person"]:
        if required not in keep_meta:
            keep_meta.append(required)

    result = df[keep_meta].copy()
    missing_columns: list[str] = []

    for idx in indices:
        for axis in ["x", "y", "z"]:
            source_col = find_existing_column(df, expected_base_names_for_index(idx), axis)
            output_col = f"{canonical_base_name(idx)}_{axis}"
            if source_col is None:
                # Keep shape consistent; imputation/fillna will turn this into 0.
                result[output_col] = pd.NA
                missing_columns.append(output_col)
            else:
                result[output_col] = df[source_col]

    landmark_cols = [c for c in result.columns if c.endswith(("_x", "_y", "_z"))]

    if missing_columns:
        print(f"[WARN] {len(missing_columns)} colunas de landmark não encontradas; serão preenchidas com 0 após imputação/fillna.")
        print("       Exemplos:", missing_columns[:10])

    if use_imputation:
        result = impute_by_video(result, landmark_cols)
    else:
        result[landmark_cols] = result[landmark_cols].fillna(0)

    # Sort for deterministic sequence order.
    result = result.sort_values(["video_name", "frame"]).reset_index(drop=True)
    return result


def load_or_create_preprocessed(
    data_csv: Path,
    cache_dir: Path,
    subset: str,
    use_imputation: bool,
    force: bool = False,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / f"ksl_{subset}_{impute_label(use_imputation)}.csv"
    if output_path.exists() and not force:
        print(f"Usando CSV pré-processado em cache: {output_path}")
        return output_path

    print(f"Lendo CSV bruto: {data_csv}")
    raw_df = pd.read_csv(data_csv, na_values=["NaN", "nan", ""])
    print(f"Linhas: {len(raw_df):,} | Colunas: {len(raw_df.columns):,}")

    processed = prepare_subset_dataframe(raw_df, subset, use_imputation)
    processed.to_csv(output_path, index=False)
    print(f"CSV pré-processado salvo: {output_path}")
    print(f"Shape processado: {processed.shape}")
    return output_path


def build_run_plan(df: pd.DataFrame) -> list[tuple[Any, Any]]:
    people = sorted(df["person"].dropna().unique(), key=lambda x: str(x))
    if len(people) < 2:
        raise ValueError(f"Nested LOPO precisa de pelo menos 2 pessoas. Encontradas: {people}")
    return [(test_person, val_person) for test_person in people for val_person in people if val_person != test_person]


def result_path_for(results_dir: Path, subset: str, use_imputation: bool, test_person: Any, val_person: Any) -> Path:
    return (
        results_dir
        / "runs"
        / subset
        / impute_label(use_imputation)
        / f"test={sanitize_id(test_person)}__val={sanitize_id(val_person)}"
        / "result.json"
    )


def running_path_for(result_path: Path) -> Path:
    return result_path.with_name("status_running.json")


def failed_path_for(result_path: Path) -> Path:
    return result_path.with_name("status_failed.json")


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run KSL nested LOPO batch with resume by fold.")
    parser.add_argument("--dataset", default="ksl")
    parser.add_argument("--data-csv", required=True, type=Path)
    parser.add_argument("--results-dir", default=Path("experiments/ksl_nested_lopo_resume_grid"), type=Path)
    parser.add_argument("--subset", required=True, choices=list(LANDMARK_SUBSETS.keys()))
    parser.add_argument("--imputation", required=True, type=str_to_bool)
    parser.add_argument("--max-runs", default="10", help="Number of runs to execute now, or none/all.")
    parser.add_argument("--resume", type=str_to_bool, default=True)
    parser.add_argument("--skip-failed", type=str_to_bool, default=False)
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"], help="Only used for preflight check; Trainer auto-selects device.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--image-method", default="Skeleton-DML")
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.results_dir / "preprocessed"
    trainer_tmp_dir = args.results_dir / "_trainer_tmp_outputs"
    max_runs = parse_max_runs(args.max_runs)

    # Preflight GPU check.
    try:
        import torch
        print("CUDA disponível:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
        elif args.device == "cuda":
            print("[WARN] --device cuda pedido, mas torch.cuda.is_available() = False.")
    except Exception as exc:
        print("[WARN] Não consegui checar CUDA:", repr(exc))

    print("\nConfiguração:")
    print(f"  dataset:      {args.dataset}")
    print(f"  data_csv:     {args.data_csv}")
    print(f"  results_dir:  {args.results_dir}")
    print(f"  subset:       {args.subset} ({len(subset_indices(args.subset))} landmarks)")
    print(f"  imputation:   {impute_label(args.imputation)}")
    print(f"  protocol:     nested LOPO")
    print(f"  epochs:       {args.epochs}")
    print(f"  lr/wd/batch:  {args.lr} / {args.wd} / {args.batch_size}")
    print(f"  patience:     {args.patience}")
    print(f"  max_runs:     {max_runs}")
    print(f"  resume:       {args.resume}\n")

    processed_csv = load_or_create_preprocessed(
        data_csv=args.data_csv,
        cache_dir=cache_dir,
        subset=args.subset,
        use_imputation=args.imputation,
        force=args.force_preprocess,
    )
    df = pd.read_csv(processed_csv)
    run_plan = build_run_plan(df)

    total = len(run_plan)
    completed_before = sum(
        result_path_for(args.results_dir, args.subset, args.imputation, t, v).exists()
        for t, v in run_plan
    )
    failed_before = sum(
        failed_path_for(result_path_for(args.results_dir, args.subset, args.imputation, t, v)).exists()
        for t, v in run_plan
    )

    print(f"Plano: {total} runs para subset={args.subset}, imputation={impute_label(args.imputation)}")
    print(f"Antes: completed={completed_before}, failed={failed_before}, pending={total - completed_before}\n")

    executed_now = 0
    skipped_completed = 0
    skipped_failed = 0
    failed_now = 0

    pbar = tqdm(run_plan, total=total, desc="Nested LOPO")
    for test_person, val_person in pbar:
        desc = f"{args.subset} | {impute_label(args.imputation)} | test={test_person} | val={val_person}"
        pbar.set_description(desc)

        final_result_path = result_path_for(args.results_dir, args.subset, args.imputation, test_person, val_person)
        final_failed_path = failed_path_for(final_result_path)
        final_running_path = running_path_for(final_result_path)

        if args.resume and final_result_path.exists():
            skipped_completed += 1
            continue
        if args.skip_failed and final_failed_path.exists():
            skipped_failed += 1
            continue
        if max_runs is not None and executed_now >= max_runs:
            break

        final_result_path.parent.mkdir(parents=True, exist_ok=True)
        if final_failed_path.exists():
            final_failed_path.unlink()

        run_meta = {
            "status": "running",
            "dataset": args.dataset,
            "protocol": "nested_lopo",
            "subset": args.subset,
            "imputation": args.imputation,
            "test_person": test_person,
            "val_person": val_person,
            "started_at": datetime.now().isoformat(),
        }
        save_json(final_running_path, run_meta)

        ref = (
            f"dataset={args.dataset}__protocol=nested_lopo__subset={args.subset}"
            f"__imputation={impute_label(args.imputation)}__model={args.model}"
            f"__repr={args.image_method.replace('/', '-')}__epochs={args.epochs}"
            f"__lr={args.lr:g}__wd={args.wd:g}__batch={args.batch_size}"
            f"__test={sanitize_id(test_person)}__val={sanitize_id(val_person)}"
        )

        trainer_cfg = {
            "dataset_name": args.dataset,
            "ref": ref,
            "seed": args.seed,
            "validate_people": [val_person],
            "test_people": [test_person],
            "learning_rate": args.lr,
            "weight_decay": args.wd,
            "image_method": args.image_method,
            "model": args.model,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "augment_cfg": {
                "rotation_sigma": 12,
                "zoom_sigma": 0.1,
                "translate_x_sigma": 0.06,
                "translate_y_sigma": 0.0,
                "translate_z_sigma": 0.0,
                "horizontal_flip_prob": 0.5,
            },
            "output_dir": str(trainer_tmp_dir),
            "save_model": False,  # used only if you patched Trainer; harmless otherwise
        }

        start = time.time()
        try:
            result = Trainer(trainer_cfg).run(df)
            elapsed = time.time() - start
            payload = {
                "status": "completed",
                "dataset": args.dataset,
                "protocol": "nested_lopo",
                "subset": args.subset,
                "subset_landmarks": len(subset_indices(args.subset)),
                "imputation": args.imputation,
                "imputation_label": impute_label(args.imputation),
                "test_person": test_person,
                "val_person": val_person,
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "weight_decay": args.wd,
                "batch_size": args.batch_size,
                "patience": args.patience,
                "flip": 0.5,
                "rotation_sigma": 12,
                "zoom_sigma": 0.1,
                "translate_x_sigma": 0.06,
                "model": args.model,
                "image_method": args.image_method,
                "seed": args.seed,
                "elapsed_seconds": elapsed,
                "finished_at": datetime.now().isoformat(),
                "metrics": {
                    "accuracy": result.get("test_accuracy"),
                    "precision": result.get("test_precision"),
                    "recall": result.get("test_recall"),
                    "f1": result.get("test_f1"),
                },
                "trainer_result": result,
            }
            save_json(final_result_path, payload)
            if final_running_path.exists():
                final_running_path.unlink()
            executed_now += 1
            print(f"Resultado salvo em: {final_result_path}")
        except Exception as exc:
            elapsed = time.time() - start
            failed_now += 1
            payload = {
                "status": "failed",
                "dataset": args.dataset,
                "protocol": "nested_lopo",
                "subset": args.subset,
                "imputation": args.imputation,
                "test_person": test_person,
                "val_person": val_person,
                "elapsed_seconds": elapsed,
                "failed_at": datetime.now().isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            save_json(final_failed_path, payload)
            if final_running_path.exists():
                final_running_path.unlink()
            print(f"[ERRO] Run falhou: {desc}")
            print(f"       {type(exc).__name__}: {exc}")
            # Continue with next fold instead of killing the whole batch.
        finally:
            # Remove Trainer's temporary JSON/PTH outputs to avoid filling disk.
            tmp_ref_dir = trainer_tmp_dir / ref
            if tmp_ref_dir.exists():
                shutil.rmtree(tmp_ref_dir, ignore_errors=True)

    completed_after = sum(
        result_path_for(args.results_dir, args.subset, args.imputation, t, v).exists()
        for t, v in run_plan
    )
    failed_after = sum(
        failed_path_for(result_path_for(args.results_dir, args.subset, args.imputation, t, v)).exists()
        for t, v in run_plan
    )

    summary = {
        "executed_now": executed_now,
        "skipped_completed": skipped_completed,
        "skipped_failed": skipped_failed,
        "failed_now": failed_now,
        "max_runs": max_runs,
        "status_before": {
            "total": total,
            "completed": completed_before,
            "failed": failed_before,
            "pending_or_running": total - completed_before,
        },
        "status_after": {
            "total": total,
            "completed": completed_after,
            "failed": failed_after,
            "pending_or_running": total - completed_after,
        },
    }
    print(json.dumps(summary, indent=2, default=str))
    print(f"Progresso deste subset/imputação: {completed_after}/{total} ({completed_after / total * 100:.1f}%)")


if __name__ == "__main__":
    main()

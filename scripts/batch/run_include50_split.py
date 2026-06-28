#!/usr/bin/env python3
"""
Fixed-split runner for INCLUDE-50.

This script reuses the existing Trainer, which expects LOPO-style columns:
  category, video_name, frame, person

For INCLUDE-50, there is no person/signer id. So we map:
  category   <- sign_id
  video_name <- sequence_id
  frame      <- frame_id
  person     <- split  (train / val / test)

Then Trainer is called with:
  validate_people = ["val"]
  test_people     = ["test"]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


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


from preprocessing.landmark_subsets import SUBSETS as LANDMARK_SUBSETS, indices_to_columns
from preprocessing.imputation import impute_by_video
from training.trainer import Trainer

import models.resnet18  # noqa: F401
import representations.skeleton_dml  # noqa: F401


DEFAULT_METADATA_COLUMNS = [
    "category",
    "video_name",
    "frame",
    "person",
    "missing_hand",
    "missing_face",
    "missing_hand_0",
    "missing_hand_1",
    "missing_pose",
    "sign",
    "sign_id",
    "sign_key",
    "split",
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


def impute_label(impute: bool) -> str:
    return "with_imputation" if impute else "without_imputation"


def sanitize_id(value: Any) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text)
    return text.strip("-") or "unknown"


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
    if idx < 468:
        return f"face_{idx}"
    if 468 <= idx < 489:
        return f"hand_0_{idx - 468:02d}"
    if 489 <= idx < 522:
        return f"pose_{idx - 489:02d}"
    if 522 <= idx < 543:
        return f"hand_1_{idx - 522:02d}"
    raise ValueError(f"Índice fora do intervalo MediaPipe Holistic: {idx}")


def normalize_include50_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required = ["sign_id", "sequence_id", "frame_id", "split"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV sem colunas obrigatórias para INCLUDE-50 fixed split: {missing}. "
            f"Colunas encontradas: {list(df.columns[:40])}..."
        )

    # Trainer/LopoDataset expect these names.
    df["category"] = df["sign_id"].astype(int)
    df["video_name"] = df["sequence_id"].astype(str)
    df["frame"] = df["frame_id"].astype(int)
    df["person"] = df["split"].astype(str).str.strip().str.lower()

    valid_splits = {"train", "val", "test"}
    found = set(df["person"].dropna().unique())
    invalid = found - valid_splits
    if invalid:
        raise ValueError(f"Valores inválidos em split/person: {sorted(invalid)}. Use apenas train, val, test.")

    missing_splits = valid_splits - found
    if missing_splits:
        raise ValueError(f"Faltam splits no CSV: {sorted(missing_splits)}. Encontrados: {sorted(found)}")

    return df


def subset_indices(subset_name: str) -> list[int]:
    if subset_name not in LANDMARK_SUBSETS:
        raise ValueError(f"Subset desconhecido: {subset_name}. Disponíveis: {list(LANDMARK_SUBSETS)}")
    return LANDMARK_SUBSETS[subset_name]()


def prepare_subset_dataframe(raw_df: pd.DataFrame, subset: str, use_imputation: bool) -> pd.DataFrame:
    df = normalize_include50_metadata(raw_df)
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
                result[output_col] = pd.NA
                missing_columns.append(output_col)
            else:
                result[output_col] = df[source_col]

    landmark_cols = [c for c in result.columns if c.endswith(("_x", "_y", "_z"))]

    if missing_columns:
        print(f"[WARN] {len(missing_columns)} colunas de landmark não encontradas; serão preenchidas com 0.")
        print("       Exemplos:", missing_columns[:10])

    if use_imputation:
        result = impute_by_video(result, landmark_cols)
    else:
        result[landmark_cols] = result[landmark_cols].fillna(0)

    result = result.sort_values(["video_name", "frame"]).reset_index(drop=True)
    return result


def load_or_create_preprocessed(
    data_csv: Path,
    cache_dir: Path,
    dataset: str,
    subset: str,
    use_imputation: bool,
    force: bool = False,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / f"{dataset}_{subset}_{impute_label(use_imputation)}.csv"
    if output_path.exists() and not force:
        print(f"Usando CSV pré-processado em cache: {output_path}")
        return output_path

    print(f"Lendo CSV bruto: {data_csv}")
    raw_df = pd.read_csv(data_csv, na_values=["NaN", "nan", ""])
    print(f"Linhas: {len(raw_df):,} | Colunas: {len(raw_df.columns):,}")

    processed = prepare_subset_dataframe(raw_df, subset, use_imputation)
    processed.to_csv(output_path, index=False)

    videos = processed[["video_name", "person", "category"]].drop_duplicates()
    print("\nDistribuição por split em vídeos:")
    print(videos["person"].value_counts().to_string())
    print("\nClasses por split:")
    print(videos.groupby("person")["category"].nunique().to_string())

    print(f"\nCSV pré-processado salvo: {output_path}")
    print(f"Shape processado: {processed.shape}")
    return output_path


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run INCLUDE-50 fixed train/val/test split.")
    parser.add_argument("--dataset", default="include50")
    parser.add_argument("--data-csv", default=Path("data/interim/include50/include50_mediapipe_with_split.csv"), type=Path)
    parser.add_argument("--results-dir", default=Path("experiments/include50_fixed_split_grid"), type=Path)
    parser.add_argument("--subset", required=True, choices=list(LANDMARK_SUBSETS.keys()))
    parser.add_argument("--imputation", required=True, type=str_to_bool)
    parser.add_argument("--resume", type=str_to_bool, default=True)
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--force-run", action="store_true")
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
    print(f"  protocol:     fixed split train/val/test")
    print(f"  epochs:       {args.epochs}")
    print(f"  lr/wd/batch:  {args.lr} / {args.wd} / {args.batch_size}")
    print(f"  patience:     {args.patience}")
    print(f"  seed:         {args.seed}\n")

    processed_csv = load_or_create_preprocessed(
        data_csv=args.data_csv,
        cache_dir=cache_dir,
        dataset=args.dataset,
        subset=args.subset,
        use_imputation=args.imputation,
        force=args.force_preprocess,
    )
    df = pd.read_csv(processed_csv)

    final_result_path = (
        args.results_dir
        / "runs"
        / args.subset
        / impute_label(args.imputation)
        / "fixed_split"
        / "result.json"
    )
    final_failed_path = final_result_path.with_name("status_failed.json")
    final_running_path = final_result_path.with_name("status_running.json")

    if args.resume and final_result_path.exists() and not args.force_run:
        print(f"Resultado já existe, pulando: {final_result_path}")
        return

    final_result_path.parent.mkdir(parents=True, exist_ok=True)
    if final_failed_path.exists():
        final_failed_path.unlink()

    run_meta = {
        "status": "running",
        "dataset": args.dataset,
        "protocol": "fixed_split",
        "subset": args.subset,
        "imputation": args.imputation,
        "started_at": datetime.now().isoformat(),
    }
    save_json(final_running_path, run_meta)

    ref = (
        f"dataset={args.dataset}__protocol=fixed_split__subset={args.subset}"
        f"__imputation={impute_label(args.imputation)}__model={args.model}"
        f"__repr={args.image_method.replace('/', '-')}__epochs={args.epochs}"
        f"__lr={args.lr:g}__wd={args.wd:g}__batch={args.batch_size}"
    )

    trainer_cfg = {
        "dataset_name": args.dataset,
        "ref": ref,
        "seed": args.seed,
        "validate_people": ["val"],
        "test_people": ["test"],
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
        "save_model": False,
    }

    start = time.time()
    try:
        result = Trainer(trainer_cfg).run(df)
        elapsed = time.time() - start
        payload = {
            "status": "completed",
            "dataset": args.dataset,
            "protocol": "fixed_split",
            "subset": args.subset,
            "subset_landmarks": len(subset_indices(args.subset)),
            "imputation": args.imputation,
            "imputation_label": impute_label(args.imputation),
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
        print(f"Resultado salvo em: {final_result_path}")
        print(json.dumps(payload["metrics"], indent=2, default=str))
    except Exception as exc:
        elapsed = time.time() - start
        payload = {
            "status": "failed",
            "dataset": args.dataset,
            "protocol": "fixed_split",
            "subset": args.subset,
            "imputation": args.imputation,
            "elapsed_seconds": elapsed,
            "failed_at": datetime.now().isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        save_json(final_failed_path, payload)
        if final_running_path.exists():
            final_running_path.unlink()
        print("[ERRO] Run falhou")
        print(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        tmp_ref_dir = trainer_tmp_dir / ref
        if tmp_ref_dir.exists():
            shutil.rmtree(tmp_ref_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

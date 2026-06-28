#!/usr/bin/env python3
"""
Generic nested-LOPO batch runner for ISLR landmark CSV datasets.

This script runs one subset + one imputation setting at a time, with resume by fold.
It is dataset-agnostic as long as the CSV has, or can be normalized to, these columns:
  - person/signer/interpreter
  - category/sign_id
  - video_name/sequence_id/path
  - frame/frame_id (optional; generated if missing)

Example:
python scripts/batch/run_dataset_batch.py \
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
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
from tqdm import tqdm

# File location: scripts/batch/run_dataset_batch.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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


def str_to_bool(value: Union[str, bool]) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "sim", "s", "with", "com"}:
        return True
    if value in {"false", "0", "no", "n", "nao", "não", "without", "sem"}:
        return False
    raise argparse.ArgumentTypeError("Valor booleano inválido: %s" % value)


def parse_max_runs(value: Optional[str]) -> Optional[int]:
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
        raise ValueError("Índice de landmark inválido ou sem colunas: %s" % idx)
    return strip_axis(cols[0])


def numeric_aliases_for_index(idx: int) -> list[str]:
    if idx < 468:
        return ["face_%s" % idx]
    if 468 <= idx < 489:
        h = idx - 468
        return ["hand_0_%s" % h, "hand_0_%02d" % h, "left_hand_%s" % h, "left_hand_%02d" % h]
    if 489 <= idx < 522:
        p = idx - 489
        return ["pose_%s" % p, "pose_%02d" % p, "body_%s" % p, "body_%02d" % p]
    if 522 <= idx < 543:
        h = idx - 522
        return ["hand_1_%s" % h, "hand_1_%02d" % h, "right_hand_%s" % h, "right_hand_%02d" % h]
    raise ValueError("Índice fora do intervalo MediaPipe Holistic: %s" % idx)


def expected_base_names_for_index(idx: int) -> list[str]:
    base_names = []
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


def find_existing_column(df: pd.DataFrame, base_names: list[str], axis: str) -> Optional[str]:
    candidates = []
    for base in base_names:
        candidates.extend(["%s_%s" % (base, axis), "%s.%s" % (base, axis), "%s:%s" % (base, axis)])
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def canonical_base_name(idx: int) -> str:
    # Stable numeric output names. LopoDataset will still detect x/y/z columns.
    if idx < 468:
        return "face_%s" % idx
    if 468 <= idx < 489:
        return "hand_0_%02d" % (idx - 468)
    if 489 <= idx < 522:
        return "pose_%02d" % (idx - 489)
    if 522 <= idx < 543:
        return "hand_1_%02d" % (idx - 522)
    raise ValueError("Índice fora do intervalo MediaPipe Holistic: %s" % idx)


def normalize_metadata(df: pd.DataFrame, person_col: Optional[str] = None, category_col: Optional[str] = None,
                       video_col: Optional[str] = None, frame_col: Optional[str] = None) -> pd.DataFrame:
    df = df.copy()

    explicit_renames = {}
    if person_col and person_col in df.columns and person_col != "person":
        explicit_renames[person_col] = "person"
    if category_col and category_col in df.columns and category_col != "category":
        explicit_renames[category_col] = "category"
    if video_col and video_col in df.columns and video_col != "video_name":
        explicit_renames[video_col] = "video_name"
    if frame_col and frame_col in df.columns and frame_col != "frame":
        explicit_renames[frame_col] = "frame"
    if explicit_renames:
        df = df.rename(columns=explicit_renames)

    rename_candidates = {
        "interpreter": "person",
        "signer": "person",
        "participant_id": "person",
        "frame_id": "frame",
        "sign_id": "category",
        "label": "category",
        "class": "category",
        "sequence_id": "video_name",
        "path": "video_name",
        "file": "video_name",
        "filename": "video_name",
    }
    for old, new in rename_candidates.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    required = ["person", "category", "video_name"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "CSV sem colunas obrigatórias %s. Colunas encontradas: %s..." % (missing, list(df.columns[:40]))
        )

    if "frame" not in df.columns:
        # fallback: preserve input order within each video
        df["frame"] = df.groupby("video_name").cumcount()

    # Keep original class text if useful.
    if "sign" not in df.columns:
        df["sign"] = df["category"]

    # Map categories to consecutive integers expected by the training code.
    categories = sorted(df["category"].dropna().unique(), key=lambda x: str(x))
    df["category"] = df["category"].map({c: i for i, c in enumerate(categories)})
    return df


def subset_indices(subset_name: str) -> list[int]:
    if subset_name not in LANDMARK_SUBSETS:
        raise ValueError("Subset desconhecido: %s. Disponíveis: %s" % (subset_name, list(LANDMARK_SUBSETS)))
    return LANDMARK_SUBSETS[subset_name]()


def prepare_subset_dataframe(raw_df: pd.DataFrame, subset: str, use_imputation: bool,
                             person_col: Optional[str] = None, category_col: Optional[str] = None,
                             video_col: Optional[str] = None, frame_col: Optional[str] = None) -> pd.DataFrame:
    df = normalize_metadata(raw_df, person_col=person_col, category_col=category_col, video_col=video_col, frame_col=frame_col)
    indices = subset_indices(subset)

    keep_meta = [c for c in DEFAULT_METADATA_COLUMNS if c in df.columns]
    for required in ["category", "video_name", "frame", "person"]:
        if required not in keep_meta:
            keep_meta.append(required)

    result = df[keep_meta].copy()
    missing_columns = []

    for idx in indices:
        for axis in ["x", "y", "z"]:
            source_col = find_existing_column(df, expected_base_names_for_index(idx), axis)
            output_col = "%s_%s" % (canonical_base_name(idx), axis)
            if source_col is None:
                result[output_col] = pd.NA
                missing_columns.append(output_col)
            else:
                result[output_col] = df[source_col]

    landmark_cols = [c for c in result.columns if c.endswith(("_x", "_y", "_z"))]

    if missing_columns:
        print("[WARN] %d colunas de landmark não encontradas; serão preenchidas com 0 após imputação/fillna." % len(missing_columns))
        print("       Exemplos:", missing_columns[:10])

    if use_imputation:
        result = impute_by_video(result, landmark_cols)
    else:
        result[landmark_cols] = result[landmark_cols].fillna(0)

    result = result.sort_values(["video_name", "frame"]).reset_index(drop=True)
    return result


def load_or_create_preprocessed(data_csv: Path, cache_dir: Path, dataset: str, subset: str, use_imputation: bool,
                                force: bool = False, person_col: Optional[str] = None,
                                category_col: Optional[str] = None, video_col: Optional[str] = None,
                                frame_col: Optional[str] = None) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / "%s_%s_%s.csv" % (dataset, subset, impute_label(use_imputation))
    if output_path.exists() and not force:
        print("Usando CSV pré-processado em cache: %s" % output_path)
        return output_path

    print("Lendo CSV bruto: %s" % data_csv)
    raw_df = pd.read_csv(data_csv, na_values=["NaN", "nan", ""])
    print("Linhas: %s | Colunas: %s" % (format(len(raw_df), ","), format(len(raw_df.columns), ",")))

    processed = prepare_subset_dataframe(
        raw_df,
        subset,
        use_imputation,
        person_col=person_col,
        category_col=category_col,
        video_col=video_col,
        frame_col=frame_col,
    )
    processed.to_csv(output_path, index=False)
    print("CSV pré-processado salvo: %s" % output_path)
    print("Shape processado: %s" % (processed.shape,))
    return output_path


def build_run_plan(df: pd.DataFrame, protocol: str = "nested_lopo", fixed_val_person: Optional[Any] = None) -> list[tuple[Any, Any]]:
    people = sorted(df["person"].dropna().unique(), key=lambda x: str(x))
    if len(people) < 2:
        raise ValueError("LOPO precisa de pelo menos 2 pessoas. Encontradas: %s" % people)

    if protocol == "nested_lopo":
        return [(test_person, val_person) for test_person in people for val_person in people if val_person != test_person]

    if protocol == "lopo_fixed_val":
        if fixed_val_person is None:
            # Deterministic fallback: for each test, first person different from test.
            return [(test_person, next(p for p in people if p != test_person)) for test_person in people]
        return [(test_person, fixed_val_person) for test_person in people if str(test_person) != str(fixed_val_person)]

    raise ValueError("Protocolo desconhecido: %s" % protocol)


def result_path_for(results_dir: Path, subset: str, use_imputation: bool, test_person: Any, val_person: Any) -> Path:
    return (
        results_dir
        / "runs"
        / subset
        / impute_label(use_imputation)
        / "test=%s__val=%s" % (sanitize_id(test_person), sanitize_id(val_person))
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
    parser = argparse.ArgumentParser(description="Run generic nested LOPO batch with resume by fold.")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g., ksl, include50, minds, ufop.")
    parser.add_argument("--data-csv", required=True, type=Path)
    parser.add_argument("--results-dir", required=True, type=Path)
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
    parser.add_argument("--protocol", default="nested_lopo", choices=["nested_lopo", "lopo_fixed_val"])
    parser.add_argument("--fixed-val-person", default=None, help="Used only with --protocol lopo_fixed_val.")
    parser.add_argument("--person-col", default=None, help="Optional raw CSV column to use as person.")
    parser.add_argument("--category-col", default=None, help="Optional raw CSV column to use as category.")
    parser.add_argument("--video-col", default=None, help="Optional raw CSV column to use as video_name.")
    parser.add_argument("--frame-col", default=None, help="Optional raw CSV column to use as frame.")
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.results_dir / "preprocessed"
    trainer_tmp_dir = args.results_dir / "_trainer_tmp_outputs"
    max_runs = parse_max_runs(args.max_runs)

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
    print("  dataset:      %s" % args.dataset)
    print("  data_csv:     %s" % args.data_csv)
    print("  results_dir:  %s" % args.results_dir)
    print("  subset:       %s (%s landmarks)" % (args.subset, len(subset_indices(args.subset))))
    print("  imputation:   %s" % impute_label(args.imputation))
    print("  protocol:     %s" % args.protocol)
    print("  epochs:       %s" % args.epochs)
    print("  lr/wd/batch:  %s / %s / %s" % (args.lr, args.wd, args.batch_size))
    print("  patience:     %s" % args.patience)
    print("  max_runs:     %s" % max_runs)
    print("  resume:       %s\n" % args.resume)

    processed_csv = load_or_create_preprocessed(
        data_csv=args.data_csv,
        cache_dir=cache_dir,
        dataset=args.dataset,
        subset=args.subset,
        use_imputation=args.imputation,
        force=args.force_preprocess,
        person_col=args.person_col,
        category_col=args.category_col,
        video_col=args.video_col,
        frame_col=args.frame_col,
    )
    df = pd.read_csv(processed_csv)
    run_plan = build_run_plan(df, protocol=args.protocol, fixed_val_person=args.fixed_val_person)

    total = len(run_plan)
    completed_before = sum(result_path_for(args.results_dir, args.subset, args.imputation, t, v).exists() for t, v in run_plan)
    failed_before = sum(failed_path_for(result_path_for(args.results_dir, args.subset, args.imputation, t, v)).exists() for t, v in run_plan)

    print("Plano: %d runs para dataset=%s, subset=%s, imputation=%s" % (total, args.dataset, args.subset, impute_label(args.imputation)))
    print("Antes: completed=%d, failed=%d, pending=%d\n" % (completed_before, failed_before, total - completed_before))

    executed_now = 0
    skipped_completed = 0
    skipped_failed = 0
    failed_now = 0

    pbar = tqdm(run_plan, total=total, desc=args.protocol)
    for test_person, val_person in pbar:
        desc = "%s | %s | test=%s | val=%s" % (args.subset, impute_label(args.imputation), test_person, val_person)
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
            "protocol": args.protocol,
            "subset": args.subset,
            "imputation": args.imputation,
            "test_person": test_person,
            "val_person": val_person,
            "started_at": datetime.now().isoformat(),
        }
        save_json(final_running_path, run_meta)

        ref = (
            "dataset=%s__protocol=%s__subset=%s"
            "__imputation=%s__model=%s"
            "__repr=%s__epochs=%s"
            "__lr=%g__wd=%g__batch=%s"
            "__test=%s__val=%s"
            % (
                args.dataset,
                args.protocol,
                args.subset,
                impute_label(args.imputation),
                args.model,
                args.image_method.replace("/", "-"),
                args.epochs,
                args.lr,
                args.wd,
                args.batch_size,
                sanitize_id(test_person),
                sanitize_id(val_person),
            )
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
            "save_model": False,
        }

        start = time.time()
        try:
            result = Trainer(trainer_cfg).run(df)
            elapsed = time.time() - start
            payload = {
                "status": "completed",
                "dataset": args.dataset,
                "protocol": args.protocol,
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
            print("Resultado salvo em: %s" % final_result_path)
        except Exception as exc:
            elapsed = time.time() - start
            failed_now += 1
            payload = {
                "status": "failed",
                "dataset": args.dataset,
                "protocol": args.protocol,
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
            print("[ERRO] Run falhou: %s" % desc)
            print("       %s: %s" % (type(exc).__name__, exc))
        finally:
            tmp_ref_dir = trainer_tmp_dir / ref
            if tmp_ref_dir.exists():
                shutil.rmtree(tmp_ref_dir, ignore_errors=True)

    completed_after = sum(result_path_for(args.results_dir, args.subset, args.imputation, t, v).exists() for t, v in run_plan)
    failed_after = sum(failed_path_for(result_path_for(args.results_dir, args.subset, args.imputation, t, v)).exists() for t, v in run_plan)

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
    print("Progresso deste subset/imputação: %d/%d (%.1f%%)" % (completed_after, total, completed_after / total * 100))


if __name__ == "__main__":
    main()

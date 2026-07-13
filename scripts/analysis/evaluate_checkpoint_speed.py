#!/usr/bin/env python3
"""
Evaluate checkpoint speed and metrics for the ISLR repository.

Compatible with the current repository layout:

  src/
  scripts/
  data/interim/...
  experiments/...

Main use case: evaluate inference speed of a trained checkpoint on a CSV of
MediaPipe landmarks, using the same Skeleton-DML + model pipeline used in
training.

Example for INCLUDE-50 split test set:

python scripts/analysis/evaluate_checkpoint_speed.py \
  --dataset include50 \
  --data-csv data/interim/include50/include50_mediapipe_with_split.csv \
  --checkpoint-path experiments/include50_split_grid/.../best_model.pth \
  --subset 2nd \
  --imputation true \
  --category-col sign_id \
  --video-col sequence_id \
  --frame-col frame_id \
  --person-col split \
  --eval-person test \
  --device cuda

By default, the preprocessed subset is kept in memory to avoid creating large
extra CSV files. Add --cache-preprocessed only if you have enough disk space.

For MINDS/KSL/UFOP with LOPO-style CSVs, use person/interpreter/signer as the
person column and set --eval-person to the test signer/person used by the
checkpoint.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


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

# Project imports.
from preprocessing.landmark_subsets import SUBSETS as LANDMARK_SUBSETS, indices_to_columns  # noqa: E402
from preprocessing.imputation import impute_by_video  # noqa: E402
from training.lopo_dataset import LopoDataset  # noqa: E402
from models.base_model import BaseModel  # noqa: E402

# Register the classes used by the repository registries.
import models.resnet18  # noqa: E402,F401
import representations.skeleton_dml  # noqa: E402,F401

try:
    from representations.base_representation import BaseRepresentation  # noqa: E402
except Exception as exc:  # pragma: no cover - defensive error message
    raise ImportError(
        "Não consegui importar representations.base_representation.BaseRepresentation. "
        "Confira o nome do arquivo/classe em src/representations/."
    ) from exc


DEFAULT_META_COLUMNS = [
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


def sanitize_id(value: Any) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text)
    return text.strip("-") or "unknown"


def impute_label(use_imputation: bool) -> str:
    return "with_imputation" if use_imputation else "without_imputation"


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


def subset_indices(subset_name: str) -> list[int]:
    if subset_name not in LANDMARK_SUBSETS:
        raise ValueError(f"Subset desconhecido: {subset_name}. Disponíveis: {list(LANDMARK_SUBSETS)}")
    return LANDMARK_SUBSETS[subset_name]()


def normalize_metadata(
    raw_df: pd.DataFrame,
    category_col: str,
    video_col: str,
    frame_col: str,
    person_col: str,
) -> pd.DataFrame:
    """Create the columns expected by LopoDataset: category, video_name, frame, person."""
    df = raw_df.copy()

    required = [category_col, video_col, frame_col, person_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"CSV sem colunas obrigatórias {missing}. "
            f"Colunas encontradas: {list(df.columns[:60])}..."
        )

    df["category"] = df[category_col]
    # Keep numeric class ids when they already exist. Otherwise map labels to consecutive ids.
    if not pd.api.types.is_numeric_dtype(df["category"]):
        categories = sorted(df["category"].dropna().unique(), key=lambda x: str(x))
        df["category"] = df["category"].map({c: i for i, c in enumerate(categories)})
    df["category"] = df["category"].astype(int)

    df["video_name"] = df[video_col].astype(str)
    df["frame"] = df[frame_col].astype(int)
    df["person"] = df[person_col].astype(str).str.strip()

    return df


def prepare_subset_dataframe(
    raw_df: pd.DataFrame,
    subset: str,
    use_imputation: bool,
    category_col: str,
    video_col: str,
    frame_col: str,
    person_col: str,
) -> pd.DataFrame:
    df = normalize_metadata(raw_df, category_col, video_col, frame_col, person_col)
    indices = subset_indices(subset)

    keep_meta = [col for col in DEFAULT_META_COLUMNS if col in df.columns]
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


def load_or_create_preprocessed_df(
    data_csv: Path,
    cache_dir: Path,
    dataset: str,
    subset: str,
    use_imputation: bool,
    category_col: str,
    video_col: str,
    frame_col: str,
    person_col: str,
    force: bool = False,
    cache_preprocessed: bool = False,
) -> pd.DataFrame:
    """Load and preprocess the CSV.

    By default this works in memory and does not write a second large CSV to disk.
    Use --cache-preprocessed only when you plan to run the same speed evaluation
    repeatedly and have enough disk space.
    """
    cache_name = (
        f"{dataset}_{subset}_{impute_label(use_imputation)}"
        f"__cat={sanitize_id(category_col)}__video={sanitize_id(video_col)}"
        f"__frame={sanitize_id(frame_col)}__person={sanitize_id(person_col)}.csv"
    )
    output_path = cache_dir / cache_name

    if cache_preprocessed and output_path.exists() and not force:
        print(f"Usando CSV pré-processado em cache: {output_path}")
        processed = pd.read_csv(output_path)
    else:
        print(f"Lendo CSV bruto: {data_csv}")
        raw_df = pd.read_csv(data_csv, na_values=["NaN", "nan", ""])
        print(f"Linhas: {len(raw_df):,} | Colunas: {len(raw_df.columns):,}")

        processed = prepare_subset_dataframe(
            raw_df=raw_df,
            subset=subset,
            use_imputation=use_imputation,
            category_col=category_col,
            video_col=video_col,
            frame_col=frame_col,
            person_col=person_col,
        )

        if cache_preprocessed:
            cache_dir.mkdir(parents=True, exist_ok=True)
            processed.to_csv(output_path, index=False)
            print(f"CSV pré-processado salvo em cache: {output_path}")

    videos = processed[["video_name", "person", "category"]].drop_duplicates()
    print("\nDistribuição por person/split em vídeos:")
    print(videos["person"].value_counts().to_string())
    print("\nClasses por person/split:")
    print(videos.groupby("person")["category"].nunique().to_string())
    print(f"Shape processado: {processed.shape}")
    return processed


def get_image_method(image_method_name: str):
    image_method_type = BaseRepresentation.get_by_name(image_method_name)
    if image_method_type is None:
        available = getattr(BaseRepresentation, "_registry", None)
        raise ValueError(f"Representação não encontrada: {image_method_name}. Registry: {available}")
    return image_method_type()


def build_model(model_name: str, num_classes: int):
    model_type = BaseModel.get_by_name(model_name)
    if model_type is None:
        available = getattr(BaseModel, "_registry", None)
        raise ValueError(f"Modelo não encontrado: {model_name}. Registry: {available}")
    base_model = model_type(num_classes)
    model = base_model.get_model()
    image_size = getattr(base_model, "image_size", (224, 224))
    return base_model, model, image_size


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict.keys()):
        return {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    return state_dict


def load_checkpoint_into_model(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {checkpoint_path}")

    print(f"Carregando checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        raise ValueError("Formato de checkpoint inválido. Esperava dict/state_dict.")

    state_dict = strip_module_prefix(state_dict)
    model.load_state_dict(state_dict)


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested in {"cuda", "auto"} and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        print("[WARN] --device cuda pedido, mas CUDA não está disponível. Usando CPU.")
    return torch.device("cpu")


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def save_confusion_matrix(path: Path, labels: list[int], predictions: list[int], title: str) -> None:
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint metrics and inference speed.")
    parser.add_argument("--dataset", default="include50")
    parser.add_argument("--data-csv", required=True, type=Path)
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument("--output-dir", default=None, type=Path)

    parser.add_argument("--subset", required=True, choices=list(LANDMARK_SUBSETS.keys()))
    parser.add_argument("--imputation", required=True, type=str_to_bool)

    parser.add_argument("--category-col", default="sign_id")
    parser.add_argument("--video-col", default="sequence_id")
    parser.add_argument("--frame-col", default="frame_id")
    parser.add_argument("--person-col", default="split")
    parser.add_argument("--eval-person", default="test", help="Value from --person-col to evaluate, e.g. test or signer id.")

    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--image-method", default="Skeleton-DML")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--cache-preprocessed", action="store_true", help="Save/use a preprocessed CSV cache. Disabled by default to save disk space.")
    parser.add_argument("--save-confusion-matrix", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        args.output_dir = PROJECT_ROOT / "reports" / "speed" / f"{args.dataset}_{args.subset}_{impute_label(args.imputation)}_{timestamp}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = PROJECT_ROOT / "experiments" / f"{args.dataset}_speed_eval_cache" / "preprocessed"

    print("=" * 70)
    print("Checkpoint speed evaluation")
    print(f"Projeto:       {PROJECT_ROOT}")
    print(f"Dataset:       {args.dataset}")
    print(f"CSV:           {args.data_csv}")
    print(f"Checkpoint:    {args.checkpoint_path}")
    print(f"Subset:        {args.subset} ({len(subset_indices(args.subset))} landmarks)")
    print(f"Imputation:    {impute_label(args.imputation)}")
    print(f"Eval person:   {args.eval_person}")
    print(f"Output dir:    {args.output_dir}")
    print("=" * 70)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = load_or_create_preprocessed_df(
        data_csv=args.data_csv,
        cache_dir=cache_dir,
        dataset=args.dataset,
        subset=args.subset,
        use_imputation=args.imputation,
        category_col=args.category_col,
        video_col=args.video_col,
        frame_col=args.frame_col,
        person_col=args.person_col,
        force=args.force_preprocess,
        cache_preprocessed=args.cache_preprocessed,
    )
    num_classes = int(df["category"].nunique())
    eval_df = df[df["person"].astype(str) == str(args.eval_person)].copy()
    if eval_df.empty:
        found = sorted(df["person"].astype(str).unique().tolist())
        raise ValueError(f"Nenhuma amostra encontrada para eval-person={args.eval_person}. Encontrados: {found}")

    eval_videos = eval_df["video_name"].nunique()
    print(f"\nAmostras/frames no split avaliado: {len(eval_df):,}")
    print(f"Vídeos no split avaliado:          {eval_videos:,}")
    print(f"Classes no dataset completo:       {num_classes}")
    print(f"Classes no split avaliado:         {eval_df['category'].nunique()}")

    image_method = get_image_method(args.image_method)
    base_model, model, image_size = build_model(args.model, num_classes)

    device = resolve_device(args.device)
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Torch: {torch.__version__} | CUDA: {torch.version.cuda}")

    load_checkpoint_into_model(model, args.checkpoint_path, device)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    test_dataset = LopoDataset(
        df,
        transform,
        transform_distance=False,
        augment=False,
        seed=args.seed,
        image_method=image_method,
        person_in=[args.eval_person],
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(f"\nDataset de avaliação: {len(test_dataset)} vídeos/amostras")
    print(f"Batch size:           {args.batch_size}")
    print(f"Total batches:        {len(test_loader)}")

    # Warmup.
    if args.warmup_iters > 0:
        print(f"\nWarmup: {args.warmup_iters} iterações")
        with torch.no_grad():
            dummy = torch.randn(1, 3, image_size[0], image_size[1], device=device)
            for _ in range(args.warmup_iters):
                _ = model(dummy)
            sync_if_cuda(device)

    all_predictions: list[int] = []
    all_labels: list[int] = []
    batch_durations_ms: list[float] = []

    print("\nIniciando avaliação...")
    total_start = time.perf_counter()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            sync_if_cuda(device)
            batch_start = time.perf_counter()
            outputs = model(data)
            sync_if_cuda(device)
            batch_end = time.perf_counter()

            predicted = torch.argmax(outputs, dim=1)
            all_predictions.extend(predicted.detach().cpu().numpy().astype(int).tolist())
            all_labels.extend(target.detach().cpu().numpy().astype(int).tolist())
            batch_durations_ms.append((batch_end - batch_start) * 1000.0)

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx:>5}/{len(test_loader)} | {batch_durations_ms[-1]:.2f} ms")

    sync_if_cuda(device)
    total_end = time.perf_counter()
    total_seconds = total_end - total_start

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_predictions, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average="weighted", zero_division=0)

    batch_arr = np.array(batch_durations_ms, dtype=float)
    num_samples = len(all_labels)

    results = {
        "dataset": args.dataset,
        "protocol": "checkpoint_speed_eval",
        "data_csv": str(args.data_csv),
        "checkpoint_path": str(args.checkpoint_path),
        "subset": args.subset,
        "subset_landmarks": len(subset_indices(args.subset)),
        "imputation": args.imputation,
        "imputation_label": impute_label(args.imputation),
        "category_col": args.category_col,
        "video_col": args.video_col,
        "frame_col": args.frame_col,
        "person_col": args.person_col,
        "eval_person": args.eval_person,
        "model": args.model,
        "image_method": args.image_method,
        "seed": args.seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "num_classes_full_dataset": num_classes,
        "num_eval_samples": num_samples,
        "num_eval_videos": len(test_dataset),
        "batch_size": args.batch_size,
        "warmup_iters": args.warmup_iters,
        "end_to_end_prediction_seconds": float(total_seconds),
        "avg_end_to_end_time_per_sample_ms": float(total_seconds / max(num_samples, 1) * 1000.0),
        "end_to_end_samples_per_second": float(num_samples / total_seconds) if total_seconds > 0 else None,
        "batch_timing_ms": {
            "min": float(batch_arr.min()) if len(batch_arr) else None,
            "max": float(batch_arr.max()) if len(batch_arr) else None,
            "mean": float(batch_arr.mean()) if len(batch_arr) else None,
            "std": float(batch_arr.std()) if len(batch_arr) else None,
            "median": float(np.median(batch_arr)) if len(batch_arr) else None,
        },
        "metrics": {
            "accuracy": float(accuracy),
            "precision_weighted": float(precision),
            "recall_weighted": float(recall),
            "f1_weighted": float(f1),
        },
        "batch_durations_ms": batch_durations_ms,
        "evaluated_at": datetime.now().isoformat(),
    }

    json_path = args.output_dir / "speed_eval_results.json"
    save_json(json_path, results)

    if args.save_confusion_matrix:
        cm_path = args.output_dir / "confusion_matrix.png"
        save_confusion_matrix(
            cm_path,
            all_labels,
            all_predictions,
            title=f"Confusion Matrix - {args.dataset.upper()} - {args.subset}",
        )
    else:
        cm_path = None

    print("\n" + "=" * 70)
    print("RESULTADOS")
    print(f"Samples avaliadas:      {num_samples}")
    print(f"Vídeos avaliados:       {len(test_dataset)}")
    print(f"Tempo total end-to-end: {total_seconds:.4f} s")
    print(f"Tempo médio/amostra:    {results['avg_end_to_end_time_per_sample_ms']:.2f} ms")
    print(f"Samples por segundo:    {results['end_to_end_samples_per_second']:.2f}")
    print(f"Forward batch mean±std: {results['batch_timing_ms']['mean']:.2f} ± {results['batch_timing_ms']['std']:.2f} ms")
    print("-" * 70)
    print(f"Accuracy:               {accuracy:.4f}")
    print(f"Precision weighted:     {precision:.4f}")
    print(f"Recall weighted:        {recall:.4f}")
    print(f"F1 weighted:            {f1:.4f}")
    print("-" * 70)
    print(f"JSON salvo em:          {json_path}")
    if cm_path is not None:
        print(f"Matriz salva em:        {cm_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

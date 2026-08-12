#!/usr/bin/env python3
from __future__ import annotations

"""Check the local environment before running extraction or training."""

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
EXPECTED_SUBSET_COUNTS = {
    "all": 543,
    "1st": 118,
    "2nd": 80,
    "laines": 67,
    "arcanjo": 75,
}


def print_check(label: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {label}")
    if detail:
        print(f"       {detail}")


def check_python() -> None:
    version = sys.version_info
    supported = version.major == 3 and 8 <= version.minor <= 10
    print_check(
        "Python version",
        supported,
        f"{sys.version.split()[0]} at {sys.executable}; recommended: Python 3.10",
    )


def check_venv() -> None:
    in_virtual_environment = (
        hasattr(sys, "real_prefix")
        or sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    )
    print_check(
        "Virtual environment",
        in_virtual_environment,
        f"sys.prefix={sys.prefix}",
    )


def check_paths() -> None:
    print_check("Project root", PROJECT_ROOT.exists(), str(PROJECT_ROOT))
    print_check("src directory", SRC_DIR.exists(), str(SRC_DIR))
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    print_check("src on sys.path", str(SRC_DIR) in sys.path, str(SRC_DIR))


def check_package(package_name: str) -> None:
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, "__version__", "version not reported")
        print_check(package_name, True, str(version))
    except Exception as error:
        print_check(package_name, False, repr(error))


def check_torch() -> None:
    try:
        import torch

        print_check("torch", True, torch.__version__)
        cuda_available = torch.cuda.is_available()
        gpu_name = (
            torch.cuda.get_device_name(0)
            if cuda_available
            else "No CUDA GPU detected"
        )
        print_check("CUDA available in PyTorch", cuda_available, gpu_name)
    except Exception as error:
        print_check("torch/CUDA", False, repr(error))


def check_landmark_subsets() -> None:
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    try:
        from preprocessing.landmark_subsets import SUBSETS

        print_check("Import preprocessing.landmark_subsets", True)
        for subset_name, expected_count in EXPECTED_SUBSET_COUNTS.items():
            if subset_name not in SUBSETS:
                print_check(f"Subset {subset_name}", False, "not found")
                continue
            count = len(SUBSETS[subset_name]())
            print_check(
                f"Subset {subset_name}",
                count == expected_count,
                f"found={count}, expected={expected_count}",
            )
    except Exception as error:
        print_check("Import landmark subsets", False, repr(error))


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def check_dataset_configs() -> None:
    config_dir = PROJECT_ROOT / "configs" / "datasets"
    print_check("configs/datasets directory", config_dir.exists(), str(config_dir))
    for config_path in sorted(config_dir.glob("*.env")):
        config = read_env_file(config_path)
        dataset = config.get("DATASET", config_path.stem)
        data_csv = PROJECT_ROOT / config.get("DATA_CSV", "")
        results_dir = PROJECT_ROOT / config.get("RESULTS_DIR", "")
        print_check(f"Configuration: {dataset}", True, str(config_path))
        print_check(f"CSV: {dataset}", data_csv.exists(), str(data_csv))
        results_dir.mkdir(parents=True, exist_ok=True)
        print_check(f"Results directory: {dataset}", results_dir.exists(), str(results_dir))


def check_output_directories() -> None:
    for directory in (
        PROJECT_ROOT / "experiments",
        PROJECT_ROOT / "logs",
        PROJECT_ROOT / "reports",
    ):
        directory.mkdir(parents=True, exist_ok=True)
        print_check(f"Output directory: {directory.name}", directory.exists(), str(directory))


def main() -> None:
    print("=" * 80)
    print("ISLR ENVIRONMENT CHECK")
    print("=" * 80)

    print("\n[1] Python and virtual environment")
    check_python()
    check_venv()

    print("\n[2] Repository paths")
    check_paths()

    print("\n[3] Core packages")
    for package in (
        "numpy",
        "pandas",
        "sklearn",
        "PIL",
        "cv2",
        "tqdm",
        "matplotlib",
        "timm",
    ):
        check_package(package)

    print("\n[4] PyTorch and GPU")
    check_torch()

    print("\n[5] Landmark subsets")
    check_landmark_subsets()

    print("\n[6] Dataset configurations")
    check_dataset_configs()

    print("\n[7] Output directories")
    check_output_directories()

    print("\n" + "=" * 80)
    print("Environment check completed.")
    print("=" * 80)


if __name__ == "__main__":
    main()

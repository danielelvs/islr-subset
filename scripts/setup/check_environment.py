#!/usr/bin/env python3
from __future__ import annotations

import importlib
import os
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
    mark = "✅" if ok else "❌"
    status = "OK" if ok else "ERRO"
    print(f"{mark} {label}: {status}")
    if detail:
        print(f"   {detail}")


def check_python() -> None:
    version = sys.version_info
    ok = version.major == 3 and version.minor >= 8
    print_check("Python", ok, f"{sys.version.split()[0]} em {sys.executable}")


def check_venv() -> None:
    in_venv = hasattr(sys, "real_prefix") or sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    print_check("Ambiente virtual", in_venv, f"sys.prefix={sys.prefix}")


def check_paths() -> None:
    print_check("Raiz do projeto", PROJECT_ROOT.exists(), str(PROJECT_ROOT))
    print_check("Pasta src", SRC_DIR.exists(), str(SRC_DIR))
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    print_check("src no sys.path", str(SRC_DIR) in sys.path, str(SRC_DIR))


def check_package(package_name: str) -> None:
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, "__version__", "versão não informada")
        print_check(package_name, True, str(version))
    except Exception as exc:
        print_check(package_name, False, repr(exc))


def check_torch() -> None:
    try:
        import torch
        print_check("torch", True, torch.__version__)
        cuda_ok = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "GPU não detectada"
        print_check("CUDA disponível no PyTorch", cuda_ok, gpu_name)
    except Exception as exc:
        print_check("torch/CUDA", False, repr(exc))


def check_landmark_subsets() -> None:
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    try:
        from preprocessing.landmark_subsets import SUBSETS
        print_check("Import preprocessing.landmark_subsets", True)
        for subset_name, expected_count in EXPECTED_SUBSET_COUNTS.items():
            if subset_name not in SUBSETS:
                print_check(f"Subset {subset_name}", False, "não encontrado")
                continue
            count = len(SUBSETS[subset_name]())
            print_check(f"Subset {subset_name}", count == expected_count, f"encontrado={count}, esperado={expected_count}")
    except Exception as exc:
        print_check("Import dos subsets", False, repr(exc))


def read_env_file(path: Path) -> dict[str, str]:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def check_dataset_configs() -> None:
    config_dir = PROJECT_ROOT / "configs" / "datasets"
    print_check("Pasta configs/datasets", config_dir.exists(), str(config_dir))
    for cfg in sorted(config_dir.glob("*.env")):
        data = read_env_file(cfg)
        dataset = data.get("DATASET", cfg.stem)
        data_csv = PROJECT_ROOT / data.get("DATA_CSV", "")
        results_dir = PROJECT_ROOT / data.get("RESULTS_DIR", "")
        print_check(f"Config {dataset}", True, str(cfg))
        print_check(f"CSV {dataset}", data_csv.exists(), str(data_csv))
        results_dir.mkdir(parents=True, exist_ok=True)
        print_check(f"Results dir {dataset}", results_dir.exists(), str(results_dir))


def check_output_dirs() -> None:
    for directory in [PROJECT_ROOT / "experiments", PROJECT_ROOT / "logs", PROJECT_ROOT / "reports"]:
        directory.mkdir(parents=True, exist_ok=True)
        print_check(f"Pasta {directory.name}", directory.exists(), str(directory))


def main() -> None:
    print("=" * 80)
    print("CHECK ENVIRONMENT - ISLR SUBSET")
    print("=" * 80)

    print("\n[1] Python e ambiente")
    check_python()
    check_venv()

    print("\n[2] Caminhos")
    check_paths()

    print("\n[3] Pacotes principais")
    for package in ["numpy", "pandas", "sklearn", "PIL", "cv2", "tqdm", "matplotlib", "timm"]:
        check_package(package)

    print("\n[4] PyTorch/GPU")
    check_torch()

    print("\n[5] Subsets")
    check_landmark_subsets()

    print("\n[6] Configs dos datasets")
    check_dataset_configs()

    print("\n[7] Pastas de saída")
    check_output_dirs()

    print("\n" + "=" * 80)
    print("Fim do check.")
    print("=" * 80)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys


def main():
    try:
        import torch
        print("torch:", torch.__version__)
        print("CUDA disponível:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
    except Exception as exc:
        print("Erro ao importar torch:", repr(exc))

    print("\nnvidia-smi:")
    try:
        subprocess.run(["nvidia-smi"], check=False)
    except FileNotFoundError:
        print("nvidia-smi não encontrado.")
        sys.exit(0)


if __name__ == "__main__":
    main()

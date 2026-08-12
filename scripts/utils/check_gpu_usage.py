#!/usr/bin/env python3
from __future__ import annotations

import subprocess


def main() -> None:
    try:
        import torch

        print("PyTorch:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
    except Exception as error:
        print("Failed to import PyTorch:", repr(error))

    print("\nnvidia-smi:")
    try:
        subprocess.run(["nvidia-smi"], check=False)
    except FileNotFoundError:
        print("nvidia-smi was not found.")


if __name__ == "__main__":
    main()

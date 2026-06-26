#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess


def main() -> None:
    if shutil.which("nvidia-smi"):
        print("[nvidia-smi]")
        subprocess.run(["nvidia-smi"], check=False)
    else:
        print("nvidia-smi não encontrado.")

    print("\n[torch]")
    try:
        import torch
        print("torch:", torch.__version__)
        print("CUDA disponível:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
            print("Memória alocada:", torch.cuda.memory_allocated(0))
    except Exception as exc:
        print("Erro ao importar/testar torch:", repr(exc))


if __name__ == "__main__":
    main()

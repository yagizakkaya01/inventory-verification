"""Verify the training stack sees the GPU. Run after installing PyTorch."""

from __future__ import annotations


def main() -> int:
    try:
        import torch
    except ImportError:
        print("torch not installed yet — see scripts/setup_env.ps1 step 3")
        return 1

    print(f"torch            {torch.__version__}")
    print(f"cuda available   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda version     {torch.version.cuda}")
        print(f"device count     {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  [{i}] {torch.cuda.get_device_name(i)}")
    else:
        print("!! CUDA not available — training will run on CPU (slow)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

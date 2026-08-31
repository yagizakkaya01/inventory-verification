# Environment setup for the desktop / GPU training machine (Windows + PowerShell).
# Run from the repo root:  .\scripts\setup_env.ps1
#
# Deliberately does NOT install PyTorch — that must be matched to the machine's
# CUDA version. The script prints the command to run after inspecting the GPU.

$ErrorActionPreference = "Stop"

$PyVersion = "3.11"   # ultralytics + torch + Jetson toolchain all happy here

Write-Host "== 1. Python venv ==" -ForegroundColor Cyan
if (-not (Test-Path ".venv")) {
    py -$PyVersion -m venv .venv
    Write-Host "created .venv (Python $PyVersion)"
} else {
    Write-Host ".venv already exists"
}
& .\.venv\Scripts\Activate.ps1
python --version

Write-Host "`n== 2. GPU check ==" -ForegroundColor Cyan
try { nvidia-smi } catch { Write-Warning "nvidia-smi not found — CPU-only machine?" }

Write-Host "`n== 3. Install PyTorch MANUALLY, matched to the CUDA version above ==" -ForegroundColor Yellow
Write-Host "   https://pytorch.org/get-started/locally/"
Write-Host "   e.g. CUDA 12.1:"
Write-Host "   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
Write-Host "   Then verify:  python -c `"import torch; print(torch.cuda.is_available())`""

Write-Host "`n== 4. Project deps ==" -ForegroundColor Cyan
python -m pip install --upgrade pip
Write-Host "After PyTorch is in:  pip install -r requirements.txt"
Write-Host "Then:                 python scripts\check_gpu.py; python scripts\smoke_test.py"

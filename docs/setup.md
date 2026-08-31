# Environment setup

Order matters — PyTorch must match the machine's CUDA before anything else.

## Training / desktop machine (Windows)

1. **Git** — done (repo initialized).
2. **Isolated Python env** — Python **3.10 or 3.11** (not 3.13; several CV/edge
   deps and the Jetson toolchain lag). `.\scripts\setup_env.ps1` creates `.venv`.
3. **GPU check** — `nvidia-smi`. Note the CUDA version shown top-right.
4. **PyTorch** — install matched to that CUDA from
   <https://pytorch.org/get-started/locally/>.
   This machine (RTX 4070 Laptop, driver CUDA 13.3 — forward-compatible):
   `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`
5. **Verify** — `python -c "import torch; print(torch.cuda.is_available())"` → `True`
6. **Project deps** — `pip install -r requirements.txt`
7. **Smoke test** — `python scripts\check_gpu.py` then `python scripts\smoke_test.py`

## Camera SDKs (add when the hardware is in hand)

| Device | Package | Notes |
|--------|---------|-------|
| OAK-D | `pip install depthai` | also `blobconverter` for `.blob` export |
| Orbbec Astra | OpenNI2 / `pyorbbecsdk` | vendor SDK + Python bindings |
| Endoscope / USB | none | plain `cv2.VideoCapture` |

## Jetson Nano (separate machine)

- Flash the **JetPack** image (bundles CUDA, cuDNN, TensorRT).
- Use the NVIDIA-provided PyTorch wheel for that JetPack version — do **not**
  `pip install torch` from PyPI.
- TensorRT is already present; `src/edge/export_trt.py` builds the engine.

## Open items before starting

See [open-questions.md](open-questions.md) — machine/GPU choice, company setup
doc, existing dataset, pip/proxy restrictions.

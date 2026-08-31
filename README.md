# Inventory / State Verification System (Edge AI)

Fixed camera watches a defined work area, detects a small set of objects, and
verifies the **state** of the scene against an expected configuration:

- **missing part** — an expected object is not present
- **wrong order** — objects present but arranged in the wrong sequence
- **wrong combination** — an unexpected set of objects

Output per checked frame window: `OK` or a typed error.

## Architecture

Two layers:

| Layer | Responsibility | Tech |
|-------|----------------|------|
| **Detection** | Per-frame object detection | YOLO (transfer-learning fine-tune, Ultralytics) |
| **State logic** | Temporal smoothing (N-frame consistency) → state machine → compare vs expected → `OK` / error | Plain Python |

Target: run the full pipeline in real time on edge hardware
(OAK-D, Jetson Nano, Orbbec Astra), with a benchmark vs desktop GPU.

## Repository layout

```
configs/        YAML config: dataset, training hyperparams, runtime pipeline
data/           raw captures / interim / YOLO-format datasets   (gitignored)
models/         pretrained weights / checkpoints / exports      (gitignored)
src/
  detection/    YOLO wrapper + training entrypoint
  state/        temporal smoothing, state machine, verifier
  pipeline/     real-time loop + camera sources
  edge/         ONNX / TensorRT / OAK-D blob export
  utils/        config loading, visualization
scripts/        env setup, GPU check, smoke test
benchmarks/     FPS / latency / accuracy-drop measurement
tests/          unit tests (state machine logic first)
docs/           plan, scenarios, metrics, open questions
notebooks/      exploration
```

## Getting started

See [docs/setup.md](docs/setup.md) for the full environment setup sequence.
Quick version (Windows / PowerShell, desktop GPU machine):

```powershell
.\scripts\setup_env.ps1
python scripts\check_gpu.py
python scripts\smoke_test.py
```

## Status

Early setup. Timeline and open questions: [docs/plan.md](docs/plan.md),
[docs/open-questions.md](docs/open-questions.md).

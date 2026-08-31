# Metrics (for the CV write-up)

## Detection (model quality)

| Metric | How | Target |
|--------|-----|--------|
| mAP@0.5 | `YOLO(w).val(data=configs/data.yaml)` | ≥ 0.90 |
| mAP@0.5:0.95 | same | track, no hard target |
| precision / recall per class | val output | recall ≥ 0.95 on item_c (occlusion) |

## System (verification quality)

Run the 10 test scenarios (`docs/scenarios.md`), N repeats each.

| Metric | Definition |
|--------|-----------|
| error-catch rate | designed errors correctly flagged / total designed errors |
| false-alarm rate | OK scenes flagged as error / total OK frames windows |
| miss rate | error scenes reported OK / total error windows |
| latency-to-detect | frames from state change to confirmed transition |

## Edge (deployment)

Per target (desktop GPU / Jetson / OAK-D) and precision (FP32 / FP16 / INT8):

| Metric | How |
|--------|-----|
| FPS (mean) | `benchmarks/benchmark.py` |
| latency p50 / p95 (ms) | same |
| mAP after quantization | `.val()` on the exported engine |
| accuracy drop | mAP(FP32) − mAP(INT8) |

Results land in `benchmarks/results/*.json`; summarize as one table in the report.

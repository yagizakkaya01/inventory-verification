# 20-day plan

| Days | Phase | Output |
|------|-------|--------|
| 1-2 | Scenario design, camera rig, setup | `docs/scenarios.md` filled, env working, rig placed |
| 3-6 | Data collection + labeling | ~300 images, 70/20/10 split, Roboflow project |
| 7-10 | YOLO training + accuracy | fine-tuned `best.pt`, mAP / P / R recorded |
| 11-14 | State machine + temporal smoothing + real-time | `src/pipeline/run.py` live on webcam |
| 15-17 | Edge deployment + benchmark | TensorRT/INT8 engine, OAK-D blob, `benchmarks/results/` |
| 18-20 | Error-scenario testing, docs, demo video | test report, README, demo clip |

## Milestones / gates

- **Day 2** — rig fixed, lighting stable, scenarios agreed with Ömer.
- **Day 4** — first ~50-image model trained end-to-end to validate the whole
  loop (capture → label → train → infer). Accuracy irrelevant here.
- **Day 10** — detection good enough (target mAP@0.5 ≥ 0.9 on the fixed scene).
- **Day 14** — full pipeline catches all designed error scenarios on live video.
- **Day 17** — edge numbers in hand; know the FP16/INT8 accuracy cost.

## Data collection guidance

- Vary: camera angle (small — rig is fixed), lighting, object arrangement,
  partial occlusion, distractor objects.
- **Error scenarios must be in the dataset**, not just the test script — the
  detector needs to see all three objects in every plausible position.
- Label with Roboflow model-assisted labeling once the 50-image model exists.

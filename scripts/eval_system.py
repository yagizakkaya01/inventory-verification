"""System-level check: run detector + verifier on the raw capture images and
see whether the emitted verdict matches what the scenario folder implies.

This is the metric that matters for the demo — not just mAP, but "does the
whole pipeline call the scene correctly".

    python -m scripts.eval_system --split test
    python -m scripts.eval_system --split all
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from src.detection.detector import Detector
from src.state.temporal_smoothing import SceneReading
from src.state.verifier import Verdict, Verifier
from src.utils.config import PipelineConfig

REPO = Path(__file__).resolve().parent.parent

# what verdict each scenario folder should produce
EXPECT = {
    "ok": Verdict.OK,
    "yanlis_sira": Verdict.WRONG_ORDER,
    "yanlis_kombinasyon": Verdict.WRONG_COMBINATION,
    "eksik_tufek": Verdict.MISSING,
    "eksik_tabanca": Verdict.MISSING,
    "eksik_sarjor": Verdict.MISSING,
    "bos": Verdict.EMPTY,
    # occlusion -> should still be OK (all 3 present, just partly hidden)
    "occlusion": Verdict.OK,
    # tekil_* / unlabeled -> singles, not a full scene; skip
}


def images_for(split: str) -> list[Path]:
    if split == "all":
        return sorted((REPO / "data" / "raw").rglob("*.jpg"))
    ds = REPO / "data" / "datasets" / "inventory" / "images" / split
    # dataset stems are "<folder>__<name>"; map back to raw
    out = []
    for p in ds.glob("*.jpg"):
        folder, name = p.stem.split("__", 1)
        raw = REPO / "data" / "raw" / "2026-09-01" / folder / f"{name}.jpg"
        if raw.exists():
            out.append(raw)
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--config", default="configs/pipeline.yaml")
    args = ap.parse_args()

    cfg = PipelineConfig.load(args.config)
    det = Detector(cfg.detection.weights, cfg.detection.conf, cfg.detection.iou,
                   cfg.detection.imgsz, cfg.detection.device)
    ver = Verifier(cfg.expected.sequence)

    total = correct = 0
    okerr_total = okerr_correct = 0        # OK vs "an error was flagged"
    confmat: Counter = Counter()
    misses: list[str] = []
    for img in images_for(args.split):
        folder = img.parent.name
        if folder not in EXPECT:
            continue
        reading = SceneReading.from_detections(det.infer(str(img)))
        got = ver.check(reading).verdict
        want = EXPECT[folder]

        # OK-vs-error: for occlusion a "MISSING" call is acceptable (the object
        # genuinely can't be verified when it's hidden).
        want_ok = want is Verdict.OK
        got_ok = got is Verdict.OK
        okerr_total += 1
        if want_ok == got_ok or (folder == "occlusion" and got is Verdict.MISSING):
            okerr_correct += 1

        # strict verdict-type accuracy skips the occlusion scenario
        if folder != "occlusion":
            total += 1
            confmat[(want.value, got.value)] += 1
            if got == want:
                correct += 1
            else:
                misses.append(f"{folder:20} beklenen={want.value:18} çıkan={got.value:18} "
                              f"({img.name})  gördü={reading.sequence}")

    print(f"\nOK / hata ayrımı ({args.split}): {okerr_correct}/{okerr_total} = "
          f"{okerr_correct / okerr_total:.1%}   (occlusion dahil)")
    print(f"tam verdict tipi   ({args.split}): {correct}/{total} = "
          f"{correct / total:.1%}   (occlusion hariç)\n")
    print("beklenen -> çıkan:")
    for (w, g), n in sorted(confmat.items(), key=lambda x: -x[1]):
        mark = "  " if w == g else "！"
        print(f"  {mark} {w:18} -> {g:18} {n}")
    if misses:
        print(f"\nhatalar ({len(misses)}):")
        for m in misses[:40]:
            print("  " + m)


if __name__ == "__main__":
    main()

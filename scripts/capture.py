"""OAK-D ile eğitim verisi toplama aracı.

Kullanım:
    python scripts/capture.py
    python scripts/capture.py --labels ok,eksik_malzeme,yanlis_sira

Canlı önizleme açılır. Tuşlar:
    SPACE   tek kare kaydet
    c       sürekli çekim aç/kapa (her --interval saniyede bir)
    [  ]    aktif senaryo etiketini değiştir (--labels listesinde gezinir)
    u       son kaydedilen kareyi sil (yanlış çekim)
    q / ESC çık

Kareler tam çözünürlükte şuraya kaydedilir:
    <out>/<YYYY-MM-DD>/<label>/<label>_<zaman>.jpg
"""

from __future__ import annotations

import argparse
import time
from datetime import date
from pathlib import Path

import cv2
import sys

# Add the project root to sys.path so 'src' can be imported when running directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.pipeline.sources import open_source
from src.utils.config import SourceCfg

HUD_H = 30


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--labels", default="ok,eksik_malzeme,yanlis_sira",
                   help="virgülle ayrılmış senaryo etiketleri")
    p.add_argument("--out", default=None,
                   help="kök çıktı klasörü (varsayılan: <repo>/data/raw)")
    p.add_argument("--interval", type=float, default=1.5,
                   help="sürekli çekimde saniye aralığı")
    p.add_argument("--res", default="1280x720",
                   help="çözünürlük WxH (USB2'de 1280x720 akıcı; USB3'te 1920x1080 dene)")
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    w, h = (int(v) for v in args.res.lower().split("x"))
    labels = [s.strip() for s in args.labels.split(",") if s.strip()] or ["unlabeled"]
    li = 0

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.out) if args.out else repo_root / "data" / "raw"
    session = date.today().isoformat()
    root = out_dir / session

    def count(label: str) -> int:
        d = root / label
        return len(list(d.glob("*.jpg"))) if d.exists() else 0

    cfg = SourceCfg(type="oakd", width=w, height=h, fps=args.fps)
    auto = False
    last_auto = 0.0
    last_saved: Path | None = None
    total = 0

    print(f"Oturum: {session}   etiketler: {labels}")
    print("SPACE=kaydet  c=sürekli  t=etiket değiştir  u=geri al  q=çık")

    frames = open_source(cfg)
    try:
        for frame in frames:
            label = labels[li]
            now = time.time()
            do_save = False

            disp = frame
            scale = min(1.0, 1100 / max(frame.shape[0], frame.shape[1]))
            if scale < 1:
                disp = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            disp = disp.copy()
            cv2.rectangle(disp, (0, 0), (disp.shape[1], HUD_H), (0, 0, 0), -1)
            hud = (f"[{li + 1}/{len(labels)}] {label}   bu etiket: {count(label)}"
                   f"   toplam: {total}   {'SUREKLI' if auto else 'manuel'}")
            cv2.putText(disp, hud, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow("OAK-D capture", disp)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == 32:
                do_save = True
            elif key == ord("c"):
                auto = not auto
                last_auto = now
            elif key == ord("t"):
                li = (li + 1) % len(labels)
                print(f"Aktif etiket: {labels[li]}")
            elif key == ord("u"):
                if last_saved and last_saved.exists():
                    last_saved.unlink()
                    total -= 1
                    print(f"silindi: {last_saved}")
                    last_saved = None
                else:
                    print("geri alınacak kare yok")

            if auto and now - last_auto >= args.interval:
                do_save = True
                last_auto = now

            if do_save:
                d = root / label
                d.mkdir(parents=True, exist_ok=True)
                fn = d / f"{label}_{int(now * 1000)}.jpg"
                cv2.imwrite(str(fn), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                last_saved = fn
                total += 1
                print(f"kaydedildi: {fn}  ({frame.shape[1]}x{frame.shape[0]})")
    finally:
        frames.close()
        cv2.destroyAllWindows()

    print(f"\nToplam {total} kare -> {root.resolve()}")
    for lb in labels:
        print(f"  {lb}: {count(lb)}")


if __name__ == "__main__":
    main()

"""Convert an ONNX model to an OAK-D `.blob` (MyriadX) via blobconverter.

    python -m src.edge.export_blob --onnx models/exported/best.onnx --shaves 6

Notes:
- OAK-D expects the model input normalized as the DepthAI pipeline provides it;
  keep preprocessing (0-255 -> 0-1, BGR/RGB) consistent between training and the
  on-device NN node.
- `--shaves` trades latency for leaving compute for other pipeline nodes; 6 is a
  common default on OAK-D.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx", required=True)
    p.add_argument("--shaves", type=int, default=6)
    p.add_argument("--out", default="models/exported")
    args = p.parse_args()

    import blobconverter

    blob_path = blobconverter.from_onnx(
        model=args.onnx,
        data_type="FP16",
        shaves=args.shaves,
        output_dir=args.out,
        use_cache=False,
    )
    print(f"blob: {Path(blob_path).resolve()}")


if __name__ == "__main__":
    main()

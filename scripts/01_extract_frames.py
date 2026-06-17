"""Stage 01: extract frames from the video and keep the sharpest, evenly spaced.

Usage:
  python scripts/01_extract_frames.py \
      --video input/nvs_example_input_video.mp4 \
      --out data/frames --target 250 --min-score 50 --stride 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nvs.sharpness import laplacian_variance, select_sharp_frames


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--out", default="data/frames")
    p.add_argument("--target", type=int, default=250)
    p.add_argument("--min-score", type=float, default=50.0)
    p.add_argument("--stride", type=int, default=5,
                   help="sample every Nth frame before scoring (speed)")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")

    frames, scores = [], []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            scores.append(laplacian_variance(gray))
            frames.append(frame)
        idx += 1
    cap.release()

    keep = select_sharp_frames(scores, target=args.target, min_score=args.min_score)
    if not keep:
        raise SystemExit("No frames passed the sharpness threshold; lower --min-score")

    for out_i, src_i in enumerate(keep):
        cv2.imwrite(str(out / f"frame_{out_i:05d}.png"), frames[src_i])

    print(f"Kept {len(keep)} / {len(frames)} sampled frames -> {out}")


if __name__ == "__main__":
    main()

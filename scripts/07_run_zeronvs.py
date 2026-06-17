"""Stage 07: single-image NVS baseline with ZeroNVS on the same 25 references.

Usage:
  python scripts/07_run_zeronvs.py --colmap colmap --frames data/frames \
      --out outputs/zeronvs --zeronvs third_party/ZeroNVS \
      --n-refs 25 --views-per-ref 4 --device 0
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from nvs.colmap_io import parse_images_txt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--colmap", default="colmap")
    p.add_argument("--frames", default="data/frames")
    p.add_argument("--out", required=True)
    p.add_argument("--zeronvs", required=True, help="path to third_party/ZeroNVS")
    p.add_argument("--n-refs", type=int, default=25)
    p.add_argument("--views-per-ref", type=int, default=4)
    p.add_argument("--arc-span-deg", type=float, default=40.0)
    p.add_argument("--device", default="0")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    poses = sorted(parse_images_txt(Path(args.colmap) / "sparse/0/images.txt"),
                   key=lambda pp: pp.name)
    ref_idx = np.linspace(0, len(poses) - 1, args.n_refs).astype(int)
    yaws = np.linspace(0, args.arc_span_deg, args.views_per_ref)

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.device}
    for r in ref_idx:
        ref_img = Path(args.frames) / poses[r].name
        for v, yaw in enumerate(yaws):
            dst = out / f"ref{r:05d}_view{v}.png"
            # TODO(verify): replace with the pinned ZeroNVS inference CLI.
            cmd = ["python", str(Path(args.zeronvs) / "scripts/run_inference.py"),
                   "--image", str(ref_img),
                   "--azimuth", str(float(yaw)),
                   "--elevation", "10",
                   "--output", str(dst)]
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True, env=env)

    print(f"ZeroNVS done -> {out}")


if __name__ == "__main__":
    main()

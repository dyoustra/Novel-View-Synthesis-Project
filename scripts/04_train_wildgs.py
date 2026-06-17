"""Stage 04 (Method B): train wild-gaussians on the COLMAP dataset (no masks).

Usage:
  python scripts/04_train_wildgs.py --data colmap --out outputs/wildgs --device 0
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="colmap")
    p.add_argument("--out", default="outputs/wildgs")
    p.add_argument("--device", default="0")
    args = p.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    # TODO(verify): confirm subcommand/flags against pinned wild-gaussians README.
    cmd = ["wild-gaussians", "train",
           "--data", args.data,
           "--output", args.out,
           "--backend", "colmap"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.device}
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    print(f"wild-gaussians training complete -> {args.out}")


if __name__ == "__main__":
    main()

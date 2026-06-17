"""Stage 04 (Method C): train gsplat on the masked COLMAP dataset.

Usage:
  python scripts/04_train_gsplat.py --data colmap --out outputs/gsplat \
      --trainer third_party/gsplat/examples/simple_trainer.py --device 0 --max-steps 30000
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nvs.validation import require_nonempty_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="colmap")
    p.add_argument("--out", default="outputs/gsplat")
    p.add_argument("--trainer", required=True,
                   help="path to gsplat examples/simple_trainer.py")
    p.add_argument("--device", default="0")
    p.add_argument("--max-steps", type=int, default=30000)
    p.add_argument("--data-factor", type=int, default=1)
    # test_every: gsplat holds out every Nth image from training for validation;
    # stage 05 must use the same value so it evaluates genuinely held-out frames.
    # TODO(verify): confirm --test_every is accepted by the pinned gsplat simple_trainer.
    p.add_argument("--test-every", type=int, default=8)
    args = p.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, args.trainer, "default",
        "--data_dir", args.data,
        "--data_factor", str(args.data_factor),
        "--result_dir", args.out,
        "--max_steps", str(args.max_steps),
        "--test_every", str(args.test_every),
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.device}
    print("Running:", " ".join(cmd), "with CUDA_VISIBLE_DEVICES=" + args.device)
    subprocess.run(cmd, check=True, env=env)

    require_nonempty_dir(Path(args.out) / "ckpts", "*.pt")
    print(f"gsplat training complete -> {args.out}")


if __name__ == "__main__":
    main()

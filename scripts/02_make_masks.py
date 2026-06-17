"""Stage 02: generate per-frame arm/gripper masks with SAM2 video propagation.

Click points are provided for ONE seed frame; SAM2 propagates FORWARD through
the rest (frames before the seed frame are NOT masked by propagation).
Outputs COLMAP-style mask PNGs to <out>/<frame>.png (0=ignore arm, 255=keep).

Usage:
  python scripts/02_make_masks.py --frames data/frames --out masks \
      --ckpt third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
      --cfg configs/sam2.1/sam2.1_hiera_l.yaml \
      --seed-frame 0 --points 240,360 200,300 --labels 1 1
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch

from nvs.masks import occluder_to_colmap_mask


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frames", default="data/frames")
    p.add_argument("--out", default="masks")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cfg", required=True)
    p.add_argument("--seed-frame", type=int, default=0,
                   help="frame index to place the seed clicks on; SAM2 propagates FORWARD, "
                        "so use 0 (default) or ensure the arm is masked in earlier frames "
                        "another way")
    p.add_argument("--points", nargs="+", required=True,
                   help="x,y positive clicks on the arm in the seed frame")
    p.add_argument("--labels", nargs="+", type=int, required=True,
                   help="1=foreground(arm), 0=background; one per point")
    args = p.parse_args()

    from sam2.build_sam import build_sam2_video_predictor

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame_paths = sorted(Path(args.frames).glob("frame_*.png"))
    if not frame_paths:
        raise SystemExit(f"No frames in {args.frames}")

    pts = np.array([[float(v) for v in pt.split(",")] for pt in args.points],
                   dtype=np.float32)
    lbls = np.array(args.labels, dtype=np.int32)

    predictor = build_sam2_video_predictor(args.cfg, args.ckpt)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(Path(args.frames)))
        predictor.add_new_points_or_box(
            inference_state=state, frame_idx=args.seed_frame, obj_id=1,
            points=pts, labels=lbls,
        )
        for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(state):
            occ = (mask_logits[0] > 0.0).cpu().numpy().squeeze()
            colmap_mask = occluder_to_colmap_mask(occ)
            name = frame_paths[frame_idx].name
            cv2.imwrite(str(out / name), colmap_mask)

    print(f"Wrote {len(frame_paths)} masks -> {out}")


if __name__ == "__main__":
    main()

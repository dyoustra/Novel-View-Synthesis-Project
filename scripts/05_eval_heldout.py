"""Stage 05: render held-out poses and compute PSNR/SSIM/LPIPS.

Held-out frames = every Kth frame, coupled to gsplat's test_every split so that
the evaluated frames were genuinely excluded from training (not training-set fit).
Usage:
  python scripts/05_eval_heldout.py --colmap colmap --frames data/frames \
      --ckpt outputs/gsplat/ckpts/ckpt_29999.pt --out outputs/gsplat/eval --every 8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
from gsplat import rasterization

from nvs.colmap_io import parse_images_txt
from nvs.metrics import psnr, ssim, lpips_fn
from nvs.trajectory import qvec_to_rotmat

def main() -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "render_orbits", str(Path(__file__).with_name("06_render_orbits.py")))
    render_orbits = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(render_orbits)

    p = argparse.ArgumentParser()
    p.add_argument("--colmap", default="colmap")
    p.add_argument("--frames", default="data/frames")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--every", type=int, default=8,
                   help="evaluate every Nth frame; MUST match gsplat training --test-every "
                        "so the evaluated frames were held out of training")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    poses = sorted(parse_images_txt(Path(args.colmap) / "sparse/0/images.txt"),
                   key=lambda pp: pp.name)
    heldout = poses[:: args.every]

    cam_line = [ln for ln in (Path(args.colmap) / "sparse/0/cameras.txt").read_text()
                .splitlines() if ln.strip() and not ln.startswith("#")][0].split()
    fx, fy, cx, cy = (float(cam_line[4]), float(cam_line[5]),
                      float(cam_line[6]), float(cam_line[7]))
    w, h = int(cam_line[2]), int(cam_line[3])
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                     dtype=torch.float32, device=args.device)

    means, quats, scales, opac, colors, sh_degree = render_orbits.load_splats(
        args.ckpt, args.device)
    score_lpips = lpips_fn(args.device)

    rows = []
    for pp in heldout:
        gt_path = Path(args.frames) / pp.name
        if not gt_path.exists():
            continue
        gt = cv2.cvtColor(cv2.imread(str(gt_path)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        viewmat = torch.eye(4, dtype=torch.float32, device=args.device)
        viewmat[:3, :3] = torch.from_numpy(qvec_to_rotmat(pp.qvec)).float()
        viewmat[:3, 3] = torch.tensor(pp.tvec, dtype=torch.float32)
        renders, _a, _m = rasterization(
            means, quats, scales, opac, colors,
            viewmat.unsqueeze(0), K.unsqueeze(0), width=w, height=h, sh_degree=sh_degree)
        pred = renders[0].clamp(0, 1).cpu().numpy()
        rows.append({"name": pp.name, "psnr": psnr(gt, pred),
                     "ssim": ssim(gt, pred), "lpips": score_lpips(gt, pred)})

    if not rows:
        raise SystemExit(
            "No held-out frames matched between COLMAP poses and --frames; "
            "check --frames path and that filenames match."
        )
    summary = {k: float(np.mean([r[k] for r in rows])) for k in ("psnr", "ssim", "lpips")}
    (out / "metrics.json").write_text(json.dumps(
        {"per_image": rows, "mean": summary}, indent=2))
    print("Mean metrics:", summary)


if __name__ == "__main__":
    main()

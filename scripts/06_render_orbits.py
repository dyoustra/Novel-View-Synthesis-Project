"""Stage 06: render 4 novel views per reference frame from a gsplat checkpoint.

Usage:
  python scripts/06_render_orbits.py --colmap colmap --ckpt outputs/gsplat/ckpts/ckpt_29999.pt \
      --out outputs/gsplat/novel --n-refs 25 --views-per-ref 4
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
from gsplat import rasterization

from nvs.colmap_io import parse_images_txt
from nvs.trajectory import camera_center, scene_stats, arc_poses_around
from nvs.validation import require_nonempty_dir


def load_splats(ckpt_path: str, device: str):
    # weights_only=True: gsplat ckpts are plain tensor dicts; avoids arbitrary
    # code execution from unpickling an untrusted checkpoint.
    ck = torch.load(ckpt_path, map_location=device, weights_only=True)
    s = ck["splats"]
    means = s["means"].to(device)
    quats = s["quats"].to(device)
    scales = torch.exp(s["scales"]).to(device)
    opac = torch.sigmoid(s["opacities"]).to(device)
    colors = torch.cat([s["sh0"], s["shN"]], dim=1).to(device)  # SH coeffs
    sh_degree = int(np.sqrt(colors.shape[1]) - 1)
    return means, quats, scales, opac, colors, sh_degree


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--colmap", default="colmap")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-refs", type=int, default=25)
    p.add_argument("--views-per-ref", type=int, default=4)
    p.add_argument("--arc-span-deg", type=float, default=40.0)
    p.add_argument("--elevation", type=float, default=0.3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=480)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    poses = parse_images_txt(Path(args.colmap) / "sparse/0/images.txt")
    poses = sorted(poses, key=lambda pp: pp.name)
    centers = np.array([camera_center(pp.qvec, pp.tvec) for pp in poses])
    centroid, radius, _ = scene_stats(centers)

    ref_idx = np.linspace(0, len(poses) - 1, args.n_refs).astype(int)
    means, quats, scales, opac, colors, sh_degree = load_splats(args.ckpt, args.device)

    cam_line = [ln for ln in (Path(args.colmap) / "sparse/0/cameras.txt").read_text()
                .splitlines() if ln.strip() and not ln.startswith("#")][0].split()
    fx, fy, cx, cy = (float(cam_line[4]), float(cam_line[5]),
                      float(cam_line[6]), float(cam_line[7]))
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                     dtype=torch.float32, device=args.device)

    for r in ref_idx:
        base_center = centers[r]
        start_angle = float(np.arctan2(base_center[2] - centroid[2],
                                       base_center[0] - centroid[0]))
        arc = arc_poses_around(
            centroid=centroid, radius=radius, base_height=centroid[1],
            start_angle=start_angle, n=args.views_per_ref,
            arc_span=np.deg2rad(args.arc_span_deg), elevation=args.elevation,
        )
        for v, (R, t) in enumerate(arc):
            viewmat = torch.eye(4, dtype=torch.float32, device=args.device)
            viewmat[:3, :3] = torch.from_numpy(R).float()
            viewmat[:3, 3] = torch.from_numpy(t).float()
            renders, _alphas, _meta = rasterization(
                means, quats, scales, opac, colors,
                viewmat.unsqueeze(0), K.unsqueeze(0),
                width=args.width, height=args.height,
                sh_degree=sh_degree,
            )
            img = (renders[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(str(out / f"ref{r:05d}_view{v}.png"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    require_nonempty_dir(out, "*.png")
    print(f"Rendered {args.n_refs * args.views_per_ref} novel views -> {out}")


if __name__ == "__main__":
    main()

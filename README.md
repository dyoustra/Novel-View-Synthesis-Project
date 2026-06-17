# Novel View Synthesis — Robot Manipulation Scene

Reproducible pipeline producing ≥100 novel views (25 refs × 4) via gsplat+masks,
wild-gaussians, and ZeroNVS, plus held-out PSNR/SSIM/LPIPS.

See `docs/superpowers/specs/2026-06-14-novel-view-synthesis-design.md` for the
design and `docs/superpowers/plans/2026-06-14-novel-view-synthesis.md` for the
implementation plan.

## Install
1. Install COLMAP (`brew install colmap` / apt / conda) and a CUDA-enabled PyTorch.
2. `pip install -r requirements.txt`
3. `./setup.sh`  # clones SAM2, wild-gaussians, ZeroNVS at pinned SHAs + weights

## Run (single GPU, sequential)
```bash
# 01 — extract sharp, evenly-spaced frames
python scripts/01_extract_frames.py --video input/nvs_example_input_video.mp4 --out data/frames

# 02 — SAM2 arm masks (seed a click on the arm in frame 0)
python scripts/02_make_masks.py --frames data/frames --out masks \
    --ckpt third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
    --cfg configs/sam2.1/sam2.1_hiera_l.yaml --points 240,360 --labels 1

# 03 — COLMAP SfM (poses + sparse cloud), uses masks as feature masks
scripts/03_run_colmap.sh data/frames masks colmap

# 04 — train both methods (sequential on one GPU)
python scripts/04_train_gsplat.py --data colmap --out outputs/gsplat \
    --trainer third_party/gsplat/examples/simple_trainer.py --device 0
python scripts/04_train_wildgs.py --data colmap --out outputs/wildgs --device 0

# 05 — held-out metrics (PSNR/SSIM/LPIPS)
python scripts/05_eval_heldout.py --colmap colmap --frames data/frames \
    --ckpt outputs/gsplat/ckpts/ckpt_29999.pt --out outputs/gsplat/eval

# 06 — render 25×4 novel views (orbit arcs)
python scripts/06_render_orbits.py --colmap colmap \
    --ckpt outputs/gsplat/ckpts/ckpt_29999.pt --out outputs/gsplat/novel

# 07 — ZeroNVS single-image baseline on the same 25 references
python scripts/07_run_zeronvs.py --colmap colmap --frames data/frames \
    --out outputs/zeronvs --zeronvs third_party/ZeroNVS
```

## Tests
`python -m pytest`   # pure-logic unit tests (no GPU required)

## Notes
- Default training is single GPU, sequential (Method C then Method B). Pass a
  different `--device` to each to use two GPUs.
- `04_train_wildgs.py` and `07_run_zeronvs.py` contain `TODO(verify)` markers for
  CLIs that must be confirmed against the pinned third-party repos on first GPU run.
- The scene is mostly static with the robot arm as a transient foreground occluder;
  Method C masks it explicitly, Method B (wild-gaussians) models it as transient.

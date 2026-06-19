# Novel View Synthesis — Robot Manipulation Scene

Reproducible pipeline producing ≥100 novel views (25 refs × 4) via gsplat+masks,
wild-gaussians, and ZeroNVS, plus held-out PSNR/SSIM/LPIPS.

See `docs/superpowers/specs/2026-06-14-novel-view-synthesis-design.md` for the
design and `docs/superpowers/plans/2026-06-14-novel-view-synthesis.md` for the
implementation plan.

## Install (Linux + NVIDIA GPU)
*Installed and run on Fedora distribution.*

Put the CUDA build toolchain in a conda env. gsplat compiles CUDA
kernels at install time and needs `nvcc`; a conda env pins the CUDA toolkit
independent of the system and sidesteps the common "distro GCC too new for nvcc"
build failure.

1. **NVIDIA driver** (system-level, once): e.g. Fedora `sudo dnf install akmod-nvidia`
   (via RPM Fusion), then reboot. Confirm with `nvidia-smi`; note its top-right
   "CUDA Version" — that's the max CUDA your driver supports.
2. **Conda env with CUDA toolkit + matching PyTorch:**
   ```bash
   conda create -n nvs python=3.10 && conda activate nvs
   conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit   # provides nvcc
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   ```
   Pick a `cuXXX` torch wheel at or below your driver's CUDA version (step 1).
3. **COLMAP:** easiest inside the env via `conda install -c conda-forge colmap`
   (request a `*cuda*` build for GPU SIFT, optional). Fedora's `dnf install colmap`
   also works but is typically CPU-only — fine for this dataset (250 frames @ 480p).
4. **`./setup.sh`** (run inside the activated `nvs` env) — clones SAM2,
   wild-gaussians, ZeroNVS at pinned SHAs + weights, and `pip install`s gsplat
   (compiled against the conda CUDA toolkit).

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
- The code has 5 `TODO(verify)` markers (third-party CLIs + SHA pins) to resolve on
  the GPU box. See `docs/gpu-verification-checklist.md` for the ordered runbook;
  list them anytime with `grep -rn "TODO(verify)" setup.sh scripts/`.
- The scene is mostly static with the robot arm as a transient foreground occluder;
  Method C masks it explicitly, Method B (wild-gaussians) models it as transient.

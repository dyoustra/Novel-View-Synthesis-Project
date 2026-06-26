# Novel View Synthesis — Robot Manipulation Scene

Reproducible pipeline producing ≥100 novel views (25 refs × 4) via gsplat+masks,
wild-gaussians, and ZeroNVS, plus held-out PSNR/SSIM/LPIPS.

See `docs/superpowers/specs/2026-06-14-novel-view-synthesis-design.md` for the
design and `docs/superpowers/plans/2026-06-14-novel-view-synthesis.md` for the
implementation plan.

## Install (Linux + NVIDIA GPU)
*Installed and run on Fedora distribution.*

Put the CUDA build toolchain in a conda env. The 3DGS extensions compile CUDA
kernels at install time, so **four versions must agree**: the NVIDIA driver
(the ceiling), the PyTorch wheel's bundled CUDA, the conda `nvcc`, and the host
GCC (old enough for nvcc to parse — GCC 15 is too new). The steps below pin all
four to a known-good set: driver ≥ 13.0, torch `cu130`, nvcc 13.0, GCC 13.

1. **NVIDIA driver** (system-level, once): e.g. Fedora `sudo dnf install akmod-nvidia`
   (via RPM Fusion), then reboot. Confirm with `nvidia-smi`; note its top-right
   "CUDA Version" — the max CUDA your driver supports (must be ≥ 13.0 for the
   versions below; if lower, substitute a matching `cuXXX` / `cuda-XX.X` set).
2. **Conda env with the CUDA 13.0 toolchain + matching PyTorch:**
   ```bash
   conda create -n nvs python=3.10 && conda activate nvs
   # nvcc + CUDA headers — install the components BY NAME: the `cuda-toolkit`
   # metapackage name collides with a pip package torch pulls in, so a plain
   # `conda install cuda-toolkit` silently no-ops.
   conda install -c "nvidia/label/cuda-13.0.0" "cuda-nvcc=13.0" "cuda-cudart-dev=13.0"
   # Host compiler nvcc can parse (GCC 15 fails with `__decay` undefined).
   conda install -c conda-forge "gcc_linux-64=13" "gxx_linux-64=13"
   pip install torch --index-url https://download.pytorch.org/whl/cu130
   pip install -r requirements.txt
   ```
3. **COLMAP:** easiest inside the env via `conda install -c conda-forge colmap`
   (request a `*cuda*` build for GPU SIFT, optional). Fedora's `dnf install colmap`
   also works but is typically CPU-only — fine for this dataset (~250 frames @ 480p).
4. **`./setup.sh`** (run inside the activated `nvs` env) — clones SAM2 and ZeroNVS
   at pinned SHAs + weights; `pip install`s the gsplat wheel; and clones the gsplat
   repo at the matching tag for its `examples/simple_trainer.py` (the wheel ships
   the library but not the trainer), installing the example deps with
   `--no-build-isolation` (fused-ssim / fused-bilagrid import torch at build time).
5. **NerfBaselines env for Method B (wild-gaussians):** wild-gaussians needs
   CUDA 11.8 / Python 3.11 — incompatible with the `nvs` CUDA-13 env — so it runs
   in its own NerfBaselines-managed environment. Create a small orchestrator env:
   ```bash
   conda create -n nb python=3.11 -y && conda activate nb
   pip install "nerfbaselines>=1.2.0"
   ```
   Stage 04's wild-gaussians step runs from this `nb` env; `--backend conda` makes
   NerfBaselines auto-build the isolated CUDA-11.8 method env on first run.

## Run (single GPU, sequential)
**Input:** the source video is not tracked in git (it's large and `*.mp4` is
gitignored). Place it at `input/nvs_example_input_video.mp4` before running —
copy it from wherever you were given the dataset (e.g. `rsync`/`scp` it to the box).

```bash
# 01 — extract sharp, evenly-spaced frames
python scripts/01_extract_frames.py --video input/nvs_example_input_video.mp4 --out data/frames

# 02 — SAM2 arm masks (seed on frame 85, where the arm is largest/clearest;
# propagation is bidirectional, so one seed covers all frames)
python scripts/02_make_masks.py --frames data/frames --out masks \
    --ckpt third_party/sam2/checkpoints/sam2_hiera_large.pt \
    --cfg sam2_hiera_l.yaml --seed-frame 85 \
    --points 12,270 50,230 100,160 171,153 195,196 212,191 222,182 229,236 263,330 324,281 --labels 1 1 1 1 1 1 1 1 1 1

# 03 — COLMAP SfM (poses + sparse cloud), uses masks as feature masks
scripts/03_run_colmap.sh data/frames masks colmap

# 04 — train both methods (sequential on one GPU)
python scripts/04_train_gsplat.py --data colmap --out outputs/gsplat \
    --trainer third_party/gsplat/examples/simple_trainer.py --device 0
# wild-gaussians runs from the `nb` env (conda activate nb); NerfBaselines builds
# its own CUDA-11.8 backend, so this step does NOT use the nvs env:
python scripts/04_train_wildgs.py --data colmap --out outputs/wildgs --device 0

# 05 — held-out metrics (PSNR/SSIM/LPIPS)
python scripts/05_eval_heldout.py --colmap colmap --frames data/frames \
    --ckpt outputs/gsplat/ckpts/ckpt_29999.pt --out outputs/gsplat/eval

# 06 — render 25×4 novel views (orbit arcs)
python scripts/06_render_orbits.py --colmap colmap \
    --ckpt outputs/gsplat/ckpts/ckpt_29999.pt --out outputs/gsplat/novel

# 07 — ZeroNVS single-image baseline: DEFERRED (future work, not run for this
# submission; the >=100-view deliverable is met by stages 04/06 above). See
# docs/gpu-verification-checklist.md #3. Kept for reference only:
# python scripts/07_run_zeronvs.py --colmap colmap --frames data/frames \
#     --out outputs/zeronvs --zeronvs third_party/ZeroNVS
```

## Tests
`python -m pytest`   # pure-logic unit tests (no GPU required)

## Notes
- Default training is single GPU, sequential (Method C then Method B). Pass a
  different `--device` to each to use two GPUs.
- Remaining `TODO(verify)` markers (third-party CLI/version checks) are tracked with
  per-item status in `docs/gpu-verification-checklist.md`; list them anytime with
  `grep -rn "TODO(verify)" setup.sh scripts/`. ZeroNVS (stage 07) is deferred to
  future work — see that doc's item #3.
- The scene is mostly static with the robot arm as a transient foreground occluder;
  Method C masks it explicitly, Method B (wild-gaussians) models it as transient.

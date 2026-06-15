# Novel View Synthesis Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible pipeline that takes the robot manipulation video and produces ≥100 novel views (25 references × 4 views) via three methods — gsplat+masks, wild-gaussians, and ZeroNVS — plus held-out PSNR/SSIM/LPIPS metrics.

**Architecture:** A small pure-Python package `nvs/` holds all testable logic (sharpness scoring, COLMAP I/O, trajectory math, metrics, mask conversion, validation gates). Thin CLI scripts in `scripts/` (numbered 01–07) call `nvs/` functions and invoke external tools (COLMAP binary, gsplat, wild-gaussians, SAM2, ZeroNVS). Each stage reads the previous stage's on-disk output and writes its own, so any stage re-runs in isolation.

**Tech Stack:** Python 3.10+, pytest, numpy, opencv-python, COLMAP (binary), gsplat (pip), wild-gaussians + SAM2 + ZeroNVS (cloned by `setup.sh` at pinned SHAs), torch/CUDA, lpips, scikit-image.

**Reference:** Design spec at `docs/superpowers/specs/2026-06-14-novel-view-synthesis-design.md`.

**Environment note:** Pure-function tasks (2-pure, 3-pure, 4-pure, 7-pure, 8-pure) run on any machine including the Mac (no GPU). Tasks invoking external tools (setup, COLMAP, training, rendering, ZeroNVS) require a CUDA GPU; run those on the local CUDA box or remote server. Default training is **single GPU, sequential** (Method C then Method B).

---

## File Structure

**Create:**
- `pyproject.toml` — package + pytest config
- `requirements.txt` — pinned Python deps
- `nvs/__init__.py`
- `nvs/sharpness.py` — variance-of-Laplacian scoring + frame selection
- `nvs/colmap_io.py` — parse COLMAP output, registration rate, read poses/intrinsics
- `nvs/trajectory.py` — scene centroid, orbit/arc pose generation, quaternion helpers
- `nvs/metrics.py` — PSNR/SSIM/LPIPS
- `nvs/masks.py` — SAM2 mask → COLMAP feature-mask + gsplat alpha format conversion
- `nvs/validation.py` — stage validation gates (fail-loud assertions)
- `scripts/01_extract_frames.py`
- `scripts/02_make_masks.py`
- `scripts/03_run_colmap.sh`
- `scripts/04_train_gsplat.py`
- `scripts/04_train_wildgs.py`
- `scripts/05_eval_heldout.py`
- `scripts/06_render_orbits.py`
- `scripts/07_run_zeronvs.py`
- `setup.sh`
- `tests/test_sharpness.py`, `tests/test_colmap_io.py`, `tests/test_trajectory.py`, `tests/test_metrics.py`, `tests/test_masks.py`, `tests/test_validation.py`
- `tests/conftest.py`
- `README.md`

**Modify:**
- `.gitignore` — add `data/frames/`, `masks/`, `colmap/`, `outputs/`, `third_party/`, `__pycache__/`, `.pytest_cache/`, `*.egg-info/`

**Existing (do not move):** `input/nvs_example_input_video.mp4` is the source video (default `--video` arg). `example output/` and `Task_Novel_View_Synthesis.md` stay as-is.

---

## Task 0: Project scaffold

**Files:**
- Create: `pyproject.toml`, `requirements.txt`, `nvs/__init__.py`, `tests/conftest.py`
- Modify: `.gitignore`

- [ ] **Step 1: Create the package + pytest config**

`pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "nvs"
version = "0.1.0"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
include = ["nvs*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v"
```

- [ ] **Step 2: Create requirements.txt**

```
numpy>=1.24
opencv-python>=4.8
scikit-image>=0.21
lpips>=0.1.4
pillow>=10.0
tqdm>=4.65
```
(torch + gsplat are installed per-machine in `setup.sh` because their build depends on the CUDA version.)

- [ ] **Step 3: Create empty package + conftest**

`nvs/__init__.py`:
```python
"""Novel view synthesis pipeline — shared pure logic."""
```

`tests/conftest.py`:
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
```

- [ ] **Step 4: Update .gitignore**

Append these lines to `.gitignore`:
```
data/frames/
masks/
colmap/
outputs/
third_party/
__pycache__/
.pytest_cache/
*.egg-info/
```

- [ ] **Step 5: Verify pytest collects nothing yet (clean baseline)**

Run: `python -m pytest`
Expected: `no tests ran` (exit code 5) — confirms config is valid.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml requirements.txt nvs/__init__.py tests/conftest.py .gitignore
git commit -m "chore: project scaffold for NVS pipeline"
```

---

## Task 1: setup.sh — third-party acquisition

**Files:**
- Create: `setup.sh`

This script is orchestration (clone + download), not unit-testable. It is validated by running it on a CUDA machine.

- [ ] **Step 1: Write setup.sh**

```bash
#!/usr/bin/env bash
# Clone third-party NVS tools at pinned commits and fetch model weights.
# Run on a CUDA machine. COLMAP must be installed separately (brew/apt/conda).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TP="$ROOT/third_party"
mkdir -p "$TP"

# --- pinned commit SHAs (update here to bump versions) ---
SAM2_SHA="7e1596c0b6462eb1d1ba7e1492430fed95023598"
WILDGS_SHA="main"     # TODO(verify): pin to a tested commit after first successful run
ZERONVS_SHA="main"    # TODO(verify): pin to a tested commit after first successful run

clone_at() {  # url dir sha
  local url="$1" dir="$2" sha="$3"
  if [ ! -d "$TP/$dir/.git" ]; then
    git clone "$url" "$TP/$dir"
  fi
  git -C "$TP/$dir" fetch --all
  git -C "$TP/$dir" checkout "$sha"
}

# --- gsplat: pip (compiles CUDA kernels against local torch) ---
pip install gsplat

# --- SAM2 ---
clone_at "https://github.com/facebookresearch/sam2.git" "sam2" "$SAM2_SHA"
pip install -e "$TP/sam2"
mkdir -p "$TP/sam2/checkpoints"
( cd "$TP/sam2/checkpoints" && bash download_ckpts.sh ) || \
  echo "WARN: SAM2 checkpoint download failed — fetch sam2.1_hiera_large.pt manually"

# --- wild-gaussians ---
clone_at "https://github.com/jkulhanek/wild-gaussians.git" "wild-gaussians" "$WILDGS_SHA"
pip install -e "$TP/wild-gaussians"

# --- ZeroNVS (stretch baseline) ---
clone_at "https://github.com/kylesargent/ZeroNVS.git" "ZeroNVS" "$ZERONVS_SHA"
echo "ZeroNVS cloned. Follow third_party/ZeroNVS/README for its conda env + weights."

echo "setup.sh complete."
```

- [ ] **Step 2: Make executable**

Run: `chmod +x setup.sh`

- [ ] **Step 3: Commit (do NOT run here — runs on the CUDA machine)**

```bash
git add setup.sh
git commit -m "feat: setup.sh to vendor third-party NVS tools at pinned SHAs"
```

> **Execution note for the GPU machine:** after first successful runs of wild-gaussians and ZeroNVS, replace `WILDGS_SHA`/`ZERONVS_SHA` `main` with the exact tested SHAs (`git -C third_party/<dir> rev-parse HEAD`) and re-commit.

---

## Task 2 (pure): Frame sharpness scoring & selection

**Files:**
- Create: `nvs/sharpness.py`
- Test: `tests/test_sharpness.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_sharpness.py`:
```python
import numpy as np
from nvs.sharpness import laplacian_variance, select_sharp_frames


def test_laplacian_variance_higher_for_sharper():
    flat = np.zeros((64, 64), dtype=np.uint8)
    sharp = np.zeros((64, 64), dtype=np.uint8)
    sharp[::2, :] = 255  # high-frequency stripes
    assert laplacian_variance(sharp) > laplacian_variance(flat)


def test_select_sharp_frames_picks_target_count_evenly():
    # 100 frames, all equally sharp -> even temporal sampling of 10
    scores = [10.0] * 100
    idx = select_sharp_frames(scores, target=10, min_score=1.0)
    assert len(idx) == 10
    assert idx[0] == 0
    assert idx == sorted(idx)
    assert len(set(idx)) == 10


def test_select_sharp_frames_drops_below_min_score():
    scores = [0.1] * 50 + [50.0] * 50  # first half blurry
    idx = select_sharp_frames(scores, target=10, min_score=1.0)
    assert all(i >= 50 for i in idx)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_sharpness.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'nvs.sharpness'`

- [ ] **Step 3: Implement nvs/sharpness.py**

```python
"""Frame sharpness scoring (variance of Laplacian) and even-coverage selection."""
from __future__ import annotations

import cv2
import numpy as np


def laplacian_variance(gray: np.ndarray) -> float:
    """Focus measure: variance of the Laplacian. Higher = sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def select_sharp_frames(
    scores: list[float], target: int, min_score: float
) -> list[int]:
    """Pick ~`target` frame indices with even temporal coverage, dropping any
    frame whose sharpness is below `min_score`. Within each evenly-spaced bin,
    keep the sharpest surviving frame."""
    eligible = [i for i, s in enumerate(scores) if s >= min_score]
    if not eligible:
        return []
    if len(eligible) <= target:
        return eligible
    bins = np.array_split(np.array(eligible), target)
    chosen = []
    for b in bins:
        if len(b) == 0:
            continue
        best = max(b, key=lambda i: scores[i])
        chosen.append(int(best))
    return sorted(chosen)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_sharpness.py`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add nvs/sharpness.py tests/test_sharpness.py
git commit -m "feat: frame sharpness scoring and even-coverage selection"
```

---

## Task 2 (script): Frame extraction CLI

**Files:**
- Create: `scripts/01_extract_frames.py`

- [ ] **Step 1: Write the extraction script**

```python
"""Stage 01: extract frames from the video and keep the sharpest, evenly spaced.

Usage:
  python scripts/01_extract_frames.py \
      --video input/nvs_example_input_video.mp4 \
      --out data/frames --target 250 --min-score 50 --stride 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

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
```

- [ ] **Step 2: Run it on the real video (smoke)**

Run: `python scripts/01_extract_frames.py --video input/nvs_example_input_video.mp4 --out data/frames --target 250 --stride 5`
Expected: prints `Kept <N> / <M> sampled frames -> data/frames`, with 150–250 PNGs in `data/frames/`. Spot-check a few open cleanly and are not blurry.

- [ ] **Step 3: Commit**

```bash
git add scripts/01_extract_frames.py
git commit -m "feat: stage 01 frame extraction with blur filtering"
```

---

## Task 3 (pure): Mask format conversion

**Files:**
- Create: `nvs/masks.py`
- Test: `tests/test_masks.py`

Context: SAM2 produces a boolean mask where **True = robot arm (occluder, to ignore)**. COLMAP feature masks use the opposite convention: a PNG where **0 = ignore, 255 = use**, saved as `<image_name>.png` next to the image (per COLMAP `--ImageReader.mask_path`). gsplat loss masks want **1.0 = keep, 0.0 = ignore** float arrays. These converters bridge the conventions.

- [ ] **Step 1: Write the failing tests**

`tests/test_masks.py`:
```python
import numpy as np
from nvs.masks import occluder_to_colmap_mask, occluder_to_keep_alpha


def test_occluder_to_colmap_mask_inverts():
    occ = np.array([[True, False], [False, True]])
    out = occluder_to_colmap_mask(occ)
    assert out.dtype == np.uint8
    assert out[0, 0] == 0 and out[1, 1] == 0      # arm -> ignore
    assert out[0, 1] == 255 and out[1, 0] == 255  # scene -> use


def test_occluder_to_keep_alpha_is_float_complement():
    occ = np.array([[True, False]])
    alpha = occluder_to_keep_alpha(occ)
    assert alpha.dtype == np.float32
    assert alpha[0, 0] == 0.0 and alpha[0, 1] == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_masks.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'nvs.masks'`

- [ ] **Step 3: Implement nvs/masks.py**

```python
"""Convert SAM2 occluder masks (True = arm) to downstream mask conventions."""
from __future__ import annotations

import numpy as np


def occluder_to_colmap_mask(occluder: np.ndarray) -> np.ndarray:
    """COLMAP mask PNG: 0 = ignore pixel, 255 = use pixel."""
    keep = ~occluder.astype(bool)
    return (keep.astype(np.uint8) * 255)


def occluder_to_keep_alpha(occluder: np.ndarray) -> np.ndarray:
    """gsplat loss mask: 1.0 = keep (supervise), 0.0 = ignore."""
    keep = ~occluder.astype(bool)
    return keep.astype(np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_masks.py`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add nvs/masks.py tests/test_masks.py
git commit -m "feat: occluder mask format conversion"
```

---

## Task 3 (script): SAM2 mask generation CLI

**Files:**
- Create: `scripts/02_make_masks.py`

> **Verify against the pinned SAM2 version** (`third_party/sam2`): the import path and predictor API below match SAM2.1 (`sam2.build_sam.build_sam2_video_predictor`). If the pinned SHA differs, adjust the constructor/method names per its README.

- [ ] **Step 1: Write the mask script**

```python
"""Stage 02: generate per-frame arm/gripper masks with SAM2 video propagation.

Click points are provided for ONE seed frame; SAM2 propagates through the rest.
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
    p.add_argument("--seed-frame", type=int, default=0)
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
```

- [ ] **Step 2: Run on the GPU machine (interactive seed tuning)**

Run the command in the docstring; open a few `masks/frame_*.png` and confirm the arm is black (0) and the scene white (255). Re-seed points if the arm is missed. Fallback if SAM2 won't install: a `--threshold` color mask of the dark arm (document in README).

- [ ] **Step 3: Commit**

```bash
git add scripts/02_make_masks.py
git commit -m "feat: stage 02 SAM2 arm mask generation"
```

---

## Task 4 (pure): COLMAP output parsing & registration rate

**Files:**
- Create: `nvs/colmap_io.py`
- Test: `tests/test_colmap_io.py`

COLMAP writes text models as `cameras.txt`, `images.txt`, `points3D.txt`. We need: number of registered images, and per-image camera poses (quaternion + translation). The `images.txt` format: two lines per image, the first is `IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME`.

- [ ] **Step 1: Write the failing tests**

`tests/test_colmap_io.py`:
```python
from pathlib import Path
from nvs.colmap_io import parse_images_txt, registration_rate

SAMPLE = """# Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
1 1.0 0.0 0.0 0.0 0.5 0.0 0.0 1 frame_00000.png
10.0 20.0 1.0
2 0.707 0.0 0.707 0.0 1.0 2.0 3.0 1 frame_00001.png
15.0 25.0 2.0
"""


def test_parse_images_txt(tmp_path: Path):
    f = tmp_path / "images.txt"
    f.write_text(SAMPLE)
    poses = parse_images_txt(f)
    assert len(poses) == 2
    assert poses[0].name == "frame_00000.png"
    assert poses[0].qvec == (1.0, 0.0, 0.0, 0.0)
    assert poses[1].tvec == (1.0, 2.0, 3.0)


def test_registration_rate(tmp_path: Path):
    f = tmp_path / "images.txt"
    f.write_text(SAMPLE)
    assert registration_rate(f, total_input_frames=4) == 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_colmap_io.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'nvs.colmap_io'`

- [ ] **Step 3: Implement nvs/colmap_io.py**

```python
"""Read COLMAP text reconstruction output (poses, registration rate)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImagePose:
    image_id: int
    qvec: tuple[float, float, float, float]  # world-to-camera (QW,QX,QY,QZ)
    tvec: tuple[float, float, float]
    camera_id: int
    name: str


def parse_images_txt(path: Path) -> list[ImagePose]:
    """Parse COLMAP images.txt. Pose lines alternate with 2D-point lines."""
    poses: list[ImagePose] = []
    lines = [ln for ln in Path(path).read_text().splitlines()
             if ln.strip() and not ln.startswith("#")]
    # Every other line (starting at 0) is a pose header; the line after is points.
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        poses.append(ImagePose(
            image_id=int(parts[0]),
            qvec=(float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])),
            tvec=(float(parts[5]), float(parts[6]), float(parts[7])),
            camera_id=int(parts[8]),
            name=parts[9],
        ))
    return poses


def registration_rate(images_txt: Path, total_input_frames: int) -> float:
    """Fraction of input frames COLMAP successfully registered."""
    if total_input_frames <= 0:
        return 0.0
    return len(parse_images_txt(images_txt)) / total_input_frames
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_colmap_io.py`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add nvs/colmap_io.py tests/test_colmap_io.py
git commit -m "feat: COLMAP images.txt parsing and registration rate"
```

---

## Task 4 (validation): COLMAP registration gate

**Files:**
- Create: `nvs/validation.py`
- Test: `tests/test_validation.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_validation.py`:
```python
import pytest
from nvs.validation import require_registration_rate, require_nonempty_dir


def test_require_registration_rate_passes():
    require_registration_rate(0.9, threshold=0.8)  # no raise


def test_require_registration_rate_fails():
    with pytest.raises(RuntimeError, match="registration rate"):
        require_registration_rate(0.3, threshold=0.8)


def test_require_nonempty_dir(tmp_path):
    with pytest.raises(RuntimeError, match="no files"):
        require_nonempty_dir(tmp_path, "*.png")
    (tmp_path / "a.png").write_bytes(b"x")
    require_nonempty_dir(tmp_path, "*.png")  # no raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_validation.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'nvs.validation'`

- [ ] **Step 3: Implement nvs/validation.py**

```python
"""Fail-loud validation gates run at stage boundaries."""
from __future__ import annotations

from pathlib import Path


def require_registration_rate(rate: float, threshold: float = 0.8) -> None:
    if rate < threshold:
        raise RuntimeError(
            f"COLMAP registration rate {rate:.2f} < {threshold:.2f}. "
            "Try denser frame sampling, exhaustive matching, or check masks."
        )


def require_nonempty_dir(path: Path, pattern: str) -> None:
    if not list(Path(path).glob(pattern)):
        raise RuntimeError(f"{path} has no files matching {pattern!r}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_validation.py`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add nvs/validation.py tests/test_validation.py
git commit -m "feat: stage validation gates"
```

---

## Task 4 (script): COLMAP SfM CLI

**Files:**
- Create: `scripts/03_run_colmap.sh`

- [ ] **Step 1: Write the COLMAP script**

```bash
#!/usr/bin/env bash
# Stage 03: COLMAP SfM (sequential matching) producing poses + sparse cloud
# in the layout gsplat expects: <workdir>/{images, sparse/0}.
#
# Usage: scripts/03_run_colmap.sh data/frames masks colmap
set -euo pipefail

FRAMES="${1:-data/frames}"
MASKS="${2:-masks}"
WORK="${3:-colmap}"

mkdir -p "$WORK"
DB="$WORK/database.db"

MASK_ARG=()
if [ -d "$MASKS" ] && [ -n "$(ls -A "$MASKS" 2>/dev/null)" ]; then
  MASK_ARG=(--ImageReader.mask_path "$MASKS")
fi

colmap feature_extractor \
  --database_path "$DB" \
  --image_path "$FRAMES" \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model OPENCV \
  "${MASK_ARG[@]}"

colmap sequential_matcher --database_path "$DB"

mkdir -p "$WORK/sparse"
colmap mapper \
  --database_path "$DB" \
  --image_path "$FRAMES" \
  --output_path "$WORK/sparse"

# gsplat wants images alongside sparse/0; symlink the frame dir.
ln -sfn "$(cd "$FRAMES" && pwd)" "$WORK/images"

# Convert model 0 to TXT for parsing/validation.
colmap model_converter \
  --input_path "$WORK/sparse/0" \
  --output_path "$WORK/sparse/0" \
  --output_type TXT

echo "COLMAP done. Validate with: python -c \"from nvs.colmap_io import registration_rate; \
from nvs.validation import require_registration_rate; \
n=len(list(__import__('pathlib').Path('$FRAMES').glob('frame_*.png'))); \
require_registration_rate(registration_rate('$WORK/sparse/0/images.txt', n)); print('OK')\""
```

- [ ] **Step 2: Make executable + run on GPU machine**

Run: `chmod +x scripts/03_run_colmap.sh && scripts/03_run_colmap.sh data/frames masks colmap`
Then run the printed validation command. Expected: prints `OK`. If it raises, registration was too low — re-sample frames denser or switch to `colmap exhaustive_matcher`.

- [ ] **Step 3: Commit**

```bash
git add scripts/03_run_colmap.sh
git commit -m "feat: stage 03 COLMAP SfM with feature masks"
```

---

## Task 5: gsplat training (Method C)

**Files:**
- Create: `scripts/04_train_gsplat.py`

gsplat ships `examples/simple_trainer.py`. The cleanest path is to invoke it as a subprocess on the masked COLMAP dataset. simple_trainer reads masks from the dataset if present (per-image alpha); we pass our COLMAP workdir directly.

> **Verify against the pinned gsplat version:** flag names (`--data_dir`, `--data_factor`, `--result_dir`) match current `simple_trainer.py default`. Confirm `examples/simple_trainer.py` exists in the installed gsplat (it lives in the repo's `examples/`; if pip-installed without examples, clone gsplat into `third_party/` in setup.sh and point `--trainer` at it).

- [ ] **Step 1: Write the training wrapper**

```python
"""Stage 04 (Method C): train gsplat on the masked COLMAP dataset.

Usage:
  python scripts/04_train_gsplat.py --data colmap --out outputs/gsplat \
      --trainer third_party/gsplat/examples/simple_trainer.py --device 0 --max-steps 30000
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

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
    args = p.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, args.trainer, "default",
        "--data_dir", args.data,
        "--data_factor", str(args.data_factor),
        "--result_dir", args.out,
        "--max_steps", str(args.max_steps),
    ]
    env = {"CUDA_VISIBLE_DEVICES": args.device}
    print("Running:", " ".join(cmd), "with CUDA_VISIBLE_DEVICES=" + args.device)
    subprocess.run(cmd, check=True, env={**__import__("os").environ, **env})

    require_nonempty_dir(Path(args.out) / "ckpts", "*.pt")
    print(f"gsplat training complete -> {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke run (short) on GPU machine**

Run with `--max-steps 500` first. Expected: a checkpoint appears under `outputs/gsplat/ckpts/`. `require_nonempty_dir` passes. Then run the full 30000.

- [ ] **Step 3: Commit**

```bash
git add scripts/04_train_gsplat.py
git commit -m "feat: stage 04 gsplat training (Method C)"
```

---

## Task 6: wild-gaussians training (Method B)

**Files:**
- Create: `scripts/04_train_wildgs.py`

> **Verify against pinned wild-gaussians:** its CLI is `wild-gaussians train --data <colmap> --output <dir>` (per its README). Confirm the exact subcommand/flags after `setup.sh`; adjust the `cmd` list to match.

- [ ] **Step 1: Write the training wrapper**

```python
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
```

- [ ] **Step 2: Smoke run on GPU machine**

Run a short-iteration variant per the tool's flags; confirm a model artifact appears in `outputs/wildgs/`.

- [ ] **Step 3: Commit**

```bash
git add scripts/04_train_wildgs.py
git commit -m "feat: stage 04 wild-gaussians training (Method B)"
```

---

## Task 7 (pure): Image quality metrics

**Files:**
- Create: `nvs/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_metrics.py`:
```python
import numpy as np
from nvs.metrics import psnr, ssim


def test_psnr_identical_is_infinite():
    img = np.random.rand(32, 32, 3).astype(np.float32)
    assert psnr(img, img) == float("inf")


def test_psnr_decreases_with_noise():
    img = np.full((32, 32, 3), 0.5, dtype=np.float32)
    noisy = np.clip(img + 0.1, 0, 1).astype(np.float32)
    assert 0 < psnr(img, noisy) < 100


def test_ssim_identical_is_one():
    img = np.random.rand(64, 64, 3).astype(np.float32)
    assert abs(ssim(img, img) - 1.0) < 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_metrics.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'nvs.metrics'`

- [ ] **Step 3: Implement nvs/metrics.py**

```python
"""Image-quality metrics. PSNR/SSIM are pure numpy/skimage; LPIPS is lazy (torch)."""
from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """a, b are float images in [0, 1]."""
    mse = float(np.mean((a - b) ** 2))
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Structural similarity for [0,1] RGB images."""
    return float(structural_similarity(a, b, channel_axis=-1, data_range=1.0))


def lpips_fn(device: str = "cuda"):
    """Return a callable (a,b)->float using LPIPS(AlexNet). Lazy: imports torch."""
    import lpips as _lpips
    import torch

    net = _lpips.LPIPS(net="alex").to(device).eval()

    def _score(a: np.ndarray, b: np.ndarray) -> float:
        def to_t(x):
            t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()
            return (t * 2 - 1).to(device)  # LPIPS expects [-1,1]
        with torch.no_grad():
            return float(net(to_t(a), to_t(b)).item())

    return _score
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_metrics.py`
Expected: PASS (3 passed). (LPIPS is not unit-tested — it needs torch + weights; it's exercised in stage 05.)

- [ ] **Step 5: Commit**

```bash
git add nvs/metrics.py tests/test_metrics.py
git commit -m "feat: PSNR/SSIM/LPIPS metrics"
```

---

## Task 8 (pure): Camera trajectory generation

**Files:**
- Create: `nvs/trajectory.py`
- Test: `tests/test_trajectory.py`

We need: scene centroid + mean radius/height from camera centers, and for a reference camera, 4 novel poses on an elevated arc around the centroid. COLMAP gives world-to-camera (R, t); the camera center in world coords is `C = -R^T t`.

- [ ] **Step 1: Write the failing tests**

`tests/test_trajectory.py`:
```python
import numpy as np
from nvs.trajectory import (
    qvec_to_rotmat, camera_center, scene_stats, arc_poses_around,
)


def test_qvec_identity():
    R = qvec_to_rotmat((1.0, 0.0, 0.0, 0.0))
    assert np.allclose(R, np.eye(3))


def test_camera_center_identity_pose():
    # R=I, t=(1,2,3) -> center = -I^T t = (-1,-2,-3)
    c = camera_center((1.0, 0.0, 0.0, 0.0), (1.0, 2.0, 3.0))
    assert np.allclose(c, [-1.0, -2.0, -3.0])


def test_scene_stats_centroid_and_radius():
    centers = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=float)
    centroid, radius, height = scene_stats(centers)
    assert np.allclose(centroid, [0, 0, 0])
    assert abs(radius - 1.0) < 1e-6


def test_arc_poses_count_and_look_at_centroid():
    poses = arc_poses_around(
        centroid=np.zeros(3), radius=2.0, base_height=0.0,
        start_angle=0.0, n=4, arc_span=np.pi / 2, elevation=0.5,
    )
    assert len(poses) == 4
    for R, t in poses:
        C = -R.T @ t                      # camera center
        forward = R[2]                    # world-to-cam: row 2 is viewing dir
        to_center = -C / np.linalg.norm(C)
        assert np.dot(forward / np.linalg.norm(forward), to_center) > 0.9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_trajectory.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'nvs.trajectory'`

- [ ] **Step 3: Implement nvs/trajectory.py**

```python
"""Camera-pose math and novel-view arc generation (COLMAP world-to-camera)."""
from __future__ import annotations

import numpy as np


def qvec_to_rotmat(qvec: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def camera_center(qvec, tvec) -> np.ndarray:
    R = qvec_to_rotmat(qvec)
    return -R.T @ np.asarray(tvec, dtype=float)


def scene_stats(centers: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return (centroid, mean_radius, mean_height) of camera centers."""
    centroid = centers.mean(axis=0)
    radius = float(np.linalg.norm(centers - centroid, axis=1).mean())
    height = float(centers[:, 1].mean())
    return centroid, radius, height


def _look_at(eye: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build world-to-camera (R, t) for a camera at `eye` looking at `target`."""
    up = np.array([0.0, 1.0, 0.0])
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, fwd)
    # world-to-camera rows: [right, -up, forward] (OpenCV-style: +z forward)
    R = np.stack([right, -true_up, fwd], axis=0)
    t = -R @ eye
    return R, t


def arc_poses_around(
    centroid: np.ndarray, radius: float, base_height: float,
    start_angle: float, n: int, arc_span: float, elevation: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """`n` poses on an elevated horizontal arc around `centroid`, all looking at it."""
    poses = []
    angles = (np.linspace(0, arc_span, n) + start_angle) if n > 1 else [start_angle]
    for a in angles:
        eye = centroid + np.array([
            radius * np.cos(a),
            base_height + elevation,
            radius * np.sin(a),
        ])
        poses.append(_look_at(eye, centroid))
    return poses
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_trajectory.py`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add nvs/trajectory.py tests/test_trajectory.py
git commit -m "feat: camera trajectory generation for novel views"
```

---

## Task 8 (script): Orbit rendering + held-out eval

**Files:**
- Create: `scripts/06_render_orbits.py`
- Create: `scripts/05_eval_heldout.py`

> **Verify against pinned gsplat:** checkpoint dict keys. simple_trainer saves `outputs/gsplat/ckpts/ckpt_<step>.pt` as `{"step":..., "splats": {"means","quats","scales","opacities","sh0","shN"}}`. Confirm key names; adjust loader if different. `rasterization` import is `from gsplat import rasterization`.

- [ ] **Step 1: Write the orbit renderer**

```python
"""Stage 06: render 4 novel views per reference frame from a gsplat checkpoint.

Usage:
  python scripts/06_render_orbits.py --colmap colmap --ckpt outputs/gsplat/ckpts/ckpt_29999.pt \
      --out outputs/gsplat/novel --n-refs 25 --views-per-ref 4
"""
from __future__ import annotations

import argparse
from pathlib import Path

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

    # Intrinsics: read fx,fy,cx,cy from cameras.txt (assume single shared camera).
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
```

- [ ] **Step 2: Write the held-out eval script**

```python
"""Stage 05: render held-out poses and compute PSNR/SSIM/LPIPS.

Held-out frames = every Kth frame (excluded from training by name list).
Usage:
  python scripts/05_eval_heldout.py --colmap colmap --frames data/frames \
      --ckpt outputs/gsplat/ckpts/ckpt_29999.pt --out outputs/gsplat/eval --every 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from gsplat import rasterization

from nvs.colmap_io import parse_images_txt
from nvs.metrics import psnr, ssim, lpips_fn
from nvs.trajectory import qvec_to_rotmat

import importlib.util
spec = importlib.util.spec_from_file_location(
    "render_orbits", str(Path(__file__).with_name("06_render_orbits.py")))
render_orbits = importlib.util.module_from_spec(spec)
spec.loader.exec_module(render_orbits)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--colmap", default="colmap")
    p.add_argument("--frames", default="data/frames")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--every", type=int, default=10)
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

    summary = {k: float(np.mean([r[k] for r in rows])) for k in ("psnr", "ssim", "lpips")}
    (out / "metrics.json").write_text(json.dumps(
        {"per_image": rows, "mean": summary}, indent=2))
    print("Mean metrics:", summary)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke run both on GPU machine**

Run stage 06 then stage 05 against the short-trained checkpoint. Expected: 100 PNGs in `outputs/gsplat/novel/`; `outputs/gsplat/eval/metrics.json` with finite mean PSNR/SSIM/LPIPS.

- [ ] **Step 4: Commit**

```bash
git add scripts/06_render_orbits.py scripts/05_eval_heldout.py
git commit -m "feat: stage 05 held-out eval and stage 06 orbit rendering"
```

---

## Task 9: ZeroNVS diffusion baseline

**Files:**
- Create: `scripts/07_run_zeronvs.py`

> **Verify against pinned ZeroNVS:** its inference entrypoint and pose-conditioning format. ZeroNVS is threestudio-based; the most robust integration is a subprocess call to its provided inference script. The wrapper below selects the same 25 reference frames and the same arc angles as stage 06, then shells out per reference. Adjust the inner command to the repo's actual inference CLI after setup.

- [ ] **Step 1: Write the ZeroNVS wrapper**

```python
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
from pathlib import Path

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
    # Same relative yaw offsets as the gsplat arc, centered on the reference.
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
```

- [ ] **Step 2: Run on GPU machine after ZeroNVS env is set up**

Expected: 100 PNGs in `outputs/zeronvs/`, named to match the gsplat outputs for side-by-side report figures.

- [ ] **Step 3: Commit**

```bash
git add scripts/07_run_zeronvs.py
git commit -m "feat: stage 07 ZeroNVS single-image NVS baseline"
```

---

## Task 10: README, report scaffold, full-pipeline smoke

**Files:**
- Create: `README.md`
- Create: `report/README.md` (figure/section checklist for the PDF)

- [ ] **Step 1: Write README.md**

````markdown
# Novel View Synthesis — Robot Manipulation Scene

Reproducible pipeline producing ≥100 novel views (25 refs × 4) via gsplat+masks,
wild-gaussians, and ZeroNVS, plus held-out PSNR/SSIM/LPIPS.

## Install
1. Install COLMAP (`brew install colmap` / apt / conda) and a CUDA-enabled PyTorch.
2. `pip install -r requirements.txt`
3. `./setup.sh`  # clones SAM2, wild-gaussians, ZeroNVS at pinned SHAs + weights

## Run (single GPU, sequential)
```bash
python scripts/01_extract_frames.py --video input/nvs_example_input_video.mp4 --out data/frames
python scripts/02_make_masks.py --frames data/frames --out masks --ckpt <sam2_ckpt> --cfg <sam2_cfg> --points 240,360 --labels 1
scripts/03_run_colmap.sh data/frames masks colmap
python scripts/04_train_gsplat.py --data colmap --out outputs/gsplat --trainer third_party/gsplat/examples/simple_trainer.py
python scripts/04_train_wildgs.py --data colmap --out outputs/wildgs
python scripts/05_eval_heldout.py --colmap colmap --frames data/frames --ckpt outputs/gsplat/ckpts/ckpt_29999.pt --out outputs/gsplat/eval
python scripts/06_render_orbits.py --colmap colmap --ckpt outputs/gsplat/ckpts/ckpt_29999.pt --out outputs/gsplat/novel
python scripts/07_run_zeronvs.py --colmap colmap --frames data/frames --out outputs/zeronvs --zeronvs third_party/ZeroNVS
```

## Tests
`python -m pytest`   # pure-logic unit tests (no GPU required)
````

- [ ] **Step 2: Write report/README.md (PDF checklist)**

```markdown
# Report checklist (PDF deliverable)
- (a) Approach & steps: pipeline diagram (frames→masks→COLMAP→{gsplat,wildgs}→render; +ZeroNVS)
- (b) Results: grid of 25 refs × 4 novel views for gsplat AND wild-gaussians;
      metrics table (mean PSNR/SSIM/LPIPS from outputs/*/eval/metrics.json)
- (b) Comparison figures: gsplat vs wild-gaussians (arm handling); 3DGS vs ZeroNVS (same ref+angle)
- (c) Failure cases: arm residue/floaters, extrapolation breakdown, ZeroNVS identity drift
- (d) Limitations & fixes: sparse coverage, dynamic content, diffusion geometry inconsistency
```

- [ ] **Step 3: Run the full unit-test suite (clean machine check)**

Run: `python -m pytest`
Expected: all pure-logic tests pass (sharpness, masks, colmap_io, validation, metrics, trajectory).

- [ ] **Step 4: End-to-end smoke run (GPU machine)**

Run the README commands with `--target 20` (stage 01) and `--max-steps 500` (stage 04) to confirm the whole chain executes on the real video before a full multi-hour run. Expected: novel views and metrics.json produced without error.

- [ ] **Step 5: Commit**

```bash
git add README.md report/README.md
git commit -m "docs: README, report checklist, full-pipeline smoke"
```

---

## Self-Review Notes (addressed)

- **Spec coverage:** frames+blur (T2), SAM2 masks (T3), COLMAP+masks+registration gate (T4), gsplat Method C (T5), wild-gaussians Method B (T6), held-out PSNR/SSIM/LPIPS (T7+T8 script), orbit rendering 25×4 (T8), ZeroNVS (T9), setup.sh vendoring (T1), README/report (T10). I2I stretch goal intentionally omitted (optional per spec §3).
- **Single-GPU sequential default:** stages 04 take `--device`; README runs them sequentially. Multi-GPU is available by passing different `--device` values but is not the default (spec §9).
- **External-tool API risk:** gsplat flags/rasterization verified via docs. wild-gaussians, SAM2, ZeroNVS CLIs carry explicit `TODO(verify)` against pinned SHAs — confirm during execution.
- **Type consistency:** `load_splats` defined in 06 and reused by 05 via import; `parse_images_txt`/`camera_center`/`scene_stats`/`arc_poses_around` signatures consistent across scripts.

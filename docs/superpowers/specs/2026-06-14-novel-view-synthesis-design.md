# Novel View Synthesis of a Robot Manipulation Scene — Design Spec

**Date:** 2026-06-14
**Status:** Approved design, pending implementation plan

## 1. Problem & Goal

Given a single ~4-minute video (480×480, 30 fps, 7119 frames) of a robot  
manipulating objects on a cloth-covered table in a gym-like room, produce  
**novel views** — renderings of the scene from camera viewpoints not present in
the video.

**Deliverables (per the assignment):**

- A GitHub repo with install + run instructions.
- A PDF report covering: (a) approach & steps, (b) results on ≥25 reference
images with **4 novel views each** (≥100 novel views), (c) failure cases,
(d) limitations & how to address them.

## 2. Input Analysis (what the video actually contains)

Confirmed by sampling 7 frames across the timeline:

- **Camera:** eye-in-hand / FPS view that translates and rotates substantially
around the table → **strong parallax**, good for Structure-from-Motion (SfM).
- **Scene:** **mostly static** — a gray tablecloth, a yellow goblet, a green
bottle (early frames), stickers, wood floor, wall partition.
- **Dynamic content:** the **robot arm/gripper is a transient foreground
occluder** present in a large fraction of frames; there is **minor object
manipulation** at one moment (the goblet is touched; the bottle appears to be
moved/leaves frame). No large-scale rearrangement.

**Implication:** This is a *mostly-static scene with a moving foreground
occluder*, not a fully dynamic scene. A full 4D/dynamic method is unnecessary.
The arm must be excluded from reconstruction (either by explicit masks or by a
transient-aware model) to avoid floaters and corrupted camera poses.

**Note on "novel views of the robot":** because the camera is mounted on the
arm (eye-in-hand), the arm is the *viewpoint*, not a separable subject held
still in front of the camera. The realistic, well-posed target is therefore
**novel views of the robot's workspace/scene with the arm removed** — which is
exactly what the assignment's reference example outputs are (a static scene, no
robot visible).

## 3. Chosen Approach

Three reconstruction/synthesis methods, sharing a common front-end, to enable
two comparisons in the report (robust-vs-masking, and 3DGS-vs-diffusion).

**Shared front-end:** video → blur-filtered frames → COLMAP SfM (camera poses +
sparse point cloud).

**3DGS route (two methods, compared):**

- **Method C — Mask + gsplat:** SAM2 video segmentation produces per-frame arm
masks. Masks are used (1) as COLMAP feature masks so the arm produces no
feature matches, and (2) as gsplat loss/alpha masks so training never tries
to explain arm pixels. Plain gsplat (mature, fast) trains on the static scene.
- **Method B — wild-gaussians:** run on the same frames; its per-image
uncertainty model treats the arm (and the briefly-moved object) as transient
occluders automatically — **no masks required**.

B and C converge on the same kind of output (clean static scene, arm removed);
they differ in *how* the arm is excluded (model-inferred uncertainty vs.
explicit masks). This difference is itself a report result.

**Diffusion route (single-image baseline):**

- **ZeroNVS** — zero-shot single-image NVS, conditioned on one reference image +
a relative target pose. Run on the **same 25 reference frames** used by the
3DGS route so viewpoints are directly comparable.
- **"From an Image to a Scene" (I2I)** — optional stretch goal if ZeroNVS lands
cleanly and time remains.

## 4. Repository Structure

```
novel-view-synthesis/
├── README.md                  # install + run instructions (deliverable)
├── setup.sh                   # clone third-party repos at pinned SHAs + fetch weights
├── data/
│   ├── input_video.mp4
│   └── frames/                # extracted frames (gitignored)
├── masks/                     # SAM2 arm masks (gitignored)
├── colmap/                    # SfM output: poses + sparse cloud (gitignored)
├── scripts/
│   ├── 01_extract_frames.py   # ffmpeg + blur (variance-of-Laplacian) filtering
│   ├── 02_make_masks.py        # SAM2 arm/gripper masks (feed COLMAP + gsplat)
│   ├── 03_run_colmap.sh        # SfM: sequential matching + mapper (uses feature masks)
│   ├── 04_train_gsplat.py      # Method C: 3DGS w/ loss masks  ┐ sibling stage,
│   ├── 04_train_wildgs.py      # Method B: wild-gaussians        ┘ parallelizable (2 GPUs)
│   ├── 05_eval_heldout.py      # PSNR/SSIM/LPIPS on held-out frames (B & C)
│   ├── 06_render_orbits.py     # auto orbit/arc trajectories → 4 views per ref (B & C)
│   └── 07_run_zeronvs.py       # diffusion baseline (ZeroNVS)
├── outputs/
│   ├── gsplat/                 # Method C model + novel views + metrics
│   ├── wildgs/                 # Method B model + novel views + metrics
│   └── zeronvs/                # diffusion novel views
├── report/                     # PDF report + figures
└── third_party/                # wild-gaussians, SAM2, ZeroNVS (cloned by setup.sh, gitignored)
```

**Third-party code & vendoring.** The methods themselves are existing research
code; our `scripts/` are thin wrappers that get data into each tool's expected
format, invoke its entrypoint, and collect outputs. Acquisition:

- **COLMAP** — installed binary (`brew install colmap` / apt / conda), invoked
  via CLI by `03_run_colmap.sh`. Not vendored.
- **gsplat** — `pip install gsplat` (compiles CUDA kernels). Not vendored.
- **wild-gaussians, SAM2, ZeroNVS** — cloned by `setup.sh` into `third_party/`
  **at pinned commit SHAs**, with model weights downloaded (SAM2 checkpoint,
  ZeroNVS weights). `setup.sh` is preferred over git submodules because these
  tools are *consumed as-is* (not modified) and require weight downloads +
  CUDA/conda setup that submodules cannot express; pinned SHAs in the script
  give the same reproducibility as submodules with one legible install command.

Each numbered script is a standalone stage: it reads the previous stage's output
from disk and writes its own, so any stage can be re-run and debugged in
isolation.

## 5. Pipeline Stages

**01 — Frame extraction & blur filtering.** Extract ~150–300 frames from the
video (even temporal sampling). Reject motion-blurred frames via
variance-of-Laplacian sharpness threshold (blurry frames poison SfM matching).

**02 — Arm masks (SAM2).** Click the arm in one frame; SAM2 propagates the
mask through the video. Add correction clicks where it drifts (~5–10 min total).
Mask also covers the briefly-manipulated object during its motion window. Runs
*before* COLMAP because its masks feed both COLMAP (stage 03) and gsplat
(stage 04). Fallback if SAM2 install is painful: black-arm color/brightness
threshold or a fixed foreground-region mask.

**03 — COLMAP SfM.** Feature extraction → **sequential matching** (exploits
video frame overlap; far faster than exhaustive) → mapper (bundle adjustment) →
export poses + sparse cloud. Flags: `--ImageReader.single_camera 1` (one fixed
camera), feature masks from stage 02. Convert output to the format gsplat /
wild-gaussians expect.

**04 — Training (two sibling methods, same stage).** Both consume the shared
COLMAP output and write to disjoint outputs, so they are independent and may run
**sequentially on one GPU or in parallel across two GPUs/machines** (see §9).

- **04 (C) — gsplat:** initialize Gaussians from the COLMAP sparse cloud, ~30k
  iterations with densification/pruning, **arm pixels excluded from the loss via
  masks**. Save model to `outputs/gsplat/`.
- **04 (B) — wild-gaussians:** same COLMAP input, no masks; transient occluders
  handled by the model's uncertainty. Save model to `outputs/wildgs/`.

**05 — Held-out evaluation.** Hold out ~10% of frames from training (both B and
C). Render their exact COLMAP poses; compute **PSNR / SSIM / LPIPS** vs. real
frames. (LPIPS weighted heavily — it tracks floater/sharp-but-wrong artifacts
that PSNR misses.) Kept separate from the 100 deliverable orbits.

**06 — Orbit rendering.** Compute scene centroid + mean camera radius/height
from COLMAP poses. Pick **25 reference frames** (even temporal sampling). For
each, anchor an elevated arc around the centroid at that frame's pose and sample
**4 novel poses** — offsets large enough to be clearly novel, but within the
reconstructed region so quality holds → **25×4 = 100 novel views per method**.

**07 — ZeroNVS.** For each of the same 25 reference frames: single image +
relative target pose → novel view, reusing the same target offsets as the 3DGS
orbits where ZeroNVS pose conditioning allows.

## 6. Deliverable Mapping (rubric compliance)

- **≥25 reference images, 4 novel views each:** 25 evenly-sampled frames × 4
arc-sampled novel poses = **100 novel views**, each traceable to its reference,
produced by each of gsplat (C), wild-gaussians (B), and ZeroNVS.
- **Quantitative results:** PSNR/SSIM/LPIPS on held-out frames (B and C).
- **Comparisons in report:** B vs. C (robust-by-design vs. explicit masking);
3DGS vs. ZeroNVS (full reconstruction vs. single-image diffusion prior).
- **Failure cases:** arm/occluder residue, floaters, extrapolation breakdown,
ZeroNVS identity drift & hallucinated occluded geometry.

## 7. Error Handling

Fail loud at stage boundaries — a half-working reconstruction is worse than one
that errors, because it surfaces only when the report looks bad.

- **03 (COLMAP):** assert most frames registered (common silent failure:
only ~10% registered). Report registration rate.
- **04 (training):** assert sane final Gaussian count; renders not black.
- **06 (rendering):** assert non-degenerate (non-black, non-empty) outputs.
- All stages validate inputs exist before running.

## 8. Testing

Research pipeline, not a unit-testable library → testing is mostly **validation
gates on real data**:

- **End-to-end smoke run:** 20 frames → COLMAP → 500-iter train → render 1 view,
to confirm the chain executes on the actual video before a multi-hour run.
- **Unit tests** for the pure/deterministic functions: sharpness scoring,
trajectory-generation math, metric computation.

## 9. Compute & Timing

Generic CUDA instructions (runs on local CUDA box or remote GPU server
interchangeably; Mac used for COLMAP prep/orchestration only — heavy training
not on MPS).

Rough wall-clock on an RTX 4090 / A6000-class GPU:


| Stage                         | Time                                |
| ----------------------------- | ----------------------------------- |
| Frame extraction              | ~1 min                              |
| SAM2 masks                    | ~5–10 min (interactive)             |
| COLMAP SfM                    | ~10–40 min (**main failure point**) |
| gsplat (C) 30k iters          | ~20–40 min                          |
| wild-gaussians (B)            | ~similar–longer                     |
| Render orbits                 | ~few min                            |
| ZeroNVS setup                 | hours (**main time sink**)          |
| ZeroNVS inference (100 views) | ~30–60 min                          |

**Training concurrency (stage 04).** Methods B and C share the immutable COLMAP
output and write to disjoint output dirs, so they are independent.
- **Default — single GPU, sequential:** run C then B. This is the target for
  implementation. Concurrent runs on one GPU only time-slice the same cores (no
  wall-clock gain) and risk CUDA OOM at peak Gaussian count, so we do not do that.
- **Future optimization — two GPUs/machines:** dispatch C to one device and B to
  the other for a genuine ~2× speedup. The stage scripts take an explicit device
  argument so this is available later without code changes, but it is **not** the
  default. Parallelism is gated by the *GPU resource*, not the code.


## 10. Risks & Mitigations

- **COLMAP fails to converge** (textureless floor, reflective surfaces, arm
occlusion) → sequential matching, arm feature masks, single-camera intrinsics;
validate registration rate and fall back to denser frame sampling if low.
- **Arm residue / floaters** → masks (C) and transient model (B); inspect
held-out renders.
- **Extrapolation breakdown** → anchor orbits near observed poses; keep offsets
within the reconstructed region.
- **ZeroNVS dependency hell** → time-boxed; it is a comparison baseline, not the
core deliverable; I2I only if ZeroNVS is clean. Also maybe a Docker image.
- **Minor object manipulation** → mask the moved object during its motion window
(C) or rely on transient handling (B); worst case drop that frame range.

## 11. Out of Scope (YAGNI)

- Dynamic / 4D Gaussian Splatting (scene is ~static).
- Reconstructing the moving arm as a subject.
- Real-time / interactive viewer.
- Multi-scene generalization (single input video only).


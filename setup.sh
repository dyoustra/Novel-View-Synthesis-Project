#!/usr/bin/env bash
# Clone third-party NVS tools at pinned commits and fetch model weights.
# Run on a CUDA machine, INSIDE your activated conda env. The env must already
# have the CUDA 13.0 toolchain + GCC 13 from README "Install" (nvcc must match
# torch's CUDA, and the host GCC must be old enough for nvcc — GCC 15 fails).
# COLMAP is installed separately (conda-forge / dnf) — see README "Install".
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TP="$ROOT/third_party"
mkdir -p "$TP"

# --- pinned versions (update here to bump) ---
SAM2_SHA="7e1596c0b6462eb1d1ba7e1492430fed95023598"
GSPLAT_VER="1.5.3"    # proven: CUDA 13.0 / torch cu130 / GCC 13
# Method B (wild-gaussians) runs via NerfBaselines from the `nb` env, not pinned
# here — pin the nerfbaselines version in that env instead (README "Install" step 5).
ZERONVS_SHA="main"    # ZeroNVS deferred (future work); unpinned until an actual run

clone_at() {  # url dir sha
  local url="$1" dir="$2" sha="$3"
  if [ ! -d "$TP/$dir/.git" ]; then
    git clone "$url" "$TP/$dir"
  fi
  git -C "$TP/$dir" fetch --all
  git -C "$TP/$dir" checkout "$sha"
}

# --- gsplat: pip wheel (library / CUDA kernels) + repo for the trainer ---
# The wheel ships `gsplat` the library but NOT examples/simple_trainer.py, which
# stage 04 invokes. Clone the repo at the tag matching the wheel so the trainer's
# API lines up, and install its example deps. --no-build-isolation: fused-ssim /
# fused-bilagrid import torch in their build, which an isolated build env hides.
pip install "gsplat==$GSPLAT_VER"
clone_at "https://github.com/nerfstudio-project/gsplat.git" "gsplat" "v$GSPLAT_VER"
pip install --no-build-isolation -r "$TP/gsplat/examples/requirements.txt"

# --- SAM2 ---
clone_at "https://github.com/facebookresearch/sam2.git" "sam2" "$SAM2_SHA"
pip install -e "$TP/sam2"
mkdir -p "$TP/sam2/checkpoints"
( cd "$TP/sam2/checkpoints" && bash download_ckpts.sh ) || \
  echo "WARN: SAM2 checkpoint download failed — fetch sam2.1_hiera_large.pt manually"

# --- wild-gaussians (Method B): intentionally NOT installed here ---
# It needs CUDA 11.8 / Py3.11 (incompatible with this CUDA-13 env) and ships as a
# NerfBaselines method. It runs from the separate `nb` env via `--backend conda`,
# which builds its isolated env automatically. See README "Install" step 5.

# --- ZeroNVS (stretch baseline) ---
clone_at "https://github.com/kylesargent/ZeroNVS.git" "ZeroNVS" "$ZERONVS_SHA"
echo "ZeroNVS cloned. Follow third_party/ZeroNVS/README for its conda env + weights."

echo "setup.sh complete."

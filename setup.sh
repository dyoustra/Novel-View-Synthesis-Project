#!/usr/bin/env bash
# Clone third-party NVS tools at pinned commits and fetch model weights.
# Run on a CUDA machine, INSIDE your activated conda env (so gsplat compiles
# against the conda CUDA toolkit / nvcc). COLMAP is installed separately
# (conda-forge / dnf) — see README "Install".
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

#!/usr/bin/env bash
# Stage 03: COLMAP SfM (sequential matching) producing poses + sparse cloud
# in the layout gsplat expects: <workdir>/{images, sparse/0}.
#
# Usage: scripts/03_run_colmap.sh data/frames masks colmap
set -euo pipefail

# COLMAP's CLI binary is Qt-linked and inits a GUI platform even for headless
# commands; over SSH (no X display) that aborts. Force the offscreen platform.
export QT_QPA_PLATFORM=offscreen

FRAMES="${1:-data/frames}"
MASKS="${2:-masks}"
WORK="${3:-colmap}"

mkdir -p "$WORK"
DB="$WORK/database.db"

# COLMAP looks for each mask at <mask_path>/<image_filename>.png — i.e. the FULL
# image name plus a doubled ".png" (mask for frame_00000.png is frame_00000.png.png).
# Our SAM2 masks are named frame_00000.png, so point COLMAP at a staging dir of
# correctly-named symlinks. A name mismatch makes COLMAP SILENTLY skip masking
# (no error) and extract features on the arm — so this naming must be exact.
MASK_ARG=()
if [ -d "$MASKS" ] && [ -n "$(ls -A "$MASKS" 2>/dev/null)" ]; then
  CMASKS="$WORK/colmap_masks"
  rm -rf "$CMASKS"; mkdir -p "$CMASKS"
  MASKS_ABS="$(cd "$MASKS" && pwd)"
  for m in "$MASKS_ABS"/*.png; do
    ln -sfn "$m" "$CMASKS/$(basename "$m").png"
  done
  MASK_ARG=(--ImageReader.mask_path "$CMASKS")
fi

# GPU SIFT uses OpenGL (SiftGPU), which needs a display context unavailable over
# headless SSH (opengl_utils context create fails). Use CPU SIFT — fast enough for
# 250 frames @ 480p. (On a machine with a display / EGL, drop the use_gpu flags.)
colmap feature_extractor \
  --database_path "$DB" \
  --image_path "$FRAMES" \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model OPENCV \
  --SiftExtraction.use_gpu 0 \
  "${MASK_ARG[@]}"

colmap sequential_matcher \
  --database_path "$DB" \
  --SiftMatching.use_gpu 0

mkdir -p "$WORK/sparse"
colmap mapper \
  --database_path "$DB" \
  --image_path "$FRAMES" \
  --output_path "$WORK/sparse"

# COLMAP may split into several sub-models and does NOT guarantee sparse/0 is the
# largest (it numbers by creation order). Pick the model with the most registered
# images and promote it to sparse/0, since downstream (gsplat/eval/render) all read
# sparse/0. Swap dirs so the largest becomes 0.
BEST=""; BEST_N=-1
for d in "$WORK"/sparse/*/; do
  [ -e "$d/images.bin" ] || [ -e "$d/images.txt" ] || continue
  idx="$(basename "$d")"
  n="$(colmap model_analyzer --path "$d" 2>&1 | grep -oE 'Registered images: [0-9]+' | grep -oE '[0-9]+' || echo 0)"
  echo "  model $idx: ${n:-0} registered images"
  if [ "${n:-0}" -gt "$BEST_N" ]; then BEST_N="${n:-0}"; BEST="$idx"; fi
done
echo "Largest model: sparse/$BEST ($BEST_N registered images)"
if [ -n "$BEST" ] && [ "$BEST" != "0" ]; then
  mv "$WORK/sparse/0" "$WORK/sparse/_swap_tmp"
  mv "$WORK/sparse/$BEST" "$WORK/sparse/0"
  mv "$WORK/sparse/_swap_tmp" "$WORK/sparse/$BEST"
fi

# gsplat wants images alongside sparse/0; symlink the frame dir.
ln -sfn "$(cd "$FRAMES" && pwd)" "$WORK/images"

# Convert the (now largest) model 0 to TXT for parsing/validation.
colmap model_converter \
  --input_path "$WORK/sparse/0" \
  --output_path "$WORK/sparse/0" \
  --output_type TXT

echo "COLMAP done. Validate registration rate with nvs.colmap_io + nvs.validation."

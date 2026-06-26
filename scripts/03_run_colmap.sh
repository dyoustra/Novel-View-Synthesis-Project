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

echo "COLMAP done. Validate registration rate with nvs.colmap_io + nvs.validation."

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

echo "COLMAP done. Validate registration rate with nvs.colmap_io + nvs.validation."

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
    """Parse COLMAP images.txt. Each registered image has a pose line followed
    by a 2D-points line (which may be empty). Recognize pose lines structurally
    (>=10 tokens, integer image id) and skip everything else, so empty point
    lines and trailing blanks don't break alignment."""
    poses: list[ImagePose] = []
    for ln in Path(path).read_text().splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 10 or not parts[0].lstrip("-").isdigit():
            continue  # 2D-points line or other non-pose content
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

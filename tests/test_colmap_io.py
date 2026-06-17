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

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


def test_parse_images_txt_handles_blank_points_line(tmp_path: Path):
    # second image has an EMPTY 2D-points line
    text = (
        "# comment\n"
        "1 1.0 0.0 0.0 0.0 0.5 0.0 0.0 1 frame_00000.png\n"
        "10.0 20.0 1.0\n"
        "2 0.707 0.0 0.707 0.0 1.0 2.0 3.0 1 frame_00001.png\n"
        "\n"  # blank points line (image with zero observed points)
        "3 1.0 0.0 0.0 0.0 9.0 9.0 9.0 1 frame_00002.png\n"
        "5.0 5.0 1.0\n"
    )
    f = tmp_path / "images.txt"
    f.write_text(text)
    poses = parse_images_txt(f)
    assert [p.name for p in poses] == [
        "frame_00000.png", "frame_00001.png", "frame_00002.png"]
    assert poses[2].tvec == (9.0, 9.0, 9.0)


def test_parse_images_txt_ignores_integer_points_line(tmp_path: Path):
    # a points line with >=10 integer tokens must NOT be parsed as a pose
    text = (
        "1 1.0 0.0 0.0 0.0 0.5 0.0 0.0 1 frame_00000.png\n"
        "10 20 1 30 40 2 50 60 3 70 80 4\n"
    )
    f = tmp_path / "images.txt"
    f.write_text(text)
    poses = parse_images_txt(f)
    assert len(poses) == 1
    assert poses[0].name == "frame_00000.png"


def test_parse_images_txt_tolerates_trailing_blank(tmp_path: Path):
    text = (
        "1 1.0 0.0 0.0 0.0 0.5 0.0 0.0 1 frame_00000.png\n"
        "10.0 20.0 1.0\n"
        "\n"  # trailing blank at EOF
    )
    f = tmp_path / "images.txt"
    f.write_text(text)
    poses = parse_images_txt(f)
    assert len(poses) == 1
    assert poses[0].name == "frame_00000.png"

import pytest
from nvs.validation import require_registration_rate, require_nonempty_dir


def test_require_registration_rate_passes():
    require_registration_rate(0.9, threshold=0.8)


def test_require_registration_rate_fails():
    with pytest.raises(RuntimeError, match="registration rate"):
        require_registration_rate(0.3, threshold=0.8)


def test_require_nonempty_dir(tmp_path):
    with pytest.raises(RuntimeError, match="no files"):
        require_nonempty_dir(tmp_path, "*.png")
    (tmp_path / "a.png").write_bytes(b"x")
    require_nonempty_dir(tmp_path, "*.png")

"""Fail-loud validation gates run at stage boundaries."""
from __future__ import annotations

from pathlib import Path


def require_registration_rate(rate: float, threshold: float = 0.8) -> None:
    if rate < threshold:
        raise RuntimeError(
            f"COLMAP registration rate {rate:.2f} < {threshold:.2f}. "
            "Try denser frame sampling, exhaustive matching, or check masks."
        )


def require_nonempty_dir(path: Path, pattern: str) -> None:
    if not list(Path(path).glob(pattern)):
        raise RuntimeError(f"{path} has no files matching {pattern!r}")

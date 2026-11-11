"""Vendored model code: Music-Source-Separation-Training (MIT), pinned to one commit.

MSST is not a pip package, so the repo is cloned into the cache and put on ``sys.path`` only
while a model class is imported.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

MSST_URL = "https://github.com/ZFTurbo/Music-Source-Separation-Training.git"
MSST_COMMIT = "1d4b3c431b37de696b3795ed2018493aacba946f"
VENDOR_DIR = Path.home() / ".cache" / "bandscribe" / "vendor"


class VendorError(RuntimeError):
    pass


def _git(args: list[str], cwd: Path) -> str:
    proc = subprocess.run(["git", *args], cwd=cwd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise VendorError(f"git {' '.join(args)} failed: {proc.stderr.strip()[:300]}")
    return proc.stdout.strip()


def ensure_msst() -> Path:
    """Return the MSST checkout at the pinned commit, cloning it on first use."""
    path = VENDOR_DIR / "msst"
    if not (path / "models").is_dir():
        VENDOR_DIR.mkdir(parents=True, exist_ok=True)
        _git(["clone", "-q", MSST_URL, "msst"], VENDOR_DIR)
    head = _git(["rev-parse", "HEAD"], path)
    if head != MSST_COMMIT:
        _git(["fetch", "-q", "origin", MSST_COMMIT], path)
        _git(["checkout", "-q", MSST_COMMIT], path)
    return path


def import_msst_model(module: str, name: str):
    """Import ``name`` from an MSST module such as ``models.scnet``."""
    import importlib

    root = str(ensure_msst())
    sys.path.insert(0, root)
    try:
        return getattr(importlib.import_module(module), name)
    finally:
        sys.path.remove(root)

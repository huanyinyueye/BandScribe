"""Download model files into the cache and enforce the registry sha256 pin (spec section 7)."""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

from bandscribe.models.registry import ModelFile, Registry, load_registry, verify_file

USER_AGENT = "bandscribe-model-fetch"


class UnresolvedFileError(RuntimeError):
    """The registry has no download URL for this file yet."""


def model_dir(reg: Registry, model_id: str) -> Path:
    return reg.cache_dir / model_id


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    with urllib.request.urlopen(req, timeout=120) as resp, part.open("wb") as out:  # noqa: S310
        shutil.copyfileobj(resp, out, length=1 << 20)
    part.replace(dest)


def ensure_file(model_id: str, file: ModelFile, reg: Registry | None = None) -> tuple[Path, str]:
    """Return (local path, sha256) for a registry file, downloading it if missing.

    A pinned hash that does not match raises ``HashMismatchError``; the file is kept for
    inspection but never loaded.
    """
    reg = reg or load_registry()
    path = model_dir(reg, model_id) / file.name
    if not path.is_file():
        if not file.url:
            raise UnresolvedFileError(f"{model_id}/{file.name}: no download URL in the registry")
        _download(file.url, path)
    return path, verify_file(path, file.sha256)


def ensure_model(model_id: str, reg: Registry | None = None) -> dict[str, tuple[Path, str]]:
    reg = reg or load_registry()
    entry = reg.get(model_id)
    return {f.name: ensure_file(model_id, f, reg) for f in entry.files}

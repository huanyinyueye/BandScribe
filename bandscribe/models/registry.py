"""Model registry: source URL, version, sha256, license and VRAM for every model.

Weights are only ever loaded through :func:`verify_file`, which enforces the sha256 pin
(spec section 7). A ``null`` hash means "not pinned yet" and is reported, never silently trusted.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REGISTRY_PATH = Path(__file__).with_name("registry.yaml")

TASKS = {"separation", "transcription", "drum_transcription", "beat_tracking", "tab"}
STATUSES = {"default", "candidate", "baseline", "disabled"}
VERIFICATION_TAGS = {"V-SRC", "V-2ND", "CONFLICT", "UNVERIFIED", "MUST-BENCH"}


class RegistryError(ValueError):
    """The registry file is malformed."""


class HashMismatchError(RuntimeError):
    """A weight file does not match its pinned sha256."""


@dataclass(frozen=True)
class ModelFile:
    name: str
    url: str | None
    sha256: str | None
    size_bytes: int | None


@dataclass(frozen=True)
class ModelEntry:
    id: str
    task: str
    name: str
    status: str
    milestone: str
    version: str | None
    source_urls: tuple[str, ...]
    files: tuple[ModelFile, ...]
    license_code: str | None
    license_weights: str | None
    license_note: str
    vram_measured_gb: float | None
    vram_note: str
    verification: tuple[str, ...]
    stems: tuple[str, ...] = field(default_factory=tuple)
    gated: bool = False  # needs a Hugging Face login and licence acceptance by the user

    @property
    def enabled_by_default(self) -> bool:
        return self.status == "default"

    @property
    def non_commercial(self) -> bool:
        return "NC" in (self.license_weights or "")


@dataclass(frozen=True)
class Registry:
    cache_dir: Path
    models: dict[str, ModelEntry]

    def get(self, model_id: str) -> ModelEntry:
        try:
            return self.models[model_id]
        except KeyError:
            raise KeyError(f"unknown model id {model_id!r}; known: {sorted(self.models)}") from None

    def by_task(self, task: str) -> list[ModelEntry]:
        return [m for m in self.models.values() if m.task == task]

    def unpinned_files(self) -> list[tuple[str, str]]:
        """(model_id, file name) pairs whose sha256 is not pinned yet."""
        return [(m.id, f.name) for m in self.models.values() for f in m.files if not f.sha256]


def _require(d: dict[str, Any], key: str, where: str) -> Any:
    if key not in d:
        raise RegistryError(f"{where}: missing required key {key!r}")
    return d[key]


def _parse_entry(model_id: str, raw: dict[str, Any]) -> ModelEntry:
    where = f"models.{model_id}"
    task = _require(raw, "task", where)
    if task not in TASKS:
        raise RegistryError(f"{where}: unknown task {task!r}")
    status = _require(raw, "status", where)
    if status not in STATUSES:
        raise RegistryError(f"{where}: unknown status {status!r}")
    urls = _require(raw, "source_urls", where)
    if not urls or not all(isinstance(u, str) and u.startswith("https://") for u in urls):
        raise RegistryError(f"{where}: source_urls must be a non-empty list of https URLs")
    lic = _require(raw, "license", where)
    vram = _require(raw, "vram_gb", where)
    tags = tuple(_require(raw, "verification", where))
    bad = set(tags) - VERIFICATION_TAGS
    if bad:
        raise RegistryError(f"{where}: unknown verification tags {sorted(bad)}")

    files = []
    for f in _require(raw, "files", where):
        sha = f.get("sha256")
        if sha is not None and (len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)):
            raise RegistryError(f"{where}: sha256 for {f.get('name')!r} is not 64 lowercase hex")
        url = f.get("url")
        if url is not None and not url.startswith("https://"):
            raise RegistryError(f"{where}: file url must be https")
        files.append(ModelFile(_require(f, "name", where), url, sha, f.get("size_bytes")))

    return ModelEntry(
        id=model_id,
        task=task,
        name=_require(raw, "name", where),
        status=status,
        milestone=_require(raw, "milestone", where),
        version=raw.get("version"),
        source_urls=tuple(urls),
        files=tuple(files),
        license_code=lic.get("code"),
        license_weights=lic.get("weights"),
        license_note=lic.get("note", ""),
        vram_measured_gb=vram.get("measured"),
        vram_note=vram.get("note", ""),
        verification=tags,
        stems=tuple(raw.get("stems") or ()),
        gated=bool(raw.get("gated", False)),
    )


def load_registry(path: Path | None = None) -> Registry:
    path = path or REGISTRY_PATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise RegistryError(f"{path}: schema_version must be 1")
    models_raw = _require(raw, "models", str(path))
    models = {mid: _parse_entry(mid, entry) for mid, entry in models_raw.items()}
    cache_dir = Path(_require(raw, "cache_dir", str(path))).expanduser()
    return Registry(cache_dir=cache_dir, models=models)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def verify_file(path: Path, expected_sha256: str | None) -> str:
    """Hash ``path``; raise if it differs from a pinned hash. Returns the actual hash.

    With ``expected_sha256=None`` the file is unpinned: the hash is returned so the caller can
    record it in the registry and the QA report, but nothing is asserted.
    """
    actual = sha256_file(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise HashMismatchError(
            f"{path.name}: sha256 {actual} does not match pinned {expected_sha256}"
        )
    return actual

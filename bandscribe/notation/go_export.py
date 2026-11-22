"""Run the Go exporter (``cmd/bandscribe-export``) on a score.

The binary is ``bin/bandscribe-export[.exe]`` in the repository, or ``BANDSCRIBE_EXPORTER``.
In a source checkout with Go installed it is rebuilt when a Go source file is newer than it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from bandscribe.notation.score_model import Score

ROOT = Path(__file__).resolve().parent.parent.parent
EXE = ROOT / "bin" / ("bandscribe-export.exe" if os.name == "nt" else "bandscribe-export")


class ExportError(RuntimeError):
    pass


def _stale(exe: Path) -> bool:
    if not exe.is_file():
        return True
    built = exe.stat().st_mtime
    sources = [ROOT / "go.mod", *(ROOT / "cmd").rglob("*.go"), *(ROOT / "internal").rglob("*.go")]
    return any(p.stat().st_mtime > built for p in sources if p.is_file())


def exporter() -> Path:
    env = os.environ.get("BANDSCRIBE_EXPORTER")
    if env:
        return Path(env)
    go = shutil.which("go")
    if go and (ROOT / "go.mod").is_file() and _stale(EXE):
        proc = subprocess.run([go, "build", "-o", str(EXE), "./cmd/bandscribe-export"], cwd=ROOT,
                              capture_output=True, text=True, encoding="utf-8", check=False)
        if proc.returncode != 0:
            raise ExportError(f"building bandscribe-export failed: {proc.stderr.strip()[-800:]}")
    if EXE.is_file():
        return EXE
    found = shutil.which("bandscribe-export")
    if found:
        return Path(found)
    raise ExportError("bandscribe-export not found; run `go build -o bin/ ./cmd/bandscribe-export`")


def export(score: Score, fmt: str, path: Path) -> dict:
    """Write one export (gp, gp5, midi, musicxml) and return the exporter's report for it."""
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "score.json"
        score.save(src)
        proc = subprocess.run([str(exporter()), "--score", str(src), f"--{fmt}", str(path)],
                              capture_output=True, text=True, encoding="utf-8", check=False)
    if proc.returncode != 0:
        raise ExportError(proc.stderr.strip()[-1500:] or f"bandscribe-export exited {proc.returncode}")
    result = json.loads(proc.stdout)["results"][0]
    if not result["ok"]:
        raise ExportError(result.get("error") or f"{fmt} export failed")
    return result.get("report") or {}

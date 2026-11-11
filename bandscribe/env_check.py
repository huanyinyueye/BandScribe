"""Environment detection for ``bandscribe doctor`` (spec 4.1 step 1, 7, 12.2, 12.5).

Every check returns a :class:`Check` and never raises: a broken tool is a finding, not a crash.
Severity: ``ok`` / ``warn`` (pipeline works with reduced features) / ``fail`` (cannot run).
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from bandscribe import __version__

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPORTER_DIR = REPO_ROOT / "exporter"

TARGET_PYTHON = (3, 11)
MIN_NODE_MAJOR = 18
MIN_VRAM_GB_RECOMMENDED = 9.5  # RTX 3080 10 GB reports ~10.0 GiB
MIN_FREE_DISK_GB = 20.0

Runner = Callable[[list[str]], "tuple[int, str]"]


@dataclass
class Check:
    name: str
    status: str  # ok | warn | fail
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    hint: str | None = None


def _run(cmd: list[str], timeout: float = 20.0) -> tuple[int, str]:
    """Run a command; return (exit code, combined output). 127 if it cannot be started."""
    try:
        proc = subprocess.run(
            cmd, check=False, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace"
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _which(name: str) -> str | None:
    return shutil.which(name)


def check_os() -> Check:
    system, release, version = platform.system(), platform.release(), platform.version()
    details = {"system": system, "release": release, "version": version,
               "machine": platform.machine()}
    if system == "Windows":
        # Windows 11 still reports release "10"; builds >= 22000 are Windows 11.
        build = version.rsplit(".", 1)[-1]
        if release == "10" and build.isdigit() and int(build) >= 22000:
            release = details["release"] = "11"
        return Check("os", "ok", f"Windows {release} ({version})", details)
    return Check("os", "warn", f"{system} {release}: only Windows 11 native is tested", details)


def check_python() -> Check:
    v = sys.version_info
    details = {"version": platform.python_version(), "executable": sys.executable}
    if (v.major, v.minor) == TARGET_PYTHON:
        return Check("python", "ok", f"Python {details['version']}", details)
    return Check("python", "warn", f"Python {details['version']} (project targets 3.11)", details,
                 hint="run through `uv run bandscribe ...` to use the pinned interpreter")


def check_uv(which: Callable[[str], str | None] = _which, run: Runner = _run) -> Check:
    for cmd in ([which("uv")] if which("uv") else []) + [[sys.executable, "-m", "uv"]]:
        argv = [cmd, "--version"] if isinstance(cmd, str) else [*cmd, "--version"]
        code, out = run(argv)
        if code == 0:
            on_path = isinstance(cmd, str)
            return Check("uv", "ok" if on_path else "warn", out.strip(),
                         {"on_path": on_path},
                         None if on_path else "uv works as a module but is not on PATH")
    # Inside a uv-managed venv uv itself is usually not importable; the env still proves it.
    if os.environ.get("UV") or (Path(sys.prefix) / "pyvenv.cfg").is_file() and "uv" in (
        Path(sys.prefix) / "pyvenv.cfg"
    ).read_text(encoding="utf-8", errors="replace"):
        return Check("uv", "ok", "running inside a uv-managed environment", {"on_path": False})
    return Check("uv", "warn", "uv not found", hint="install: https://docs.astral.sh/uv/")


def check_nvidia_smi(which: Callable[[str], str | None] = _which, run: Runner = _run) -> Check:
    exe = which("nvidia-smi")
    if not exe:
        return Check("gpu", "warn", "nvidia-smi not found: no NVIDIA driver; CPU only",
                     hint="install the NVIDIA driver")
    code, out = run([exe, "--query-gpu=name,memory.total,memory.used,driver_version",
                     "--format=csv,noheader,nounits"])
    if code != 0 or not out.strip():
        return Check("gpu", "warn", "nvidia-smi failed", {"output": out.strip()[:400]})
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            total, used = float(parts[1]), float(parts[2])
        except ValueError:
            continue
        gpus.append({"name": parts[0], "vram_total_gb": round(total / 1024, 2),
                     "vram_used_gb": round(used / 1024, 2), "driver": parts[3]})
    if not gpus:
        return Check("gpu", "warn", "could not parse nvidia-smi output",
                     {"output": out.strip()[:400]})
    g = gpus[0]
    summary = (f"{g['name']}, {g['vram_total_gb']} GB VRAM "
               f"({g['vram_used_gb']} GB in use), driver {g['driver']}")
    if g["vram_total_gb"] < MIN_VRAM_GB_RECOMMENDED:
        return Check("gpu", "warn", summary, {"gpus": gpus},
                     hint="below the 10 GB the defaults are tuned for; expect smaller chunks")
    return Check("gpu", "ok", summary, {"gpus": gpus})


def check_torch() -> Check:
    try:
        import torch
    except ImportError as exc:
        return Check("torch", "fail", f"PyTorch not importable: {exc}", hint="run `uv sync`")
    details: dict[str, Any] = {"version": torch.__version__,
                               "cuda_build": torch.version.cuda,
                               "cuda_available": torch.cuda.is_available()}
    if not torch.cuda.is_available():
        hint = ("this is a CPU-only torch build; reinstall from the CUDA index (`uv sync`)"
                if torch.version.cuda is None else "CUDA build present but no usable device")
        return Check("torch", "warn", f"torch {torch.__version__}: CUDA unavailable, CPU only "
                     "(very slow)", details, hint)
    try:
        props = torch.cuda.get_device_properties(0)
        details |= {"device": props.name,
                    "vram_total_gb": round(props.total_memory / 1024**3, 2),
                    "cudnn": torch.backends.cudnn.version()}
        # Prove the device actually computes, in FP16 as the separators will (spec 12.2).
        x = torch.ones(256, 256, device="cuda", dtype=torch.float16)
        details["fp16_matmul_ok"] = bool((x @ x)[0, 0].item() == 256.0)
        del x
        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001 - any CUDA failure is a finding
        return Check("torch", "fail", f"CUDA reported available but failed: {exc}", details)
    if not details["fp16_matmul_ok"]:
        return Check("torch", "fail", "CUDA FP16 smoke test returned a wrong result", details)
    return Check("torch", "ok", f"torch {torch.__version__} (CUDA {torch.version.cuda}) on "
                 f"{details['device']}, {details['vram_total_gb']} GB, FP16 OK", details)


def check_ffmpeg(which: Callable[[str], str | None] = _which, run: Runner = _run) -> Check:
    candidates: list[tuple[str, str]] = []
    if exe := which("ffmpeg"):
        candidates.append((exe, "PATH"))
    with contextlib.suppress(Exception):  # missing or broken bundle: just no candidate
        import imageio_ffmpeg

        candidates.append((imageio_ffmpeg.get_ffmpeg_exe(), "imageio-ffmpeg (bundled)"))
    for exe, origin in candidates:
        code, out = run([exe, "-version"])
        if code == 0:
            first = out.splitlines()[0] if out else ""
            m = re.search(r"ffmpeg version (\S+)", first)
            return Check("ffmpeg", "ok", f"ffmpeg {m.group(1) if m else '?'} via {origin}",
                         {"path": exe, "origin": origin, "ffprobe": which("ffprobe")})
    return Check("ffmpeg", "fail", "ffmpeg not found (needed to decode MP3)",
                 hint="run `uv sync` (bundles ffmpeg) or install ffmpeg on PATH")


def check_node(which: Callable[[str], str | None] = _which, run: Runner = _run) -> Check:
    exe = which("node")
    if not exe:
        return Check("node", "fail", "Node.js not found (needed for the .gp exporter and GUI "
                     "build)", hint=f"install Node.js >= {MIN_NODE_MAJOR}")
    code, out = run([exe, "--version"])
    m = re.match(r"v(\d+)\.(\d+)\.(\d+)", out.strip())
    if code != 0 or not m:
        return Check("node", "fail", "node --version failed", {"output": out.strip()[:200]})
    details: dict[str, Any] = {"version": out.strip(), "path": exe}
    npm = which("npm")
    if npm:
        ncode, nout = run([npm, "--version"])
        details["npm"] = nout.strip().splitlines()[0] if ncode == 0 and nout.strip() else None
    if int(m.group(1)) < MIN_NODE_MAJOR:
        return Check("node", "fail", f"Node {out.strip()} is too old", details,
                     hint=f"install Node.js >= {MIN_NODE_MAJOR}")
    if not details.get("npm"):
        return Check("node", "warn", f"Node {out.strip()} but npm not found", details,
                     hint="npm is needed to install alphaTab and build the GUI")
    return Check("node", "ok", f"Node {out.strip()}, npm {details['npm']}", details)


def check_alphatab(exporter_dir: Path = EXPORTER_DIR, which: Callable[[str], str | None] = _which,
                   run: Runner = _run) -> Check:
    pkg = exporter_dir / "node_modules" / "@coderline" / "alphatab" / "package.json"
    if not pkg.is_file():
        return Check("alphatab", "warn", "alphaTab not installed (needed for .gp export, M6)",
                     hint=f"run `npm install` in {exporter_dir}")
    try:
        meta = json.loads(pkg.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return Check("alphatab", "warn", f"alphaTab package.json unreadable: {exc}")
    details = {"version": meta.get("version"), "license": meta.get("license")}
    node = which("node")
    if node:
        # Confirm the exporter class the .gp path depends on really exists in this version.
        script = ("import('@coderline/alphatab').then(a=>{const ok=typeof a.exporter?."
                  "Gp7Exporter==='function';console.log(ok?'GP7_OK':'GP7_MISSING');})"
                  ".catch(e=>{console.log('IMPORT_FAIL '+e.message)})")
        prev = os.getcwd()
        try:
            os.chdir(exporter_dir)
            _, out = run([node, "-e", script])
        finally:
            os.chdir(prev)
        details["gp7_exporter"] = "GP7_OK" in out
        if not details["gp7_exporter"]:
            return Check("alphatab", "fail", f"alphaTab {details['version']} loaded but "
                         "Gp7Exporter is unavailable", details | {"output": out.strip()[:300]})
    return Check("alphatab", "ok", f"alphaTab {details['version']} ({details['license']}), "
                 "Gp7Exporter present", details)


_MUSESCORE_NAMES = ("MuseScore4", "mscore4", "MuseScore4.exe", "mscore", "musescore")


def _musescore_candidates() -> list[Path]:
    out = []
    for var in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
        base = os.environ.get(var)
        if base:
            out += [Path(base) / "MuseScore 4" / "bin" / "MuseScore4.exe",
                    Path(base) / "Programs" / "MuseScore 4" / "bin" / "MuseScore4.exe"]
    return out


def find_musescore(which: Callable[[str], str | None] = _which) -> str | None:
    if env := os.environ.get("BANDSCRIBE_MUSESCORE"):
        return env if Path(env).is_file() else None
    for name in _MUSESCORE_NAMES:
        if exe := which(name):
            return exe
    for path in _musescore_candidates():
        if path.is_file():
            return str(path)
    return None


def check_musescore(which: Callable[[str], str | None] = _which) -> Check:
    exe = find_musescore(which)
    if exe:
        return Check("musescore", "ok", f"MuseScore found: {exe}", {"path": exe})
    return Check("musescore", "warn", "MuseScore 4 not found: PDF export and the MuseScore "
                 "open-file test will be skipped",
                 hint="install MuseScore 4 or set BANDSCRIBE_MUSESCORE to MuseScore4.exe")


def check_registry() -> Check:
    try:
        from bandscribe.models.registry import load_registry

        reg = load_registry()
    except Exception as exc:  # noqa: BLE001 - malformed registry is a finding
        return Check("registry", "fail", f"model registry invalid: {exc}")
    unbenched = [m.id for m in reg.models.values() if m.vram_measured_gb is None]
    unresolved = [m.id for m in reg.models.values()
                  if not m.files or any(not f.sha256 for f in m.files)]
    details = {"models": len(reg.models), "cache_dir": str(reg.cache_dir),
               "vram_not_benchmarked": unbenched, "weights_not_pinned": unresolved,
               "non_commercial": [m.id for m in reg.models.values() if m.non_commercial]}
    summary = (f"{len(reg.models)} models; {len(unresolved)} without pinned sha256, "
               f"{len(unbenched)} without measured VRAM")
    return Check("registry", "ok", summary, details)


def check_disk(path: Path | None = None) -> Check:
    path = path or Path.home()
    try:
        free = shutil.disk_usage(path).free / 1024**3
    except OSError as exc:
        return Check("disk", "warn", f"cannot read free space at {path}: {exc}")
    details = {"path": str(path), "free_gb": round(free, 1)}
    if free < MIN_FREE_DISK_GB:
        return Check("disk", "warn", f"{free:.1f} GB free at {path}", details,
                     hint=f"model weights and stem caches need roughly {MIN_FREE_DISK_GB:.0f} GB")
    return Check("disk", "ok", f"{free:.1f} GB free at {path}", details)


ALL_CHECKS: tuple[Callable[[], Check], ...] = (
    check_os, check_python, check_uv, check_nvidia_smi, check_torch, check_ffmpeg,
    check_node, check_alphatab, check_musescore, check_registry, check_disk,
)


def run_checks(checks: tuple[Callable[[], Check], ...] = ALL_CHECKS) -> list[Check]:
    results = []
    for fn in checks:
        try:
            results.append(fn())
        except Exception as exc:  # noqa: BLE001 - doctor must always produce a full report
            results.append(Check(fn.__name__.removeprefix("check_"), "fail",
                                 f"check crashed: {type(exc).__name__}: {exc}"))
    return results


def overall_status(checks: list[Check]) -> str:
    statuses = {c.status for c in checks}
    return "fail" if "fail" in statuses else "warn" if "warn" in statuses else "ok"


def build_report(checks: list[Check]) -> dict[str, Any]:
    return {"bandscribe_version": __version__, "overall": overall_status(checks),
            "checks": [asdict(c) for c in checks]}


def format_report(checks: list[Check]) -> str:
    label = {"ok": "[ OK ]", "warn": "[WARN]", "fail": "[FAIL]"}
    width = max(len(c.name) for c in checks)
    lines = [f"BandScribe {__version__} environment report", ""]
    for c in checks:
        lines.append(f"{label[c.status]} {c.name.ljust(width)}  {c.summary}")
        if c.hint and c.status != "ok":
            lines.append(f"{' ' * (8 + width)} -> {c.hint}")
    counts = {s: sum(c.status == s for c in checks) for s in ("ok", "warn", "fail")}
    lines += ["", (f"Overall: {overall_status(checks).upper()} "
                  f"({counts['ok']} ok, {counts['warn']} warn, {counts['fail']} fail)")]
    return "\n".join(lines)

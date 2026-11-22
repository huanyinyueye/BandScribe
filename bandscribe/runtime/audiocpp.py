"""Find and run the audio.cpp runtime (audiocpp_cli.exe + its CUDA DLLs and model specs).

audio.cpp is BandScribe's default inference engine (see AGENTS.md). Each call is a separate
process, so VRAM is released when the process exits. The CLI writes its text results to
stdout, which can be megabytes for transcription, so stdout always goes to a log file.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

EXE = "audiocpp_cli.exe" if os.name == "nt" else "audiocpp_cli"
ENV_RUNTIME = "BANDSCRIBE_AUDIOCPP"
ENV_ENGINE = "BANDSCRIBE_ENGINE"  # auto (default) | audiocpp | torch


class AudiocppError(RuntimeError):
    pass


def _candidates() -> list[Path]:
    out = []
    if os.environ.get(ENV_RUNTIME):
        out.append(Path(os.environ[ENV_RUNTIME]))
    out.append(Path(sys.executable).resolve().parent / "audiocpp")  # packaged app layout
    out.append(Path(__file__).resolve().parents[2] / ".cache" / "audiocpp-runtime")  # dev
    return out


@lru_cache(maxsize=1)
def runtime_dir() -> Path | None:
    for d in _candidates():
        if (d / EXE).is_file() and (d / "model_specs").is_dir():
            return d
    return None


def engine() -> str:
    """Engine for separation, MuScriptor and beats: "audiocpp" (default) or "torch"."""
    want = os.environ.get(ENV_ENGINE, "auto").strip().lower()
    if want not in ("auto", "audiocpp", "torch"):
        raise ValueError(f"{ENV_ENGINE} must be auto, audiocpp or torch, got {want!r}")
    if want == "auto":
        return "audiocpp" if runtime_dir() is not None else "torch"
    if want == "audiocpp" and runtime_dir() is None:
        raise AudiocppError(f"audio.cpp runtime not found; set {ENV_RUNTIME} to its folder")
    return want


def _env(rt: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = str(rt) + os.pathsep + env.get("PATH", "")  # cuBLAS DLLs sit next to the exe
    return env


def _flags() -> int:
    return subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


@lru_cache(maxsize=1)
def list_devices() -> tuple[dict, ...]:
    rt = runtime_dir()
    if rt is None:
        return ()
    try:
        proc = subprocess.run([str(rt / EXE), "--list-devices", "--json"], capture_output=True,
                              text=True, timeout=60, env=_env(rt), creationflags=_flags())
        return tuple(json.loads(proc.stdout).get("devices", []))
    except (OSError, ValueError, subprocess.SubprocessError):
        return ()


def cuda_device() -> dict | None:
    return next((d for d in list_devices() if d.get("backend") == "cuda"), None)


def backend() -> str:
    return "cuda" if cuda_device() is not None else "cpu"


def threads() -> int:
    return max(1, min(16, (os.cpu_count() or 4) - 2))


def _call(cmd: list[str], rt: Path, log: Path, watch: list[Path] | None,
          on_done: Callable[[int], None] | None) -> int:
    with log.open("w", encoding="utf-8", errors="replace") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=_env(rt),
                                creationflags=_flags())
        done = 0
        try:
            while proc.poll() is None:
                time.sleep(0.5)
                if watch and on_done is not None:
                    now = sum(p.is_file() for p in watch)
                    if now > done:
                        done = now
                        on_done(done)  # may raise to cancel the job
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        return proc.returncode


def run(task: str, family: str, model: Path, args: list[str], log: Path,
        watch: list[Path] | None = None,
        on_done: Callable[[int], None] | None = None) -> dict[str, str]:
    """Run one CLI call and return the short ``key=value`` lines it printed (paths, metrics).

    stdout and stderr go to ``log``; on failure the tail of that log is in the exception.
    ``on_done(n)`` is called whenever the number of existing ``watch`` files grows, which
    gives real progress for request sequences (one output file per finished request).
    A failed CUDA run (GPU busy or out of VRAM) is retried once on CPU; the result's
    ``backend`` key says which one produced the output.
    """
    rt = runtime_dir()
    if rt is None:
        raise AudiocppError("audio.cpp runtime not found")
    started = time.perf_counter()
    log.parent.mkdir(parents=True, exist_ok=True)
    tried = []
    for dev in dict.fromkeys([backend(), "cpu"]):
        cmd = [str(rt / EXE), "--task", task, "--family", family, "--model", str(model),
               "--model-spec-override", str(rt / "model_specs"), "--backend", dev,
               "--threads", str(threads()), "--metrics", *args]
        for stale in watch or []:
            stale.unlink(missing_ok=True)  # partial outputs of a failed GPU run
        code = _call(cmd, rt, log, watch, on_done)
        text = log.read_text(encoding="utf-8", errors="replace")
        if code == 0:
            break
        tail = "\n".join(line[:300] for line in text.splitlines()[-15:])
        tried.append(f"{dev} exited with {code}:\n{tail}")
    else:
        raise AudiocppError(f"audiocpp {task}/{family} failed\n" + "\n".join(tried))
    out = {}
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep and len(line) < 2048 and " " not in key:
            out[key] = value
    out["backend"] = dev
    out["wall_seconds"] = f"{time.perf_counter() - started:.2f}"
    if tried:
        out["fallback"] = tried[0].splitlines()[0]
    return out

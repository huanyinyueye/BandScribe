"""Local web backend for the GUI (spec 12.4). Binds to 127.0.0.1 only.

The GUI has no transcription logic of its own: every job calls
``bandscribe.pipeline.run_pipeline`` with a ``PipelineConfig``. One job runs at a time (single
GPU); further jobs wait in a queue.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from bandscribe import __version__
from bandscribe.config import ConfigError, PipelineConfig, load_defaults
from bandscribe.pipeline import STAGES, PipelineCancelled

HOST = "127.0.0.1"
PROJECTS_DIR = Path.home() / ".cache" / "bandscribe" / "projects"
FRONTEND_DIST = Path(__file__).resolve().parent.parent.parent / "gui" / "dist"
SUPPORTED = {".mp3", ".wav", ".flac"}
# Rough share of the total time per stage, used only for the overall progress bar and ETA.
STAGE_WEIGHT = {"load": 0.01, "beats": 0.03, "separate": 0.30, "attribute": 0.03,
                "transcribe": 0.55, "tab": 0.01, "notation": 0.03, "qa": 0.01, "export": 0.03}
# Config fields the GUI may set.
EDITABLE = {"profile", "separator_primary", "separator_secondary", "pitched_transcribers",
            "drum_transcribers", "mix_vote", "low_confidence", "strict", "guitar_tuning",
            "bass_tuning", "instruments", "exports", "title", "max_span", "guitar_dp",
            "stem_overrides"}


@dataclass
class Job:
    id: str
    project: str
    config: dict[str, Any]
    status: str = "queued"  # queued | running | done | failed | cancelled
    stage: str = ""
    stage_fraction: float = 0.0
    overall: float = 0.0
    message: str = ""
    error: str | None = None
    started: float | None = None
    finished: float | None = None
    vram_gb: float = 0.0
    cancel_requested: bool = False
    version: int = 0  # bumped on every change; lets clients wait for updates
    _cond: threading.Condition = field(default_factory=threading.Condition, repr=False)

    def public(self) -> dict[str, Any]:
        elapsed = (self.finished or time.time()) - self.started if self.started else 0.0
        eta = None
        if self.status == "running" and self.overall > 0.03:
            eta = max(0.0, elapsed / self.overall - elapsed)
        return {"id": self.id, "project": self.project, "status": self.status,
                "stage": self.stage, "stage_fraction": round(self.stage_fraction, 4),
                "overall": round(self.overall, 4), "message": self.message, "error": self.error,
                "elapsed": round(elapsed, 1), "eta": None if eta is None else round(eta),
                "vram_gb": round(_vram_gb() if self.status == "running" else self.vram_gb, 2), "stages": list(STAGES), "version": self.version}

    def touch(self) -> None:
        with self._cond:
            self.version += 1
            self._cond.notify_all()

    def wait(self, version: int, timeout: float) -> None:
        with self._cond:
            if self.version == version:
                self._cond.wait(timeout)


class RunRequest(BaseModel):
    config: dict[str, Any] = {}


class OverrideRequest(BaseModel):
    state: str | None = None  # "absent" or None to clear


def _vram_gb() -> float:
    try:
        import torch

        return torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    except Exception:  # noqa: BLE001 - VRAM display is best effort
        return 0.0


class JobManager:
    """Single-worker job queue around ``run_pipeline``."""

    def __init__(self, projects_dir: Path, runner=None) -> None:
        self.projects_dir = projects_dir
        self.jobs: dict[str, Job] = {}
        self._queue: queue.Queue[Job] = queue.Queue()
        self._runner = runner
        self._worker = threading.Thread(target=self._loop, daemon=True, name="bandscribe-jobs")
        self._worker.start()

    def project_dir(self, project: str) -> Path:
        if not project.isalnum():
            raise HTTPException(404, "unknown project")
        path = self.projects_dir / project
        if not path.is_dir():
            raise HTTPException(404, "unknown project")
        return path

    def input_file(self, project: str) -> Path:
        files = [p for p in self.project_dir(project).glob("input.*") if p.suffix in SUPPORTED]
        if not files:
            raise HTTPException(404, "project has no audio")
        return files[0]

    def submit(self, project: str, config: dict[str, Any]) -> Job:
        job = Job(uuid.uuid4().hex[:12], project, config)
        self.jobs[job.id] = job
        self._queue.put(job)
        return job

    def active_for(self, project: str) -> Job | None:
        for job in reversed(list(self.jobs.values())):
            if job.project == project and job.status in ("queued", "running"):
                return job
        return None

    def _loop(self) -> None:
        while True:
            job = self._queue.get()
            if job.cancel_requested:
                job.status = "cancelled"
                job.touch()
                continue
            self._run(job)

    def _run(self, job: Job) -> None:
        from bandscribe.pipeline import run_pipeline

        runner = self._runner or run_pipeline
        job.status, job.started = "running", time.time()
        job.touch()
        done_weight = {s: sum(STAGE_WEIGHT[x] for x in STAGES[:i]) for i, s in enumerate(STAGES)}

        def progress(stage: str, fraction: float, message: str) -> None:
            job.stage, job.stage_fraction = stage, fraction
            if message:
                job.message = message
            job.overall = done_weight.get(stage, 0.0) + STAGE_WEIGHT.get(stage, 0.0) * fraction
            job.vram_gb = _vram_gb()
            job.touch()

        try:
            pdir = self.project_dir(job.project)
            overrides = {k: v for k, v in job.config.items() if k in EDITABLE and k != "profile"}
            cfg = PipelineConfig.from_defaults(job.config.get("profile", "quality"), **overrides)
            (pdir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=1), encoding="utf-8")
            runner(self.input_file(job.project), cfg, pdir / "run", progress,
                   lambda: job.cancel_requested)
            job.status, job.overall, job.stage_fraction = "done", 1.0, 1.0
        except PipelineCancelled:
            job.status = "cancelled"
        except Exception as exc:  # noqa: BLE001 - reported to the GUI
            job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
        finally:
            job.finished = time.time()
            job.vram_gb = _vram_gb()
            job.touch()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _waveform(path: Path, points: int = 1600) -> dict[str, Any]:
    """Min/max envelope for the import page."""
    import numpy as np

    from bandscribe.io.audio_load import load_audio

    audio = load_audio(path, normalize_peak=False)
    mono = audio.mono()
    hop = max(1, len(mono) // points)
    n = len(mono) // hop
    frames = mono[: n * hop].reshape(n, hop)
    return {"duration": round(audio.duration, 3), "orig_sr": audio.orig_sr,
            "orig_channels": audio.orig_channels, "sha256": audio.file_sha256,
            "min": [round(float(v), 4) for v in frames.min(axis=1)],
            "max": [round(float(v), 4) for v in frames.max(axis=1)],
            "peak": round(float(np.abs(mono).max()), 4) if len(mono) else 0.0}


def _quick_analysis(path: Path) -> dict[str, Any]:
    """Tempo, meter and a chroma-based key guess shown on the import page before a full run."""
    import librosa
    import numpy as np

    from bandscribe.analysis.beats import track_beats
    from bandscribe.analysis.key import MAJOR, MINOR, NAMES
    from bandscribe.analysis.meter import analyze_meter
    from bandscribe.analysis.tempo import build_tempo_map
    from bandscribe.io.audio_load import load_audio

    audio = load_audio(path)
    beats = track_beats(audio)
    tempo = build_tempo_map(beats.beats)
    meter = analyze_meter(beats.beats, beats.downbeats)
    chroma = librosa.feature.chroma_cqt(y=audio.mono(), sr=audio.sr).mean(axis=1)
    best = max(((float(np.corrcoef(chroma, np.roll(p, t))[0, 1]), t, m)
                for t in range(12) for m, p in (("major", MAJOR), ("minor", MINOR))))
    return {"bpm": tempo.global_bpm, "tempo_changes": len(tempo.segments) - 1 if tempo.segments else 0,
            "meter": list(meter.dominant), "meter_changes": max(0, len(meter.changes) - 1),
            "beats": len(beats.beats), "key_guess": f"{NAMES[best[1]]} {best[2]}",
            "key_note": "chroma-based guess; the full run estimates the key from the notes"}


def create_app(projects_dir: Path | None = None, runner=None,
               frontend_dist: Path | None = None) -> FastAPI:
    projects_dir = projects_dir or PROJECTS_DIR
    projects_dir.mkdir(parents=True, exist_ok=True)
    manager = JobManager(projects_dir, runner)
    app = FastAPI(title="BandScribe", version=__version__, docs_url=None, redoc_url=None)
    app.state.manager = manager

    @app.get("/api/info")
    def info() -> dict[str, Any]:
        from bandscribe.models.registry import load_registry
        from bandscribe.tab.tunings import TUNINGS

        reg = load_registry()
        return {
            "version": __version__,
            "defaults": {p: PipelineConfig.from_defaults(p).to_dict() for p in ("quality", "balanced")},
            "raw_defaults": load_defaults("quality"),
            "tunings": [{"key": t.key, "name": t.name, "instrument": t.instrument,
                         "strings": list(t.strings)} for t in TUNINGS.values()],
            "models": [{"id": m.id, "name": m.name, "task": m.task, "status": m.status,
                        "version": m.version, "license_code": m.license_code,
                        "license_weights": m.license_weights, "license_note": m.license_note,
                        "non_commercial": m.non_commercial, "gated": m.gated,
                        "source_urls": list(m.source_urls), "vram_note": m.vram_note}
                       for m in reg.models.values()],
            "notice": "Personal, non-commercial use only: the default pipeline uses weights "
                      "under CC BY-NC 4.0 (MuScriptor) and CC BY-NC-SA 4.0 (ADTOF). "
                      "Output must not be used commercially."}

    @app.get("/api/doctor")
    def doctor() -> dict[str, Any]:
        from bandscribe import env_check

        return env_check.build_report(env_check.run_checks())

    @app.get("/api/projects")
    def projects() -> list[dict[str, Any]]:
        out = []
        for p in sorted(projects_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            meta = _read_json(p / "project.json")
            if meta:
                manifest = _read_json(p / "run" / "manifest.json")
                out.append({**meta, "status": manifest["status"] if manifest else "new"})
        return out

    @app.post("/api/projects")
    async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in SUPPORTED:
            raise HTTPException(400, "unsupported format; use MP3, WAV or FLAC")
        data = await file.read()
        if not data:
            raise HTTPException(400, "empty file")
        pid = hashlib.sha256(data).hexdigest()[:16]
        pdir = projects_dir / pid
        existed = pdir.is_dir()
        pdir.mkdir(exist_ok=True)
        target = pdir / f"input{suffix}"
        if not existed:
            target.write_bytes(data)
        try:
            waveform = _waveform(target if target.is_file() else manager.input_file(pid))
        except Exception as exc:  # noqa: BLE001 - shown to the user
            if not existed:  # do not leave a broken project behind
                target.unlink(missing_ok=True)
                pdir.rmdir()
            raise HTTPException(400, f"cannot decode audio: {exc}") from exc
        meta = {"id": pid, "name": Path(file.filename).stem, "filename": file.filename,
                "created": time.time()}
        (pdir / "project.json").write_text(json.dumps(meta), encoding="utf-8")
        return {**meta, "waveform": waveform}

    @app.get("/api/projects/{project}")
    def project(project: str) -> dict[str, Any]:
        pdir = manager.project_dir(project)
        run = pdir / "run"
        active = manager.active_for(project)
        return {"meta": _read_json(pdir / "project.json"),
                "waveform": _waveform(manager.input_file(project)),
                "config": _read_json(pdir / "config.json"),
                "manifest": _read_json(run / "manifest.json"),
                "analysis": _read_json(run / "analysis.json"),
                "stems": _read_json(run / "stems" / "stems.json"),
                "qa": _read_json(run / "qa.json"),
                "attribution": _read_json(run / "notes" / "attribution.json"),
                "score": _score_summary(run / "score.json"),
                "exports": sorted(p.name for p in (run / "exports").glob("*")) if (run / "exports").is_dir() else [],
                "job": active.public() if active else None}

    @app.post("/api/projects/{project}/analyze")
    def analyze(project: str) -> dict[str, Any]:
        if any(j.status == "running" for j in manager.jobs.values()):
            raise HTTPException(409, "a transcription is running; the GPU is busy")
        return _quick_analysis(manager.input_file(project))

    @app.post("/api/projects/{project}/run")
    def run(project: str, req: RunRequest) -> dict[str, Any]:
        manager.input_file(project)
        if manager.active_for(project):
            raise HTTPException(409, "this project already has a job in progress")
        unknown = set(req.config) - EDITABLE
        if unknown:
            raise HTTPException(400, f"unknown config fields: {sorted(unknown)}")
        try:  # validate now so the user gets the error immediately
            overrides = {k: v for k, v in req.config.items() if k != "profile"}
            PipelineConfig.from_defaults(req.config.get("profile", "quality"), **overrides)
        except (ConfigError, TypeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return manager.submit(project, req.config).public()

    @app.post("/api/projects/{project}/stems/{stem}/override")
    def override(project: str, stem: str, req: OverrideRequest) -> dict[str, Any]:
        """Mark a stem as not containing its instrument (or clear the mark); saved in config."""
        pdir = manager.project_dir(project)
        cfg = _read_json(pdir / "config.json") or {}
        overrides = dict(cfg.get("stem_overrides") or {})
        if req.state is None:
            overrides.pop(stem, None)
        elif req.state == "absent":
            overrides[stem] = "absent"
        else:
            raise HTTPException(400, "state must be 'absent' or null")
        cfg["stem_overrides"] = overrides
        (pdir / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        return {"stem_overrides": overrides}

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str, wait: float = 0.0, version: int = -1) -> dict[str, Any]:
        j = manager.jobs.get(job_id)
        if j is None:
            raise HTTPException(404, "unknown job")
        if wait > 0:
            j.wait(version, min(wait, 25.0))
        return j.public()

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str) -> dict[str, Any]:
        j = manager.jobs.get(job_id)
        if j is None:
            raise HTTPException(404, "unknown job")
        j.cancel_requested = True
        j.message = "cancelling after the current step"
        j.touch()
        return j.public()

    @app.websocket("/api/ws/jobs/{job_id}")
    async def job_ws(ws: WebSocket, job_id: str) -> None:
        import asyncio

        await ws.accept()
        j = manager.jobs.get(job_id)
        if j is None:
            await ws.close(code=4404)
            return
        version, tick = -1, 0
        try:
            while True:
                tick += 1
                if j.version != version or tick % 8 == 0:  # at least every 2 s
                    version = j.version
                    await ws.send_json(j.public())
                    if j.status in ("done", "failed", "cancelled"):
                        break
                await asyncio.sleep(0.25)
        except WebSocketDisconnect:
            return
        await ws.close()

    @app.get("/api/projects/{project}/files/{path:path}")
    def files(project: str, path: str) -> FileResponse:
        pdir = manager.project_dir(project).resolve()
        target = (pdir / path).resolve()
        if pdir not in target.parents or not target.is_file():
            raise HTTPException(404, "file not found")
        return FileResponse(target, filename=target.name)

    @app.post("/api/projects/{project}/open-folder")
    def open_folder(project: str) -> dict[str, Any]:
        folder = manager.project_dir(project) / "run" / "exports"
        if not folder.is_dir():
            raise HTTPException(404, "nothing exported yet")
        if hasattr(os, "startfile"):
            os.startfile(folder)  # noqa: S606 - a fixed local folder, opened in Explorer
        return {"path": str(folder)}

    @app.exception_handler(ConfigError)
    async def config_error(_request, exc: ConfigError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    dist = frontend_dist or FRONTEND_DIST
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    else:
        @app.get("/")
        def no_frontend() -> JSONResponse:
            return JSONResponse({"detail": "frontend not built: run `npm install && npm run "
                                           "build` in the gui/ folder"}, status_code=503)
    return app


def _score_summary(path: Path) -> dict[str, Any] | None:
    data = _read_json(path)
    if data is None:
        return None
    return {"title": data["title"], "key": data["key"], "bars": data["bars"],
            "tracks": [{"id": t["id"], "name": t["name"], "family": t["family"], "kind": t["kind"],
                        "tuning_key": t["tuning_key"], "info": t["info"], "notes": len(t["notes"]),
                        "low_conf": sum(1 for n in t["notes"] if n["low_conf"])}
                       for t in data["tracks"]],
            "meta": data["meta"]}


def serve(port: int = 8765, open_window: bool = True, browser: bool = False) -> None:
    """Start the server on 127.0.0.1 and open the desktop window (or the browser)."""
    import uvicorn

    app = create_app()
    url = f"http://{HOST}:{port}"
    config = uvicorn.Config(app, host=HOST, port=port, log_level="warning")
    server = uvicorn.Server(config)
    if not open_window:
        server.run()
        return
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started and thread.is_alive():
        time.sleep(0.05)
    if browser:
        import webbrowser

        webbrowser.open(url)
        thread.join()
        return
    try:
        import webview
    except ImportError:
        import webbrowser

        webbrowser.open(url)
        thread.join()
        return
    webview.create_window("BandScribe", url, width=1400, height=900, min_size=(1000, 700))
    webview.start()
    server.should_exit = True

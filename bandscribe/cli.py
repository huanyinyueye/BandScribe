"""BandScribe command line. The GUI calls the same pipeline functions (spec 12.4)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="BandScribe: local band music transcription.")
models_app = typer.Typer(no_args_is_help=True, help="Inspect the model registry.")
app.add_typer(models_app, name="models")


def _utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


@app.command()
def doctor(
    as_json: bool = typer.Option(False, "--json", help="Print the report as JSON."),
) -> None:
    """Check OS, GPU/CUDA/VRAM, ffmpeg, Node/alphaTab and the model registry."""
    from bandscribe import env_check

    _utf8_stdout()
    checks = env_check.run_checks()
    if as_json:
        typer.echo(json.dumps(env_check.build_report(checks), indent=2, ensure_ascii=False))
    else:
        typer.echo(env_check.format_report(checks))
    raise typer.Exit(1 if env_check.overall_status(checks) == "fail" else 0)


@app.command()
def transcribe(
    audio: Path = typer.Argument(..., exists=True, dir_okay=False, help="MP3, WAV or FLAC file."),
    out: Path = typer.Option(None, "--out", help="Output directory (default: out/<name>)."),
    profile: str = typer.Option("quality", "--profile", help="quality or balanced."),
    strict: bool = typer.Option(False, "--strict", help="Remove low-confidence notes."),
    guitar_tuning: str = typer.Option(None, "--guitar-tuning", help="Tuning key, or auto."),
    bass_tuning: str = typer.Option(None, "--bass-tuning", help="Tuning key, or auto."),
    title: str = typer.Option(None, "--title", help="Score title (default: file name)."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Ignore cached stems."),
) -> None:
    """Transcribe a song into a full score and parts (.gp, .gp5, MIDI, MusicXML)."""
    from bandscribe.config import PipelineConfig
    from bandscribe.pipeline import run_pipeline

    _utf8_stdout()
    cfg = PipelineConfig.from_defaults(profile, strict=strict, guitar_tuning=guitar_tuning,
                                       bass_tuning=bass_tuning, title=title,
                                       reuse_cache=not no_cache)
    state = {"stage": None, "tenth": -1}

    def progress(stage: str, fraction: float, message: str) -> None:
        tenth = int(fraction * 10)
        if stage != state["stage"] or tenth != state["tenth"]:
            state.update(stage=stage, tenth=tenth)
            typer.echo(f"[{stage:10s}] {fraction:4.0%} {message}")

    result = run_pipeline(audio, cfg, out, progress)
    manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
    typer.echo("")
    for w in manifest["warnings"]:
        typer.echo(f"warning: {w}")
    typer.echo(f"tracks: {manifest['summary']['tracks']}")
    typer.echo(f"low-confidence notes: {manifest['summary']['low_confidence']}")
    for fmt, rep in manifest["exports"].items():
        typer.echo(f"{fmt}: {rep.get('path') or 'FAILED: ' + rep.get('error', '')}")
    typer.echo(f"output: {result}")


@app.command()
def gui(
    port: int = typer.Option(8765, "--port", help="Port on 127.0.0.1."),
    browser: bool = typer.Option(False, "--browser", help="Open in the browser, not a window."),
    no_window: bool = typer.Option(False, "--no-window", help="Only start the server."),
) -> None:
    """Open the graphical interface (local server on 127.0.0.1 + desktop window)."""
    from bandscribe.gui.server import HOST, serve

    typer.echo(f"BandScribe GUI: http://{HOST}:{port}")
    serve(port, open_window=not no_window, browser=browser)


@app.command()
def tunings() -> None:
    """List the tuning keys accepted by --guitar-tuning / --bass-tuning."""
    from bandscribe.tab.tunings import TUNINGS

    for t in TUNINGS.values():
        typer.echo(f"{t.key:20s} {t.instrument:7s} {t.name}")


@models_app.command("list")
def models_list() -> None:
    """List registered models with status, license and measured VRAM."""
    from bandscribe.models.registry import load_registry

    _utf8_stdout()
    reg = load_registry()
    rows = [("id", "task", "status", "weights license", "VRAM GB", "sha256")]
    for m in reg.models.values():
        pinned = bool(m.files) and all(f.sha256 for f in m.files)
        rows.append((m.id, m.task, m.status, m.license_weights or "unconfirmed",
                     "not measured" if m.vram_measured_gb is None else f"{m.vram_measured_gb:.2f}",
                     "pinned" if pinned else "not pinned"))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for r in rows:
        typer.echo("  ".join(cell.ljust(w) for cell, w in zip(r, widths, strict=False)).rstrip())


if __name__ == "__main__":
    app()

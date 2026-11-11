"""BandScribe command line. The GUI calls the same pipeline functions (spec 12.4)."""

from __future__ import annotations

import json
import sys

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
    """Check OS, GPU/CUDA/VRAM, ffmpeg, Node/alphaTab, MuseScore and the model registry."""
    from bandscribe import env_check

    _utf8_stdout()
    checks = env_check.run_checks()
    if as_json:
        typer.echo(json.dumps(env_check.build_report(checks), indent=2, ensure_ascii=False))
    else:
        typer.echo(env_check.format_report(checks))
    raise typer.Exit(1 if env_check.overall_status(checks) == "fail" else 0)


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
        typer.echo("  ".join(cell.ljust(w) for cell, w in zip(r, widths)).rstrip())


if __name__ == "__main__":
    app()

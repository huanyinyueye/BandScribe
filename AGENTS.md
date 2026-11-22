# BandScribe repo rules

`CLAUDE.md` is linked to this file (hard link now; a symlink once Windows Developer Mode is on). Edit `AGENTS.md` in place.

## Inference runtime: audio.cpp first

- New or replacement models (separation, transcription, beats, anything on the hot path) run through audio.cpp by default: add or patch the model in audio.cpp, call `audiocpp_cli`, and keep torch out of the default flow.
- audio.cpp changes live as patches in `native/audiocpp/` against the pinned commit; `native/audiocpp/build.ps1` rebuilds the CLI and stages `.cache/audiocpp-runtime/`. Never depend on a checkout in `%TEMP%`.
- A model the audio.cpp version cannot reproduce falls back to ONNX Runtime; torch stays an optional advanced backend, never a required dependency of the installer.
- Accept a port only after comparing it with the torch version on a whole song (Neo-Aspect): per-stem SI-SDR for separation, note F1 against `out/reference/neo.json` for transcription (guitar 30–90 s must stay ≥ 0.655), beat and bar positions for beat tracking. Record the numbers in `docs/e2e/<date>-<slug>/result.md`.
- Weights are never bundled: they download from Hugging Face (or the registry URL) on first launch and are converted locally without torch.
- The MuScriptor + Basic Pitch cross-model consensus stays as is when backends change.

## Where the engines plug in

- `bandscribe/runtime/audiocpp.py` finds the runtime (`BANDSCRIBE_AUDIOCPP`, `<python dir>/audiocpp`, then `.cache/audiocpp-runtime`), lists devices and runs the CLI. The CLI prints full results on stdout: always send stdout to a file.
- `BANDSCRIBE_ENGINE=auto|audiocpp|torch` picks the engine; `auto` uses audio.cpp when the runtime exists. `get_separator` / `get_transcriber` / `BeatTracker` dispatch on it.
- First-run conversion lives in `bandscribe/models/prepare.py` (checkpoint → safetensors, Beat This! ONNX weights, MuScriptor HF snapshot). Shipped static files: `bandscribe/models/audiocpp/*.json`, `bandscribe/models/onnx/*`.
- The default install (`uv sync`) has no torch; `uv sync --extra torch` adds the advanced engine. Run the no-torch E2E in `.venv-notorch` (`UV_PROJECT_ENVIRONMENT=.venv-notorch uv sync --no-dev`).
- `scripts/package_app.ps1` stages the app (embeddable Python + `.venv-notorch` packages, `python/audiocpp/`, `app/`, `BandScribe.exe`) and reports sizes; keep the 7z archive under 1 GB.

## Verify

- Python: `.venv\Scripts\python -m pytest -q` and `.venv\Scripts\python -m ruff check bandscribe`.
- GUI: `npm run build` in `gui/`.

# BandScribe

Local, quality-first transcription of band music (J-Rock, Metal, Prog): MP3 / WAV / FLAC in,
full score and per-instrument parts out (Guitar Pro 8 `.gp`, `.gp5`, MIDI, MusicXML).

- Spec: `BANDSCRIBE_SPEC.md`
- Progress and what is not verified: `PROGRESS.md`
- Measurements: `BENCHMARKS.md`
- Programmatic interfaces and the output directory layout: `docs/INTERFACES.md`

**Personal, non-commercial use only.** The default pipeline uses weights under CC BY-NC 4.0
(MuScriptor) and CC BY-NC-SA 4.0 (ADTOF); output must not be used commercially. Per-model
licences: `bandscribe models list`, or the About page of the GUI.

## Setup (Windows 11, NVIDIA GPU)

uv is installed as a Python module here, so commands start with `py -m uv`.

    py -m uv sync
    cd exporter; npm install; cd ..
    cd gui; npm install; npm run build; cd ..
    py -m uv run bandscribe doctor

MuScriptor weights are gated on Hugging Face, one repository per model size. Accept the
licence on the model page (`huggingface.co/MuScriptor/muscriptor-medium`, and `-small` /
`-large` if you want those sizes), then run `uvx hf auth login`. Without access the pipeline
skips MuScriptor, says so, and uses Basic Pitch and ADTOF alone.

## Use

Graphical interface (local server on 127.0.0.1 plus a desktop window):

    py -m uv run bandscribe gui            # add --browser to use the browser instead

Command line:

    py -m uv run bandscribe transcribe song.flac --out out/song
    py -m uv run bandscribe transcribe song.mp3 --profile balanced --guitar-tuning guitar_drop_d
    py -m uv run bandscribe tunings        # tuning keys
    py -m uv run bandscribe models list

The result is in `out/song/exports/` (`.gp` is the main file; open it in Guitar Pro 8).
Notes the pipeline is unsure about carry a `?` above the beat. `manifest.json` lists every
model with its version and sha256, peak VRAM, timings and warnings.

## Tests

    py -m uv run pytest                    # everything, including GPU and slow tests
    py -m uv run pytest -m "not slow"      # quick run
    py -m uv run python scripts/smoke_adapters.py   # every model adapter once on a short clip

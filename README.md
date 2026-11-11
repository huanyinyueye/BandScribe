# BandScribe

Local, quality-first transcription of band music (J-Rock, Metal, Prog): MP3/WAV/FLAC in,
full score and per-instrument parts out (Guitar Pro 8 `.gp`, `.gp5`, MIDI, MusicXML, PDF).
Spec: `BANDSCRIBE_SPEC.md`. Progress: `PROGRESS.md`. Measurements: `BENCHMARKS.md`.

Personal, non-commercial use only: the default pipeline uses weights under CC BY-NC 4.0
(MuScriptor) and CC BY-NC-SA 4.0 (ADTOF). Output must not be used commercially.
Per-model licenses: `bandscribe models list`.

## Setup (Windows 11, NVIDIA GPU)

    uv sync
    cd exporter && npm install && cd ..
    uv run bandscribe doctor
    uv run pytest

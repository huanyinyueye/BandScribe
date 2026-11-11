# BandScribe benchmarks

All numbers here are measured on this machine. Upstream self-reported numbers live in
`BANDSCRIBE_SPEC.md` and are not copied here. A MUST-BENCH item may become a default only after
its row below is filled in.

## Machine (from `bandscribe doctor`, 2026-11-12)

| Item | Value |
|---|---|
| OS | Windows 11 Home, build 10.0.26300 |
| GPU | NVIDIA GeForce RTX 3080, 10.0 GB VRAM, driver 610.88 |
| PyTorch | 2.11.0+cu128 (CUDA 12.8), FP16 smoke test OK |
| Python | 3.11.18 (uv 0.13.7) |
| ffmpeg | 9.0.2 essentials (bundled via imageio-ffmpeg; none on PATH) |
| Node / npm | v22.14.0 / 10.9.2 |
| alphaTab | 1.9.0, MPL-2.0 (license confirmed from the npm package), `Gp7Exporter` present |
| MuseScore 4 | not installed |

## M0: VRAM manager (real GPU tests in `tests/test_vram.py`)

| Check | Result |
|---|---|
| Model released after `ModelSlot.load` block (FP16 Linear 4096x4096) | 0 bytes left allocated |
| CUDA OOM detected and retried with a smaller chunk | pass (1 OOM, then success on CUDA) |

## Per-model peak VRAM (MUST-BENCH, spec 12.2)

| Model | Precision / chunk | Peak VRAM (GB) | s per audio minute | Status |
|---|---|---|---|---|
| BS-RoFormer SW | | | | not measured (M2) |
| SCNet XL IHF | | | | not measured (M2) |
| HTDemucs 6s | | | | not measured (M2) |
| SubRoformer | | | | not measured (M3) |
| MVSep Mega 53 | | | | not measured (M3) |
| MuScriptor medium | | | | not measured (M4) |
| MuScriptor large | | | | not measured (M4) |
| Basic Pitch | | | | not measured (M4) |
| ADTOF | | | | not measured (M4) |
| Beat This! | | | | not measured (M1) |

## Quality benchmarks (spec section 6)

| Area | Dataset | Metric | Result |
|---|---|---|---|
| Beat | GTZAN subset | beat F / downbeat F | not measured (M1) |
| Separation | MUSDB18-HQ test, MoisesDB | SDR per model per stem | not measured (M2/M3) |
| Guitar transcription | GuitarSet | note F1 | not measured (M4) |
| TAB | GuitarSet | string/fret accuracy | not measured (M5) |
| Drums | ENST / STAR / ADTOF test | per-class onset F1 | not measured (M4) |
| End to end | 3-5 user songs | manual review | not measured (M7) |

# BandScribe progress

Division of work (user instruction, 2026-11-12):
- **Framework** (implementation session): everything in `bandscribe/` except `bandscribe/qa/`,
  all model adapters, ensemble / consensus, attribution, TAB, notation, exporters, pipeline,
  CLI, GUI, `exporter/`, `gui/`, `scripts/`, `docs/INTERFACES.md`.
- **Research / QA** (Opus session): `benchmarks/`, `BENCHMARKS.md`, `tests/e2e/`,
  `bandscribe/qa/`, `bandscribe/models/defaults.yaml`, `QA_PLAN.md`: model comparison, SDR / F1
  tables, choice of defaults, QA rules, end-to-end review.

| M | Framework part | Status | Research / QA part |
|---|---|---|---|
| M0 | env check, skeleton, registry, tests, VRAM manager | done | |
| M1 | audio loading, Beat This!, tempo map, meter | done (GTZAN beat F 88.72 / downbeat 78.09) | |
| M2 | adapters: BS-RoFormer SW, SCNet XL IHF, HTDemucs 6s; ensemble + confidence | done, smoke-tested on the RTX 3080 | MUSDB18-HQ SDR table; "ensemble not worse than best single model" |
| M3 | adapters: SubRoformer, Mega 53 | done, smoke-tested | comparison and choice of defaults |
| M4 | attribution, MuScriptor / Basic Pitch / ADTOF adapters, consensus | done; MuScriptor small / large not run (see below) | GuitarSet note F1; drum F1 |
| M5 | TAB DP, tuning suggestion, drum mapping and voting | done | GuitarSet TAB accuracy |
| M6 | quantisation, score IR, `.gp` / `.gp5` / MIDI / MusicXML | done; `.gp` verified by alphaTab read-back only | opening the `.gp` in Guitar Pro 8 (user) |
| M7 | pipeline, manifest, QA hook, `bandscribe transcribe` | done | QA rules (spec section 5), report, end-to-end on real songs |
| M8 | GUI: FastAPI backend, React + alphaTab frontend, pywebview window | done in the browser; desktop window not checked (see "GUI state") | |

Interfaces for the research / QA session: `docs/INTERFACES.md`.

## What was verified, and how

- Tests: `py -m uv run pytest` -> 200 passed on 2026-11-12, including GPU tests (marked `gpu`,
  long ones `slow`).
- Adapter smoke test on a 30 s metal clip (`scripts/smoke_adapters.py` ->
  `docs/adapter_smoke.json`). This shows each adapter runs on the RTX 3080; it says nothing
  about transcription or separation quality.

| Adapter | Device | Peak VRAM | Time for 30 s | OOM retries |
|---|---|---|---|---|
| Beat This! | cuda | 0.15 GB | 4.0 s (incl. load) | 0 |
| BS-RoFormer SW | cuda, batch 2 | 3.97 GB | 4.7 s | 0 |
| SCNet XL IHF | cuda, batch 2 | 2.38 GB | 9.7 s | 0 |
| HTDemucs 6s | cuda | 0.61 GB | 1.3 s | 0 |
| SubRoformer | cuda, batch 2 | 1.64 GB | 9.6 s | 0 |
| Mega 53, config chunk 882000 | cuda | 10.25 GB (spills into shared memory) | 244 s | 0 |
| Mega 53, chunk 441000 (adapter default) | cuda | 7.04 GB | 15.6 s | 0 |
| Mega 53, chunk 220500 | cuda | 5.44 GB | 15.2 s | 0 |
| MuScriptor medium | cuda | 2.29 GB | 14.1 s (incl. load) | 0 |
| MuScriptor small / large | not run | | | |
| Basic Pitch | CPU (ONNX) | 0 | 1.7 s | 0 |
| ADTOF | CPU (ONNX) | 0 | 0.6 s | 0 |

- End to end on the same clip with the default `quality` profile: all stages run, `.gp`
  read back by alphaTab matches the score (1526 note heads), `.gp5`, MIDI and MusicXML written.

## GUI state

Checked in a browser against the real backend with the 30 s metal clip (upload through the
API, a full `quality` run, then a re-run that reused cached stems):
- Import (waveform, player, recent projects), Processing (per-stage bars, ETA, live VRAM,
  cancel), Stems (solo / mute, confidence heat strips), Score (full score and single tracks
  with TAB, low-confidence notes in orange), Export (downloads, read-back result) all rendered
  and showed the run's data; no console errors.
- Not exercised by hand: drag-and-drop upload, the Settings form, the "not this instrument"
  re-run, click-a-beat-to-seek, alphaTab playback, the About page, "open output folder".
  The backend side of upload, config validation, overrides, cancel and file access is covered
  by `tests/test_m8_gui_server.py` with a stand-in pipeline.
- `bandscribe gui` with the pywebview desktop window has not been opened; only
  `--no-window` (server + browser) was run. pywebview imports correctly.
- The interface text is Traditional Chinese.

## Not done or not verified (framework side)

- **MuScriptor small and large have never run here.** Each Hugging Face repo is gated
  separately; this machine's account has access to `muscriptor-medium` only. The adapters
  raise `TranscriberUnavailable` for them. To enable: accept the licence at
  `huggingface.co/MuScriptor/muscriptor-small` and `.../muscriptor-large`.
- **No `.gp` file has been opened in Guitar Pro 8.** Only the alphaTab read-back check ran.
- Guitar is one track, "Guitar (combined)". The stereo / register split into two guitars
  described in spec 2.2 is not implemented.
- One voice per track; sustained notes overlapping the next onset are shortened in notation.
- MusicXML has no separate TAB staff (string / fret are technical marks). No PDF export.
- Key estimation and section detection are unbenchmarked heuristics.
- The Mega 53 chunk size was reduced to fit 10 GB; the effect on its output quality is unknown.
- On the 30 s metal clip MuScriptor and Basic Pitch agreed on only about 22-24 % of guitar /
  bass notes (no systematic time offset; many octave disagreements), so most notes were marked
  low-confidence. Whether the consensus thresholds or the model pairing should change is a
  model-comparison question for the research / QA session.

## Decisions made on the user's behalf (2026-11-12)

- Downloads need no per-file confirmation (user instruction).
- MuseScore is not used; Guitar Pro is the only notation target. PDF export is dropped
  (Guitar Pro can print / export PDF from the `.gp` file).
- alphaTex is not used for `.gp` export: alphaTab 1.9.0 and 1.8.1 rejected every
  `\staff {...}` property here. `exporter/export_gp.mjs` builds alphaTab's model directly.
- Basic Pitch runs through its bundled ONNX model; TensorFlow is excluded because its pinned
  version conflicts with MuScriptor's numpy requirement.
- MSST model code (SCNet, SubRoformer, Mega 53) is a git checkout pinned to commit `1d4b3c43`
  under `~/.cache/bandscribe/vendor/msst`, not a pip dependency.

## Environment notes

- Run with `py -m uv run ...` (uv is not on PATH). Python 3.11.18, torch 2.11.0+cu128.
- Datasets on disk: GTZAN, MUSDB18-HQ test set (`~/.cache/bandscribe/datasets/`).
- Weights: `~/.cache/bandscribe/models/` (sha256 pinned in `bandscribe/models/registry.yaml`);
  MuScriptor in the Hugging Face cache; hash of medium pinned in the registry.

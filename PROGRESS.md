# BandScribe progress

| M | Scope | Status |
|---|---|---|
| M0 | env check, skeleton, registry, tests, VRAM manager | done 2026-11-12 |
| M1 | audio loading, Beat This!, tempo map, meter | done 2026-11-12 |
| M2 | separation: BS-RoFormer SW, SCNet XL IHF, HTDemucs 6s, ensemble | |
| M3 | SubRoformer, Mega 53 | |
| M4 | attribution, transcription, consensus | |
| M5 | TAB DP, tuning detection, drum mapping | |
| M6 | quantize, score IR, MIDI / MusicXML / .gp / .gp5 / PDF | |
| M7 | QA rules, report, end to end | |
| M8 | GUI | |

## M0 definition of done

- `bandscribe doctor` prints a full environment report: met (11 checks; `--json` available).
- Unit tests green: met locally (52 passed, including 2 real-GPU tests; ruff clean).
  There is no CI service: the repo has no remote, so "CI" means the local `pytest` run.

## Decisions made on the user's behalf (2026-11-12)

- Downloads need no per-file confirmation (user instruction).
- MuseScore is not used. Guitar Pro is the only notation target; the MuseScore check was
  removed from `doctor`, and PDF export through MuseScore is dropped (Guitar Pro can print /
  export PDF from the `.gp` file).

## M1 definition of done

- GTZAN beat F >= published - 3: met (88.72 vs 89.1; downbeat 78.09 vs 78.3), full dataset.

## M0 open items

- ffmpeg is only available through the bundled imageio-ffmpeg binary (no ffprobe).
- Registry: no sha256 is pinned and no weight file URL is resolved yet. Each is filled in
  when its model is downloaded in its milestone. Weight licenses still unconfirmed for
  BS-RoFormer SW, SCNet XL IHF, SubRoformer, Mega 53 and Beat This!.
- User decision 2026-11-12: downloads no longer need per-file confirmation.

## State when work paused (2026-11-12, usage limit reached)

Done and committed: M0, M1.

In progress, code written and unit-tested but milestone NOT complete:
- M2/M3: separators for BS-RoFormer SW, SCNet XL IHF, HTDemucs 6s, SubRoformer, Mega 53 all
  run on the RTX 3080 (30 s clip: peak VRAM 3.97 / 2.38 / 0.61 / 1.64 / 10.25 GB). All weights
  downloaded and sha256-verified. Ensemble + agreement confidence written.
  NOT done: the MUSDB18-HQ SDR benchmark (`benchmarks/bench_separation.py`, never run). The
  test set was still downloading (`benchmarks/fetch_musdb_test.py`, 137 of ~250 files); rerun
  that script to resume, it skips finished files. MoisesDB not obtained.
- M4/M5: Basic Pitch, ADTOF, MuScriptor backends, consensus, attribution (incl. other->guitar
  rule), tuning suggestion, TAB DP, drum voting written; 48 unit tests pass.
  NOT done: GuitarSet note-F1 and TAB accuracy benchmarks; MuScriptor is untested because its
  weights are gated (user must accept the license on Hugging Face and run `uvx hf auth login`).

Not started: quantization, score IR, exporters (M6), QA report and pipeline CLI (M7), GUI (M8).

Finding for M6: alphaTab 1.9.0 / 1.8.1 alphaTex rejected every `\staff {...}` property in
this environment, so the plan is to build the alphaTab model directly in Node.
`exporter/probe.mjs` is a working prototype: guitar TAB with tuning, tuplets, ties, beat text,
7/8 bar, tempo change, section, drums and a pitched staff all survive a
Gp7Exporter -> ScoreLoader round trip. Not yet opened in Guitar Pro 8.

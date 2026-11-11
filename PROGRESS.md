# BandScribe progress

| M | Scope | Status |
|---|---|---|
| M0 | env check, skeleton, registry, tests, VRAM manager | done 2026-11-12 |
| M1 | audio loading, Beat This!, tempo map, meter | next |
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

## M0 open items

- MuseScore 4 is not installed. PDF export and the MuseScore open-file test (M6) need it.
- ffmpeg is only available through the bundled imageio-ffmpeg binary (no ffprobe).
- Registry: no sha256 is pinned and no weight file URL is resolved yet. Each is filled in
  when its model is downloaded in its milestone. Weight licenses still unconfirmed for
  BS-RoFormer SW, SCNet XL IHF, SubRoformer, Mega 53 and Beat This!.
- User decision 2026-11-12: downloads no longer need per-file confirmation.

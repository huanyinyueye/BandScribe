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

## M0: VRAM manager (real GPU tests in `tests/test_vram.py`)

| Check | Result |
|---|---|
| Model released after `ModelSlot.load` block (FP16 Linear 4096x4096) | 0 bytes left allocated |
| CUDA OOM detected and retried with a smaller chunk | pass (1 OOM, then success on CUDA) |

## Per-model peak VRAM (MUST-BENCH, spec 12.2)

| Model | Precision / chunk | Peak VRAM (GB) | s per audio minute | Status |
|---|---|---|---|---|
| BS-RoFormer SW | FP16, chunk 588800, batch 2 | 3.97 | 5.0 (MUSDB18-HQ test, 50 tracks) | measured |
| SCNet XL IHF | FP16, chunk 485100, batch 2 | 2.38 | 10.7 (MUSDB18-HQ test) | measured |
| HTDemucs 6s | demucs default, whole-track split | 0.61 | 1.5 (MUSDB18-HQ test) | measured |
| SubRoformer | FP16, chunk 96000, batch 2 | 1.64 | 15.1 (MUSDB18-HQ test) | measured |
| MVSep Mega 53 | | | | not measured (M3) |
| MuScriptor medium | default | 2.29 | 20.5 (GuitarSet, incl. load; GPU shared part of the time) | measured |
| Basic Pitch | ONNX, CPU | 0 | 0.61 (GuitarSet) | measured |
| ADTOF | | | | not measured (M4) |
| Beat This! final0 | FP16, whole track | 0.154 | 0.40 (incl. decoding) | measured, GTZAN run |

## Quality benchmarks (spec section 6)

| Area | Dataset | Metric | Result |
|---|---|---|---|
| Beat | GTZAN, 998 tracks | beat F / downbeat F | 0.8872 / 0.7809 (see M1) |
| Separation | MUSDB18-HQ test, 50 tracks | global SDR, 4-stem avg | BS-RoFormer SW 11.37, SCNet XL IHF 10.09, SubRoformer 10.01, HTDemucs 6s 7.93 (guitar / piano: no dataset yet) |
| Guitar transcription | GuitarSet mic, 360 tracks | note F1 (onset) | consensus 0.850, MuScriptor 0.807, Basic Pitch 0.753 |
| TAB | GuitarSet, oracle pitches | string/fret accuracy | 67.3% tuned (60.1% code defaults), held-out players |
| Drums | ENST / STAR / ADTOF test | per-class onset F1 | not measured (M4) |
| End to end | 3-5 user songs | manual review | not measured (M7) |

## M1: beat tracking (2026-11-12)

`uv run python benchmarks/bench_beats.py` -> `benchmarks/results/beats_gtzan.json`.
Model: Beat This! `final0` (trained on everything except GTZAN), no DBN, FP16 on CUDA.
Data: GTZAN audio (HF `marsyas/gtzan`), annotations from `CPJKU/beat_this_annotations`.
Metric: `mir_eval.beat.f_measure`, 70 ms window, mean over tracks.

| Metric | Measured here | Published (paper Table 2, mean of 3 seeds) | Threshold (published - 3) | Met |
|---|---|---|---|---|
| Beat F | 88.72 (998 tracks) | 89.1 +/- 0.3 | 86.1 | yes |
| Downbeat F | 78.09 (992 tracks) | 78.3 +/- 0.4 | 75.3 | yes |

- `jazz.00054.wav` is skipped: the file in the GTZAN archive is corrupt (no audio stream).
- 6 tracks have beat annotations without downbeats and count only for the beat score.
- Weakest genres for beats: classical 0.653, blues 0.830, jazz 0.867; metal 0.872, rock 0.921.
- Tempo map and meter have no labelled benchmark here; they are covered by unit tests on
  synthetic beat grids only (constant tempo, tempo change, rubato, 3/4, 5/4, 7/8, mixed meter).

## Guitar transcription and TAB on GuitarSet (QA session, 2026-11-12)

`py -m uv run python benchmarks/bench_guitarset.py --audio mic` -> `results/guitarset_mic.json`.
Data: GuitarSet, all 360 excerpts, mono microphone audio (closer to a real recording than the
pickup mix). Metric: `mir_eval.transcription`, 50 ms onset tolerance; "+offset" adds offset
ratio 0.2. Mean over tracks. Input is clean solo guitar, so this is an upper bound for guitar
transcribed from a separated stem.

| System | P | R | F1 | F1 +offset | F1 comp | F1 solo |
|---|---|---|---|---|---|---|
| MuScriptor medium (guitar) | 0.772 | 0.865 | 0.807 | 0.466 | 0.742 | 0.872 |
| Basic Pitch (guitar) | 0.674 | 0.874 | 0.753 | 0.536 | 0.722 | 0.785 |
| Consensus, all notes kept | 0.587 | 0.925 | 0.711 | 0.419 | 0.663 | 0.760 |
| **Consensus, conf >= 0.5** (= agreed notes only) | **0.912** | 0.801 | **0.850** | 0.491 | 0.806 | 0.893 |

- The pipeline threshold `low_confidence: 0.5` separates exactly the agreed notes: notes below
  it are mostly false positives (precision 0.59 -> 0.91 when dropped). Keep 0.5; strict mode
  (drop) gives the best F1, default mode (mark) keeps recall for manual review.
- Mean agreement between the two backends: 0.558 of notes.
- Offsets are weak for every system (F1 +offset <= 0.54). Basic Pitch has better offsets than
  MuScriptor; using Basic Pitch offsets for agreed notes is a candidate improvement (not
  measured yet).
- Comping (strummed chords) is harder than solo lines for every system.
- MuScriptor medium: 360 jobs in 3746 s for 10968 s of audio (20.5 s per audio minute,
  including model load), peak 2.29 GB VRAM, no OOM. The implementation session used the GPU
  for part of this run, so the speed is an upper bound.

### TAB string/fret assignment (oracle pitches)

`benchmarks/tune_fret_dp.py` -> `results/fret_dp_tuning*.json` (three passes). The DP gets the
ground-truth pitches, so this measures fingering choice alone. Tuned on players 00-02, scored on
held-out players 03-05.

| FretOptions | Train (00-02) | Held-out (03-05) |
|---|---|---|
| code defaults (open_bonus 0.5, span 0.6, move 1.0, high_fret 0.05, string_jump 0.15) | 65.2% | 60.1% |
| pass 1 best (open_bonus 0) | 69.4% | 63.4% |
| **final** (open_bonus -0.75, span 1.0, move 0.5, high_fret 0.02, string_jump 0.15) | 71.2% | **67.3%** |

- The final weights are in `defaults.yaml` `tab.guitar_dp` and applied to guitar only (commit
  a1fbeba). Bass keeps the code defaults: no labelled bass data.
- About a third of notes still differ from the player's choice. Many differences are
  equally playable alternatives (same pitch at another position), so this is a lower bound on
  playability, not an error rate.
- With transcribed notes (pre-tuning weights): 62.2% of MuScriptor notes that matched a
  reference note got the reference string/fret.

## Source separation on MUSDB18-HQ (QA session, 2026-11-12)

`py -m uv run python benchmarks/bench_separation.py` -> `benchmarks/results/separation_musdb.json`
(per-track rows in `separation_musdb_tracks.jsonl`). 50 test tracks, 2 h 16 min wall time, no OOM
retries, all on CUDA.

Metric: global SDR per track, `10*log10(sum(s^2) / sum((s - s_hat)^2))`, averaged over tracks
(MSST "SDR avg", not museval's median of 1 s chunks). For 6-stem models, `other` is scored as
other + guitar + piano, because MUSDB has no guitar or piano reference.

| Model | vocals | drums | bass | other | avg |
|---|---|---|---|---|---|
| BS-RoFormer SW (6 stem) | 12.33 | **12.99** | **11.07** | **9.10** | **11.37** |
| SCNet XL IHF (4 stem) | 11.42 | 11.82 | 9.24 | 7.88 | 10.09 |
| SubRoformer (4 stem) | **12.50** | 11.92 | 8.05 | 7.55 | 10.01 |
| HTDemucs 6s (6 stem) | 8.86 | 9.53 | 7.89 | 5.46 | 7.93 |

Ensembles: `(1 - w) * BS-RoFormer SW + w * secondary`, scored from float16 copies of the estimates.

| Secondary | w | vocals | drums | bass |
|---|---|---|---|---|
| SCNet XL IHF | 0.25 | 12.45 | 13.06 | 11.07 |
| SCNet XL IHF | 0.50 | 12.31 | 12.85 | 10.67 |
| SCNet XL IHF | 0.75 | 11.95 | 12.41 | 10.03 |
| SubRoformer | 0.25 | 12.68 | **13.11** | 10.88 |
| SubRoformer | 0.50 | **12.83** | 12.94 | 10.13 |
| SubRoformer | 0.75 | 12.77 | 12.54 | 9.14 |

Decision (QA_PLAN rule: per stem, blend only where the ensemble beats the best single model):

- The primary stays BS-RoFormer SW. It is the best single model on drums, bass and other, and it is
  the only model here with guitar and piano stems.
- The secondary changes from SCNet XL IHF to SubRoformer, with weights vocals 0.5 (+0.33 dB over
  the best single model), drums 0.25 (+0.12 dB) and bass 0.
  - With bass at 0, the bass audio comes from BS-RoFormer SW only. Every mix tested made bass
    worse, by up to 1.9 dB. The secondary's bass is still used for the agreement confidence.
- SubRoformer's licence is still unconfirmed in the registry. That is acceptable for personal use.
- In the `balanced` profile (secondary off), the run costs 5.0 s per audio minute instead of
  about 20.

Caveats:

- These upstream checkpoints may have seen MUSDB18-HQ test tracks in training (their training sets
  are not fully documented), so the absolute numbers may be optimistic. The ranking is what the
  defaults rely on.
- The other three models are about 2 dB above HTDemucs 6s. That fits the published ordering.
- SubRoformer: the paper reports 11.59 avg; we measured 10.01, which is closer to MSST's ~10.4.
- Guitar and piano stem quality is not measured. MUSDB has no such references; MoisesDB or Slakh
  is still to do.


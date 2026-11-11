# BandScribe stable interfaces

Owner: implementation session (framework and model adapters). Consumers: the research / QA
session (model comparison, QA rules, end-to-end tests) and the GUI. Anything not listed here is
internal and may change. This file is updated in the same commit as any change to these
interfaces.

Run everything through the project environment: `py -m uv run ...` (uv is not on PATH).

## 1. Adapter factories

### Separators

```python
from bandscribe.separation import get_separator
sep = get_separator(model_id, **opts)          # opts: batch_size=int, chunk_size=int
res = sep.separate(mix, sr=44100, progress=None)   # mix: float32 ndarray, shape (2, n)
```

| model_id | stems | notes |
|---|---|---|
| `bs_roformer_sw` | bass, drums, other, vocals, guitar, piano | default primary |
| `scnet_xl_ihf` | drums, bass, other, vocals | default secondary |
| `htdemucs_6s` | drums, bass, other, vocals, guitar, piano | baseline; opts: `overlap`, `shifts` |
| `subroformer` | vocals, bass, drums, other | candidate |
| `mvsep_mega53` | 53 stems | `mixture_consistent = False`; needs a smaller `chunk_size` on 10 GB |

`SeparationResult` (`bandscribe.separation.base`):

| field | meaning |
|---|---|
| `stems: dict[str, Stem]` | `Stem(name, audio (2, n) float32, sr, model_id)`, `Stem.rms()` |
| `model_id`, `files_sha256` | weights actually loaded (name -> sha256, verified against the registry pin) |
| `device` | `cuda` or `cpu` (CPU only after every CUDA attempt hit OOM) |
| `batch_size`, `chunk_size` | the values that succeeded |
| `peak_vram_gb` | `torch.cuda.max_memory_allocated` of the successful attempt |
| `seconds` | inference only; model loading excluded |
| `oom_retries`, `notes` | number of CUDA OOM retries and human-readable consequences |

Input must be 44.1 kHz stereo. `sep.stem_names` and `sep.mixture_consistent` are available
before running. One model is on the GPU at a time (`bandscribe.runtime.vram.gpu_slot`); a
second `separate` while one is running raises `SlotBusyError`.

Ensemble: `bandscribe.separation.ensemble.ensemble(primary, secondary=None, weights=None)`
returns `EnsembleResult(stems, confidence: dict[str, StemConfidence], weights, results)`.
`weights[stem]` is the secondary model's weight. Only vocals / drums / bass are ever blended;
guitar, piano and other always come from the primary model.

### Transcribers

```python
from bandscribe.transcription import get_transcriber, TRANSCRIBER_IDS
tr = get_transcriber(model_id, **opts)
notes = tr.transcribe(audio, sr, instrument)   # audio: mono or (2, n); returns list[Note]
tr.last_run       # {"device", "seconds", "peak_vram_gb", "oom_retries", "jobs", ...}
tr.model_sha256   # hash of the weights that were loaded
```

| model_id | instruments | notes |
|---|---|---|
| `muscriptor_small` / `_medium` / `_large` | vocals, bass, guitar, piano, other, drums, `mix` | gated on Hugging Face per repo; raises `TranscriberUnavailable` with instructions when the account has no access. `transcribe_many([(audio, sr, instrument), ...])` runs several jobs with one model load. `instrument="mix"` is unconditioned and labels each note with the family it detected. No per-note confidence (conf = 1.0). |
| `basic_pitch` | vocals, bass, guitar, piano, other | ONNX on CPU; conf = note amplitude. opts: `onset_threshold`, `frame_threshold`, `minimum_note_ms` |
| `adtof` | drums | ONNX on CPU; 5 classes (kick 36, snare 38, tom 47, hi-hat 42, cymbal 49); conf = activation. `tr.activations(audio, sr)` returns the (frames, 5) matrix at 100 fps |

`Note` (`bandscribe.transcription.base`): `onset`, `offset` (s), `pitch` (MIDI; GM percussion key
for drums), `velocity`, `conf` in [0, 1], `sources: tuple[str, ...]`, `instrument` (one of
vocals / drums / bass / guitar / piano / other), `string` (1 = highest), `fret`, `flags`.

Voting: `bandscribe.transcription.consensus.consensus(primary, secondary, tolerance=0.05,
drop_unconfirmed=False) -> (notes, ConsensusStats)`; drums:
`bandscribe.transcription.drums.vote_drums(detailed, coarse, tolerance=0.03)`.
Confidence bands: agreed notes 0.6-1.0; primary-only <= 0.45; secondary-only <= 0.35; a single
backend with no second opinion: its own confidence x 0.8.

### Beats

`bandscribe.analysis.beats.track_beats(audio_or_mono, sr=None) -> BeatResult(beats, downbeats,
model_id, model_sha256, device)`; `BeatTracker()` as a context manager keeps the model loaded
for a batch.

### Registry and defaults

- `bandscribe.models.registry.load_registry()` -> models with `source_urls`, `version`, `files`
  (url, sha256, size), `license_code`, `license_weights`, `non_commercial`, `gated`, `status`.
- `bandscribe.models.download.ensure_model(model_id)` downloads into
  `~/.cache/bandscribe/models/<id>/` and verifies sha256.
- `bandscribe/models/defaults.yaml` (owned by the research / QA session) is read by
  `bandscribe.config.load_defaults(profile)` and `PipelineConfig.from_defaults(profile,
  **overrides)`. Keys: `separation.primary|secondary|blend_weights`,
  `transcription.pitched|drums|mix_vote`, `beats`, `thresholds.low_confidence|onset_tolerance|
  drum_onset_tolerance`, `tab.max_span|guitar_tuning|bass_tuning`, `instruments`,
  `profiles.<name>` (deep-merged overrides). `schema_version` must stay 1.

## 2. Pipeline

```python
from bandscribe.config import PipelineConfig
from bandscribe.pipeline import run_pipeline
cfg = PipelineConfig.from_defaults("quality", strict=False, title="Song")
out_dir = run_pipeline(audio_path, cfg, out_dir=None, progress=None, cancel=None)
```

CLI equivalent: `bandscribe transcribe song.flac --out out/song [--profile balanced]
[--strict] [--guitar-tuning guitar_drop_d] [--no-cache]`.

`progress(stage, fraction, message)`; stages in order: load, beats, separate, attribute,
transcribe, tab, notation, qa, export. Raises `PipelineCancelled`, `QABlocked`, or the original
exception; `manifest.json` is written in every case with `status` done / cancelled / failed.

Output directory:

```
manifest.json
analysis.json
stems/<stem>.wav            44.1 kHz stereo PCM-24
stems/stems.json
notes/raw_<model>_<family>.json
notes/mix_<model>.json
notes/consensus_<family>.json
notes/final_<family>.json
notes/attribution.json
score.json
qa.json
exports/<title>.gp  .gp5  .mid  .musicxml
```

- Note JSON files are lists of `{onset, offset, pitch, velocity, conf, sources, instrument,
  string, fret, flags}`. `bandscribe.pipeline.notes_from_json` / `notes_to_json` convert.
- `stems.json`: `key` (cache key), `stems.<name>.{model_id, weights, confidence{window, values,
  active, source}, activity{present, rms_db, relative_db, active_seconds, active_fraction}}`.
- `analysis.json`: `beats`, `downbeats`, `global_bpm`, `tempo_segments`, `meter`, `bars`, `key`,
  `sections`.
- `score.json`: `Score.to_dict()` (`bandscribe.notation.score_model`); load with
  `Score.load(path)`. Positions / durations are quarter-note fractions stored as `"n/d"`.
- `manifest.json`: `bandscribe_version`, `audio{path, sha256, duration, ...}`, `config`,
  `models.<id>{name, version, license_code, license_weights, non_commercial, source_urls,
  sha256 | files_sha256}`, `stages.<stage>` seconds, `separation.<id>{device, batch_size,
  chunk_size, peak_vram_gb, seconds, oom_retries, notes}`, `transcription.<id>{seconds,
  last_run, notes}`, `consensus.<family>`, `unavailable.<id>` reason, `vram[]` (one record per
  model load: model_id, device, peak_bytes, peak_gb, leaked_bytes, seconds), `warnings[]`,
  `qa`, `exports.<fmt>{path, verified, mismatches, ...}`, `summary`, `status`, `error`.
- Flags a note can carry: `single-source`, `no-second-opinion`, `separation-disagreement`,
  `mix-confirmed`, `mix-disagrees:<family>`, `from-other-stem`, `unplayable`, `limb-overload`,
  `mus:<muscriptor instrument group>`.

Stems are cached per output directory (same audio hash and separation settings); pass
`reuse_cache=False` or `--no-cache` to recompute.

Exporters can be called directly on a `Score`:
`bandscribe.notation.export_gp.export_gp(score, path) -> GpExportReport(verified, mismatches,
summary, ...)`, `read_gp(path)`, `export_gp5.export_gp5`, `export_midi.export_midi`,
`export_musicxml.export_musicxml`. Layout shared by all of them:
`bandscribe.notation.layout.layout_track(track, bars)`.

## 3. QA hook

Before export the pipeline calls `bandscribe.qa.run_checks(score, context) -> dict`.
The default in `bandscribe/qa/__init__.py` checks nothing; the package belongs to the QA owner.

`context` keys: `config` (PipelineConfig), `out_dir` (Path), `notes` (dict family ->
list[Note], final), `activity` (dict stem -> StemActivity), `confidence` (dict stem ->
StemConfidence), `attribution` (AttributionLog with `.removed`, `.promoted`), `manifest` (dict so
far).

The returned dict is written to `qa.json` and stored under `manifest["qa"]`, so it must be
JSON-serialisable, except for two optional keys the pipeline interprets and removes:
- `"score"`: a `Score` that replaces the input for export (it is re-saved to `score.json`).
- `"block": True` with `"reason"`: stop before export (`QABlocked`).

## 4. What the framework already enforces (so QA can test it, not re-implement it)

- `other` never becomes an instrument stem; a note from the `other` stem is re-labelled only by
  `attribution.instrument_id.promote_from_other` (needs a matching mix-transcription note and an
  active `other` stem; capped at conf 0.5, flag `from-other-stem`).
- No note is created by consensus, quantisation or layout; empty time is rests.
- Range filter per family, removal of notes where the stem is silent, cross-stem leakage
  removal, and TAB-unplayable notes are logged in `notes/attribution.json` with a reason.
- Strict mode removes notes below `low_confidence`; otherwise they are kept, `low_conf` is set
  on the score note and the beat carries the text `?` in the `.gp` / MusicXML.

Not implemented in the framework (left to QA rules): bass polyphony flagging, the per-instrument
low-confidence ratio report, `report.html`.

## 5. Known limits

- Guitar is one track, "Guitar (combined)": no lead / rhythm or left / right split.
- One voice per track: a chord lasts until the next onset; overlapping sustain is shortened.
- MusicXML has one staff per part with string / fret as technical marks, no separate TAB staff.
- `.gp5`: tracks with more than 7 strings are skipped; staff tracks use a synthetic tuning.
- Key and sections are unbenchmarked heuristics; section names are "Section N".

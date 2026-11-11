"""End-to-end pipeline shared by the CLI and the GUI (spec 4.1, 12.4).

    run_pipeline(audio_path, config) -> output directory

Output directory layout (stable, see docs/INTERFACES.md):
    manifest.json            models, versions, sha256, VRAM, timings, config, export reports
    analysis.json            beats, downbeats, tempo segments, bars, key, sections
    stems/<name>.wav         separated stems (44.1 kHz, 24-bit)
    stems/stems.json         per-stem activity and confidence windows
    notes/raw_<model>_<family>.json     each backend's notes
    notes/mix_<model>.json              unconditioned transcription of the mix
    notes/consensus_<family>.json       after voting
    notes/final_<family>.json           after attribution rules and TAB assignment
    notes/attribution.json              removed / promoted notes with reasons
    score.json               score IR
    qa.json                  result of bandscribe.qa.run_checks
    exports/<title>.gp .gp5 .mid .musicxml
"""

from __future__ import annotations

import json
import re
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from bandscribe import __version__
from bandscribe.config import PipelineConfig
from bandscribe.transcription.base import Note, TranscriberUnavailable

ProgressFn = Callable[[str, float, str], None]  # (stage, fraction 0..1 within stage, message)
STAGES = ("load", "beats", "separate", "attribute", "transcribe", "tab", "notation", "qa",
          "export")
TRACK_DEFS = {  # family -> (track name, kind, General MIDI program)
    "guitar": ("Guitar (combined)", "tab", 30),
    "bass": ("Bass", "tab", 33),
    "drums": ("Drums", "drums", 0),
    "vocals": ("Vocals", "staff", 52),
    "piano": ("Piano", "staff", 0),
    "other": ("Other (unidentified)", "staff", 80),
}
TRACK_ORDER = ("vocals", "guitar", "piano", "other", "bass", "drums")


class PipelineCancelled(RuntimeError):
    pass


class QABlocked(RuntimeError):
    pass


def notes_to_json(notes: list[Note]) -> list[dict[str, Any]]:
    return [{"onset": round(n.onset, 4), "offset": round(n.offset, 4), "pitch": n.pitch,
             "velocity": n.velocity, "conf": round(n.conf, 4), "sources": list(n.sources),
             "instrument": n.instrument, "string": n.string, "fret": n.fret,
             "flags": list(n.flags)} for n in notes]


def notes_from_json(data: list[dict[str, Any]]) -> list[Note]:
    return [Note(d["onset"], d["offset"], d["pitch"], d["velocity"], d["conf"],
                 tuple(d["sources"]), d["instrument"], d.get("string"), d.get("fret"),
                 tuple(d.get("flags", ()))) for d in data]


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")


def _safe_name(text: str) -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", text).strip(" .") or "song"


def _model_info(model_id: str) -> dict[str, Any]:
    from bandscribe.models.registry import load_registry

    base = model_id
    try:
        m = load_registry().get(base)
    except KeyError:
        return {"id": model_id}
    return {"id": model_id, "name": m.name, "version": m.version,
            "license_code": m.license_code, "license_weights": m.license_weights,
            "non_commercial": m.non_commercial, "source_urls": list(m.source_urls)}


class _Run:
    """State of one pipeline run."""

    def __init__(self, audio_path: Path, config: PipelineConfig, out_dir: Path,
                 progress: ProgressFn | None, cancel: Callable[[], bool] | None) -> None:
        self.audio_path, self.config, self.out = audio_path, config, out_dir
        self._progress, self._cancel = progress, cancel
        self.manifest: dict[str, Any] = {
            "bandscribe_version": __version__, "audio": {}, "config": config.to_dict(),
            "models": {}, "stages": {}, "separation": {}, "transcription": {},
            "unavailable": {}, "warnings": [], "exports": {}, "status": "running"}
        self._stage_started = 0.0
        self._stage = ""

    def stage(self, name: str, message: str = "") -> None:
        self.check_cancel()
        now = time.perf_counter()
        if self._stage:
            self.manifest["stages"][self._stage] = round(now - self._stage_started, 2)
        self._stage, self._stage_started = name, now
        self.report(0.0, message)

    def finish_stage(self) -> None:
        if self._stage:
            self.manifest["stages"][self._stage] = round(time.perf_counter() - self._stage_started, 2)
            self._stage = ""

    def report(self, fraction: float, message: str = "") -> None:
        if self._progress is not None:
            self._progress(self._stage, float(min(1.0, max(0.0, fraction))), message)

    def check_cancel(self) -> None:
        if self._cancel is not None and self._cancel():
            raise PipelineCancelled("cancelled by the user")

    def warn(self, text: str) -> None:
        self.manifest["warnings"].append(text)

    def model(self, model_id: str, **extra: Any) -> None:
        self.manifest["models"].setdefault(model_id, _model_info(model_id)).update(extra)

    def save_manifest(self) -> None:
        from bandscribe.runtime.vram import gpu_slot

        self.manifest["vram"] = [r.as_dict() for r in gpu_slot.records]
        _write_json(self.out / "manifest.json", self.manifest)


# --- stages --------------------------------------------------------------------------------

def _separate(run: _Run, audio) -> tuple[dict, dict]:
    from bandscribe.separation import get_separator
    from bandscribe.separation.base import Stem
    from bandscribe.separation.ensemble import StemConfidence, ensemble

    cfg = run.config
    stems_dir = run.out / "stems"
    meta_path = stems_dir / "stems.json"
    key = {"audio_sha256": audio.file_sha256, "primary": cfg.separator_primary,
           "secondary": cfg.separator_secondary, "blend_weights": cfg.blend_weights}
    if cfg.reuse_cache and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("key") == key and all((stems_dir / f"{n}.wav").is_file() for n in meta["stems"]):
            stems, conf = {}, {}
            for name, info in meta["stems"].items():
                data, sr = sf.read(stems_dir / f"{name}.wav", dtype="float32", always_2d=True)
                stems[name] = Stem(name, np.ascontiguousarray(data.T), sr, info["model_id"])
                if info.get("confidence"):
                    c = info["confidence"]
                    conf[name] = StemConfidence(c["window"], np.array(c["values"]),
                                                np.array(c["active"], dtype=bool), c["source"])
            run.manifest["separation"] = {**meta.get("manifest", {}), "cached": True}
            for mid, extra in meta.get("models", {}).items():
                run.model(mid, **extra)
            run.report(1.0, "reused cached stems")
            return stems, conf

    results = []
    ids = [cfg.separator_primary] + ([cfg.separator_secondary] if cfg.separator_secondary else [])
    for k, mid in enumerate(ids):
        run.check_cancel()
        run.report(k / len(ids), f"separating with {mid}")
        res = get_separator(mid).separate(
            audio.samples, audio.sr,
            progress=lambda f, k=k: run.report((k + f) / len(ids), f"separating with {ids[k]}"))
        results.append(res)
        run.model(mid, files_sha256=res.files_sha256)
        run.manifest["separation"][mid] = {
            "device": res.device, "batch_size": res.batch_size, "chunk_size": res.chunk_size,
            "peak_vram_gb": round(res.peak_vram_gb, 3), "seconds": round(res.seconds, 2),
            "oom_retries": res.oom_retries, "notes": list(res.notes)}
        for note in res.notes:
            run.warn(f"{mid}: {note}")
    ens = ensemble(results[0], results[1] if len(results) > 1 else None, cfg.blend_weights)
    stems_dir.mkdir(parents=True, exist_ok=True)
    meta = {"key": key, "stems": {}, "manifest": run.manifest["separation"],
            "models": {mid: {"files_sha256": r.files_sha256} for mid, r in zip(ids, results, strict=True)}}
    for name, stem in ens.stems.items():
        sf.write(stems_dir / f"{name}.wav", stem.audio.T, stem.sr, subtype="PCM_24")
        c = ens.confidence.get(name)
        meta["stems"][name] = {
            "model_id": stem.model_id, "weights": ens.weights.get(name),
            "confidence": None if c is None else {
                "window": c.window, "values": [round(float(v), 4) for v in c.values],
                "active": [bool(a) for a in c.active], "source": c.source}}
    _write_json(meta_path, meta)
    return ens.stems, ens.confidence


def _transcribe(run: _Run, audio, stems: dict, activity: dict) -> tuple[dict, dict, list | None]:
    """Run every configured backend. Returns (raw[backend][family], drums raw, mix notes)."""
    from bandscribe.transcription import get_transcriber

    cfg = run.config
    notes_dir = run.out / "notes"
    pitched = [f for f in cfg.instruments if f != "drums" and f in stems and activity[f].present
               and cfg.stem_overrides.get(f) != "absent"]
    do_drums = ("drums" in cfg.instruments and "drums" in stems and activity["drums"].present
                and cfg.stem_overrides.get("drums") != "absent")
    raw: dict[str, dict[str, list[Note]]] = {}
    mix_notes: list[Note] | None = None
    backends = list(dict.fromkeys([*cfg.pitched_transcribers, *(cfg.drum_transcribers if do_drums else []),
                                   *([cfg.mix_vote] if cfg.mix_vote else [])]))
    for bi, backend in enumerate(backends):
        run.check_cancel()
        run.report(bi / max(1, len(backends)), f"transcribing with {backend}")
        started = time.perf_counter()
        jobs: list[tuple[str, np.ndarray]] = []
        if backend in cfg.pitched_transcribers:
            jobs += [(f, stems[f].audio) for f in pitched]
        if do_drums and backend in cfg.drum_transcribers:
            jobs.append(("drums", stems["drums"].audio))
        if backend == cfg.mix_vote:
            jobs.append(("mix", audio.samples))
        try:
            tr = get_transcriber(backend)
            def sub(fraction: float, message: str = "", bi: int = bi) -> None:
                run.check_cancel()
                run.report((bi + fraction) / max(1, len(backends)), message)

            if hasattr(tr, "transcribe_many"):
                outs = tr.transcribe_many([(a, audio.sr, f) for f, a in jobs], progress=sub)
            else:
                outs = []
                for k, (f, a) in enumerate(jobs):
                    sub(k / len(jobs), f"{backend}: {f}")
                    outs.append(tr.transcribe(a, audio.sr, f))
        except TranscriberUnavailable as exc:
            run.manifest["unavailable"][backend] = str(exc)
            run.warn(f"{backend} is unavailable and was skipped: {str(exc).splitlines()[0]}")
            continue
        raw[backend] = {}
        for (family, _a), notes in zip(jobs, outs, strict=True):
            if family == "mix":
                mix_notes = notes
                _write_json(notes_dir / f"mix_{backend}.json", notes_to_json(notes))
            else:
                raw[backend][family] = notes
                _write_json(notes_dir / f"raw_{backend}_{family}.json", notes_to_json(notes))
        run.model(backend, sha256=getattr(tr, "model_sha256", None))
        run.manifest["transcription"][backend] = {
            "seconds": round(time.perf_counter() - started, 2),
            "last_run": getattr(tr, "last_run", None),
            "notes": {f: len(n) for f, n in raw[backend].items()}}
    return raw, {"pitched": pitched, "drums": do_drums}, mix_notes


def _vote_and_attribute(run: _Run, raw: dict, plan: dict, mix_notes, stems, activity,
                        confidence) -> tuple[dict[str, list[Note]], Any]:
    from bandscribe.attribution import instrument_id as attr
    from bandscribe.transcription.consensus import consensus
    from bandscribe.transcription.drums import flag_limb_overload, vote_drums

    cfg = run.config
    notes_dir = run.out / "notes"
    log = attr.AttributionLog()
    final: dict[str, list[Note]] = {}
    stats: dict[str, Any] = {}
    available = [b for b in cfg.pitched_transcribers if b in raw]
    if not available and plan["pitched"]:
        raise RuntimeError("no pitched transcriber is available; see manifest.json 'unavailable'")
    if len(available) == 1 and plan["pitched"]:
        run.warn(f"only {available[0]} is available: pitched notes have no second opinion")
    for family in plan["pitched"]:
        primary = raw[available[0]].get(family, [])
        secondary = raw[available[1]].get(family, []) if len(available) > 1 else None
        notes, st = consensus(primary, secondary, tolerance=cfg.onset_tolerance,
                              drop_unconfirmed=False)
        stats[family] = {"matched": st.matched, "primary_only": st.primary_only,
                         "secondary_only": st.secondary_only, "agreement": round(st.agreement, 3)}
        _write_json(notes_dir / f"consensus_{family}.json", notes_to_json(notes))
        notes = attr.apply_range(notes, family, log)
        notes = attr.drop_inactive(notes, activity[family], log)
        notes = attr.apply_stem_confidence(notes, confidence.get(family))
        notes = attr.vote_with_mix(notes, family, mix_notes, log, cfg.onset_tolerance)
        final[family] = notes
    final = attr.resolve_leakage(final, stems, log, cfg.onset_tolerance)
    if "other" in final:
        remaining, promoted = attr.promote_from_other(final["other"], mix_notes, activity, log,
                                                      cfg.onset_tolerance)
        final["other"] = remaining
        for family, extra in promoted.items():
            if family in cfg.instruments:
                final[family] = sorted([*final.get(family, []), *extra],
                                       key=lambda n: (n.onset, n.pitch))
    attr.assert_no_unsupported_other_to_guitar(final)

    if plan["drums"]:
        det_backend = next((b for b in cfg.drum_transcribers if b in raw and b != "adtof"), None)
        detailed = raw[det_backend].get("drums") if det_backend else None
        coarse = raw["adtof"].get("drums") if "adtof" in raw else None
        drums, st, discarded = vote_drums(detailed, coarse, tolerance=cfg.drum_onset_tolerance)
        if detailed is None or coarse is None:
            run.warn("drums: only one drum transcriber is available (no second opinion)")
        stats["drums"] = {"matched": st.matched, "primary_only": st.primary_only,
                          "secondary_only": st.secondary_only, "non_kit_discarded": discarded}
        _write_json(notes_dir / "consensus_drums.json", notes_to_json(drums))
        drums = attr.drop_inactive(drums, activity["drums"], log)
        drums = attr.apply_stem_confidence(drums, confidence.get("drums"))
        final["drums"] = flag_limb_overload(drums)

    if cfg.strict:
        for family, notes in final.items():
            kept = [n for n in notes if n.conf >= cfg.low_confidence]
            for n in notes:
                if n.conf < cfg.low_confidence:
                    log.remove(n, "strict mode: below the confidence threshold")
            final[family] = kept
    run.manifest["consensus"] = stats
    return final, log


def _assign_tab(run: _Run, final: dict[str, list[Note]], log) -> dict[str, Any]:
    from bandscribe.tab.fret_dp import DPFretAssigner, FretOptions
    from bandscribe.tab.tunings import TUNINGS, suggest_tuning

    cfg = run.config
    info: dict[str, Any] = {}
    for family, setting in (("guitar", cfg.guitar_tuning), ("bass", cfg.bass_tuning)):
        if family not in final:
            continue
        if setting == "auto":
            sug = suggest_tuning(final[family], family)
            tuning = sug.tuning
            info[family] = {"tuning": tuning.key, "name": tuning.name, "auto": True,
                            "reason": sug.reason, "ambiguous_with": list(sug.ambiguous_with)}
        else:
            if setting not in TUNINGS or TUNINGS[setting].instrument != family:
                raise ValueError(f"unknown {family} tuning {setting!r}")
            tuning = TUNINGS[setting]
            info[family] = {"tuning": tuning.key, "name": tuning.name, "auto": False}
        res = DPFretAssigner(FretOptions(max_span=cfg.max_span)).assign(final[family], tuning)
        for n in res.unplaced:
            log.remove(n, f"unplayable in {tuning.name} (no string/fret position)")
        final[family] = res.notes
        info[family]["unplaced"] = len(res.unplaced)
        info[family]["strings"] = list(tuning.strings)
    return info


def _build_score(run: _Run, audio, beat_result, tempo_map, meter, final, tab_info):
    from bandscribe.analysis.key import estimate_key
    from bandscribe.analysis.sections import detect_sections
    from bandscribe.notation.quantize import build_bars, quantize_notes
    from bandscribe.notation.score_model import Score, Track

    cfg = run.config
    all_notes = [n for notes in final.values() for n in notes]
    last = max([audio.duration * 0.0] + [n.offset for n in all_notes]) if all_notes else 0.0
    last = max(last, float(tempo_map.beats[-1]) if len(tempo_map.beats) else 0.0)
    first = min([n.onset for n in all_notes], default=0.0)
    try:
        sections = detect_sections(audio.mono(), audio.sr, meter.bars)
    except Exception as exc:  # noqa: BLE001 - sections are optional decoration
        run.warn(f"section detection failed: {exc}")
        sections = []
    bars = build_bars(tempo_map, meter, first, last, sections)
    key = estimate_key([n for f, notes in final.items() if f != "drums" for n in notes])
    key_dict = None if key is None else {
        "name": key.name, "tonic": key.tonic, "mode": key.mode, "fifths": key.fifths,
        "correlation": round(key.correlation, 3), "margin": round(key.margin, 3)}
    tracks = []
    qstats = {}
    for family in TRACK_ORDER:
        if family not in final or not final[family]:
            continue
        name, kind, program = TRACK_DEFS[family]
        sn, grid, st = quantize_notes(final[family], bars, tempo_map, drums=family == "drums",
                                      low_confidence=cfg.low_confidence)
        t = Track(family, name, family, kind, program, notes=sn, grid=grid)
        if family in tab_info:
            t.tuning_key = tab_info[family]["tuning"]
            t.tuning = tuple(tab_info[family]["strings"])
            t.info["tuning_name"] = tab_info[family]["name"]
        tracks.append(t)
        qstats[family] = asdict(st)
    title = cfg.title or run.audio_path.stem
    score = Score(title, bars, tracks, key_dict,
                  {"source_audio": run.audio_path.name, "low_confidence_threshold": cfg.low_confidence,
                   "quantize": qstats, "tab": tab_info})
    analysis = {
        "beats": [round(float(b), 4) for b in beat_result.beats],
        "downbeats": [round(float(b), 4) for b in beat_result.downbeats],
        "global_bpm": tempo_map.global_bpm,
        "tempo_segments": [asdict(s) for s in tempo_map.segments],
        "meter": {"dominant": list(meter.dominant), "pickup_beats": meter.pickup_beats,
                  "changes": [asdict(c) for c in meter.changes]},
        "bars": [{"index": b.index, "numerator": b.numerator, "denominator": b.denominator,
                  "start_time": round(b.start_time, 4), "bpm": b.bpm, "section": b.section,
                  "confident": b.confident} for b in bars],
        "key": key_dict, "sections": [asdict(s) for s in sections]}
    return score, analysis


def _export(run: _Run, score) -> None:
    from bandscribe.notation import export_gp, export_gp5, export_midi, export_musicxml

    cfg = run.config
    exports = run.out / "exports"
    base = _safe_name(score.title)
    reports = run.manifest["exports"]
    gp_ok = False
    if "gp" in cfg.exports:
        try:
            rep = export_gp.export_gp(score, exports / f"{base}.gp")
            gp_ok = rep.verified
            reports["gp"] = {"path": str(rep.path), "bytes": rep.bytes, "verified": rep.verified,
                             "mismatches": rep.mismatches, "notes_written": rep.notes_written,
                             "alphatab": rep.alphatab_version}
            if not rep.verified:
                run.warn("the .gp file differs from the score when read back: "
                         + "; ".join(rep.mismatches[:3]))
        except Exception as exc:  # noqa: BLE001 - fall back to .gp5 below
            reports["gp"] = {"error": str(exc)}
            run.warn(f".gp export failed: {str(exc).splitlines()[0]}")
    run.report(0.4)
    if "gp5" in cfg.exports or ("gp" in cfg.exports and not gp_ok):
        try:
            rep5 = export_gp5.export_gp5(score, exports / f"{base}.gp5")
            reports["gp5"] = {"path": str(rep5.path), "notes_written": rep5.notes_written,
                              "notes_dropped": rep5.notes_dropped, "warnings": rep5.warnings}
        except Exception as exc:  # noqa: BLE001
            reports["gp5"] = {"error": str(exc)}
            run.warn(f".gp5 export failed: {str(exc).splitlines()[0]}")
    run.report(0.6)
    for fmt, fn, ext in (("midi", export_midi.export_midi, "mid"),
                         ("musicxml", export_musicxml.export_musicxml, "musicxml")):
        if fmt in cfg.exports:
            try:
                reports[fmt] = {"path": str(fn(score, exports / f"{base}.{ext}"))}
            except Exception as exc:  # noqa: BLE001
                reports[fmt] = {"error": str(exc)}
                run.warn(f"{fmt} export failed: {str(exc).splitlines()[0]}")
    run.report(1.0)


# --- entry point ---------------------------------------------------------------------------

def run_pipeline(audio_path: str | Path, config: PipelineConfig | None = None,
                 out_dir: str | Path | None = None, progress: ProgressFn | None = None,
                 cancel: Callable[[], bool] | None = None) -> Path:
    """Transcribe ``audio_path`` and return the output directory.

    ``config`` defaults to ``PipelineConfig.from_defaults()``. ``out_dir`` defaults to
    ``out/<audio file name>`` under the current directory. ``progress(stage, fraction,
    message)`` is called often; ``cancel()`` returning True aborts with PipelineCancelled.
    """
    from bandscribe import qa
    from bandscribe.analysis.beats import track_beats
    from bandscribe.analysis.meter import analyze_meter
    from bandscribe.analysis.tempo import build_tempo_map
    from bandscribe.attribution.instrument_id import analyze_stems
    from bandscribe.io.audio_load import load_audio

    audio_path = Path(audio_path)
    config = config or PipelineConfig.from_defaults()
    config.validate()
    out = Path(out_dir) if out_dir else Path("out") / _safe_name(audio_path.stem)
    out.mkdir(parents=True, exist_ok=True)
    run = _Run(audio_path, config, out, progress, cancel)
    try:
        run.stage("load", "decoding audio")
        audio = load_audio(audio_path)
        run.manifest["audio"] = {
            "path": str(audio_path), "sha256": audio.file_sha256, "duration": round(audio.duration, 3),
            "orig_sr": audio.orig_sr, "orig_channels": audio.orig_channels, "gain": audio.gain}

        run.stage("beats", "tracking beats")
        beats = track_beats(audio)
        run.model(config.beat_model, sha256=beats.model_sha256)
        tempo_map = build_tempo_map(beats.beats)
        meter = analyze_meter(beats.beats, beats.downbeats)
        if len(beats.beats) < 2:
            run.warn("fewer than two beats were tracked: the score uses a fallback tempo")

        run.stage("separate", "separating stems")
        stems, confidence = _separate(run, audio)

        run.stage("attribute", "checking which stems contain an instrument")
        activity = analyze_stems(stems, audio.samples)
        stems_meta = json.loads((out / "stems" / "stems.json").read_text(encoding="utf-8"))
        for name, a in activity.items():
            stems_meta["stems"][name]["activity"] = {
                "present": a.present, "rms_db": round(a.rms_db, 2),
                "relative_db": round(a.relative_db, 2), "active_seconds": a.active_seconds,
                "active_fraction": round(a.active_fraction, 4)}
        _write_json(out / "stems" / "stems.json", stems_meta)
        if "drums" in stems and activity["drums"].present:
            try:  # spec 4.1.3: beats on the drum stem as a cross-check
                import mir_eval

                drum_beats = track_beats(stems["drums"].audio.mean(axis=0), audio.sr)
                f = float(mir_eval.beat.f_measure(beats.beats, drum_beats.beats, 0.07)) \
                    if len(beats.beats) and len(drum_beats.beats) else 0.0
                run.manifest["beat_agreement_mix_vs_drums"] = round(f, 3)
                if f < 0.8:
                    run.warn(f"beats tracked on the mix and on the drum stem agree only {f:.0%}")
            except Exception as exc:  # noqa: BLE001 - a cross-check must not stop the run
                run.warn(f"drum-stem beat cross-check failed: {exc}")

        run.stage("transcribe", "transcribing")
        raw, plan, mix_notes = _transcribe(run, audio, stems, activity)
        final, log = _vote_and_attribute(run, raw, plan, mix_notes, stems, activity, confidence)

        run.stage("tab", "assigning strings and frets")
        tab_info = _assign_tab(run, final, log)
        for family, notes in final.items():
            _write_json(out / "notes" / f"final_{family}.json", notes_to_json(notes))
        _write_json(out / "notes" / "attribution.json",
                    {"removed": log.removed, "promoted": log.promoted, "notes": log.notes})

        run.stage("notation", "quantising and building the score")
        score, analysis = _build_score(run, audio, beats, tempo_map, meter, final, tab_info)
        _write_json(out / "analysis.json", analysis)
        score.save(out / "score.json")

        run.stage("qa", "running quality checks")
        context = {"config": config, "out_dir": out, "notes": final, "activity": activity,
                   "confidence": confidence, "attribution": log, "manifest": run.manifest}
        result = qa.run_checks(score, context) or {}
        replaced = result.pop("score", None)
        if replaced is not None:
            score = replaced
            score.save(out / "score.json")
        _write_json(out / "qa.json", result)
        run.manifest["qa"] = result
        if result.get("block"):
            raise QABlocked(str(result.get("reason", "blocked by QA")))

        run.stage("export", "writing files")
        _export(run, score)
        run.finish_stage()
        run.manifest["status"] = "done"
        run.manifest["summary"] = {
            "bars": len(score.bars), "tracks": {t.family: len(t.notes) for t in score.tracks},
            "low_confidence": {t.family: sum(n.low_conf for n in t.notes) for t in score.tracks}}
        return out
    except PipelineCancelled:
        run.manifest["status"] = "cancelled"
        raise
    except Exception as exc:
        run.manifest["status"] = "failed"
        run.manifest["error"] = {"type": type(exc).__name__, "message": str(exc),
                                 "traceback": traceback.format_exc()[-4000:]}
        raise
    finally:
        run.finish_stage()
        run.save_manifest()


def with_overrides(config: PipelineConfig, **overrides: Any) -> PipelineConfig:
    cfg = replace(config, **overrides)
    cfg.validate()
    return cfg

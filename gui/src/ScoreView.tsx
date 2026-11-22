import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import * as alphaTab from '@coderline/alphatab';
import { Alignment, AlignmentAnchor, api, BarInfo, Project, Waveform } from './api';
import { familyLabel, fmtTime } from './App';
import { CorrectionPanel, EditorFocus } from './Corrections';
import { familyIcon, I, NoteGlyph } from './icons';
import { drawWave } from './wave';

const LOW_COLOR = '#c96a00';
const TICKS_PER_BEAT = 960;
const SPEEDS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1];
const ZOOMS = [0.6, 0.7, 0.8, 0.9, 1, 1.15, 1.3, 1.5];
const NUDGES = [-1, -0.1, -0.01, 0.01, 0.1, 1];
const STEM_ORDER = ['guitar', 'bass', 'piano', 'vocals', 'other', 'drums'];
const STAFF: { id: string; label: string; profile: alphaTab.StaveProfile }[] = [
  { id: 'both', label: '標準＋TAB', profile: alphaTab.StaveProfile.Default },
  { id: 'tab', label: 'TAB', profile: alphaTab.StaveProfile.Tab },
  { id: 'score', label: '標準', profile: alphaTab.StaveProfile.Score },
];

// tx = this run's transcription, fixed = the last exported corrected .gp, ref = the user's Guitar Pro reference.
type View = 'tx' | 'fixed' | 'ref';
type Panel = 'edit' | 'align';
type Low = { tick: number; track: number };
type Range = { startTick: number; endTick: number };
type RefInfo = { barTicks: number[]; meters: [number, number][]; tempo: number; end: number };

// Baseline timing: bar starts from the beat tracker, one knot per bar plus the end of the last bar.
type Base = { barTicks: number[]; barDen: number[]; endTick: number; ticks: number[]; secs: number[] };
// The one score<->recording mapping used everywhere: strictly increasing knots, linear in between.
type TimeMap = { ticks: number[]; secs: number[]; barTicks: number[]; barDen: number[] };

function buildBase(score: alphaTab.model.Score, bars: BarInfo[]): Base | null {
  const mbs = score.masterBars;
  if (!bars.length || !mbs.length) return null;
  const base: Base = { barTicks: [], barDen: [], endTick: 0, ticks: [], secs: [] };
  let bpm = bars[0].bpm, prev = -Infinity;
  const push = (tick: number, s: number) => { prev = Math.max(s, prev + 1e-3); base.ticks.push(tick); base.secs.push(prev); };
  mbs.forEach((mb, i) => {
    const b = bars[i];
    if (b) bpm = b.bpm;
    base.barTicks.push(mb.start);
    base.barDen.push(mb.timeSignatureDenominator);
    push(mb.start, b ? b.start_time : prev + (mbs[i - 1].calculateDuration() / TICKS_PER_BEAT) * (60 / bpm));
    if (b?.beat_times?.length === b.numerator + 1) {
      const step = TICKS_PER_BEAT * 4 / b.denominator;
      for (let j = 1; j < b.numerator; j++) {
        const time = b.beat_times[j];
        if (Number.isFinite(time) && time > prev && time < (b.end_time ?? Infinity)) push(mb.start + j * step, time);
      }
    }
  });
  const n = mbs.length - 1, dur = mbs[n].calculateDuration();
  base.endTick = mbs[n].start + dur;
  push(base.endTick, bars[n]?.end_time ?? prev + (dur / TICKS_PER_BEAT) * (60 / bpm));
  return base;
}

function lastAtOrBefore(list: number[], v: number): number {
  let lo = 0, hi = list.length - 1, ans = 0;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (list[mid] <= v) { ans = mid; lo = mid + 1; } else hi = mid - 1;
  }
  return ans;
}
function interp(xs: number[], ys: number[], x: number): number {
  const n = xs.length - 1;
  if (x <= xs[0]) return ys[0];
  if (x >= xs[n]) return ys[n];
  const i = Math.min(lastAtOrBefore(xs, x), n - 1);
  return ys[i] + ((x - xs[i]) * (ys[i + 1] - ys[i])) / (xs[i + 1] - xs[i]);
}

// Offset shifts the baseline. Anchors are pinned exactly; between two anchors the baseline is
// stretched to fit, before the first / after the last it is shifted to meet the nearest anchor.
function buildMap(base: Base, offset: number, anchors: AlignmentAnchor[]): TimeMap {
  const bx = base.ticks;
  const by = base.secs.map((s) => s + offset);
  const out = { barTicks: base.barTicks, barDen: base.barDen };
  const pts = anchors.map((a) => ({ tick: Math.round(a.beat * TICKS_PER_BEAT), sec: a.seconds }))
    .filter((p) => p.tick >= 0 && p.tick <= base.endTick).sort((a, b) => a.tick - b.tick);
  if (!pts.length) return { ...out, ticks: bx, secs: by };
  const B = (t: number) => interp(bx, by, t);
  const f = (t: number) => {
    const k = pts[0].tick > t ? -1 : lastAtOrBefore(pts.map((p) => p.tick), t);
    if (k < 0) return B(t) + pts[0].sec - B(pts[0].tick);
    if (k === pts.length - 1) return B(t) + pts[k].sec - B(pts[k].tick);
    const a = pts[k], b = pts[k + 1], ba = B(a.tick), bb = B(b.tick);
    return a.sec + ((B(t) - ba) * (b.sec - a.sec)) / (bb - ba);
  };
  const ticks = [...new Set([...bx, ...pts.map((p) => p.tick)])].sort((a, b) => a - b);
  return { ...out, ticks, secs: ticks.map(f) };
}

const tickToSec = (m: TimeMap, tick: number) => interp(m.ticks, m.secs, tick);
const secToTick = (m: TimeMap, sec: number) => Math.round(interp(m.secs, m.ticks, sec));
const barAtTick = (m: TimeMap, tick: number) => lastAtOrBefore(m.barTicks, tick);
const barAtSec = (m: TimeMap, sec: number) => barAtTick(m, secToTick(m, sec));
const endTickOf = (m: TimeMap) => m.ticks[m.ticks.length - 1];
const tickToBeat = (tick: number) => Math.round((tick / TICKS_PER_BEAT) * 1e6) / 1e6;

function describeTick(m: TimeMap, tick: number): string {
  const i = barAtTick(m, tick);
  const unit = (TICKS_PER_BEAT * 4) / (m.barDen[i] || 4);
  const beat = Math.round(((tick - m.barTicks[i]) / unit + 1) * 100) / 100;
  return `${i + 1} 小節 ${beat} 拍`;
}
const fmtPrecise = (s: number) => {
  const sign = s < 0 ? '-' : '';
  const a = Math.abs(s);
  return `${sign}${Math.floor(a / 60)}:${(a % 60).toFixed(2).padStart(5, '0')}`;
};
const fmtDelta = (d: number) => `${d >= 0 ? '+' : '−'}${Math.abs(d).toFixed(2)}s`;
const sameAlignment = (a: { offset: number; anchors: AlignmentAnchor[] }, b: { offset: number; anchors: AlignmentAnchor[] }) =>
  a.offset === b.offset && a.anchors.length === b.anchors.length
  && a.anchors.every((p, i) => p.beat === b.anchors[i].beat && p.seconds === b.anchors[i].seconds);
const encPath = (p: string) => p.split('/').map(encodeURIComponent).join('/');

// Colour the notes of every beat that carries the low-confidence mark ("?") and list where they are.
function markLowConfidence(score: alphaTab.model.Score): Low[] {
  const found: Low[] = [];
  const m = alphaTab.model as any;
  const color = m.Color?.fromJson ? m.Color.fromJson(LOW_COLOR) : null;
  score.tracks.forEach((track, ti) => {
    for (const staff of track.staves) for (const bar of staff.bars) for (const voice of bar.voices)
      for (const beat of voice.beats) {
        if (beat.text !== '?') continue;
        found.push({ tick: beat.absolutePlaybackStart, track: ti });
        if (!color || !m.NoteStyle || !m.NoteSubElement) continue;
        for (const note of beat.notes) {
          const style = new m.NoteStyle();
          for (const key of ['StandardNotationNoteHead', 'GuitarTabFretNumber', 'StandardNotationAccidentals']) {
            if (m.NoteSubElement[key] !== undefined) style.colors.set(m.NoteSubElement[key], color);
          }
          (note as any).style = style;
        }
      }
  });
  return found.sort((a, b) => a.tick - b.tick);
}

// Instrument family of a track in a foreign Guitar Pro file, from its General MIDI program.
function refFamily(t: alphaTab.model.Track): string {
  if (t.staves.some((s) => s.isPercussion)) return 'drums';
  const p = t.playbackInfo.program;
  if (p >= 24 && p <= 31) return 'guitar';
  if (p >= 32 && p <= 39) return 'bass';
  if (p <= 7) return 'piano';
  if (p >= 52 && p <= 54) return 'vocals';
  return 'other';
}

export function ScorePage({ project, onAttachReference }: {
  project: Project; onAttachReference: (file: File) => Promise<void>;
}) {
  const host = useRef<HTMLDivElement>(null);
  const wrap = useRef<HTMLDivElement>(null);
  const audio = useRef<HTMLAudioElement>(null);
  const refInput = useRef<HTMLInputElement>(null);
  const atRef = useRef<alphaTab.AlphaTabApi | null>(null);
  const curTick = useRef(0);
  const pendingSeek = useRef(0);
  const lastBar = useRef(-1);
  const focusSeq = useRef(0);
  const pid = project.meta.id;

  const stems = project.stems?.stems ?? {};
  const stemNames = STEM_ORDER.filter((n) => n in stems);
  const [source, setSource] = useState<string>(stemNames.includes('guitar') ? 'guitar' : 'input');
  const sourceRef = useRef(source);
  sourceRef.current = source;
  const recSource = useRef(source);

  const [view, setView] = useState<View>('tx');
  const viewRef = useRef(view);
  viewRef.current = view;
  const [atGen, setAtGen] = useState(0);
  const [fixedPath, setFixedPath] = useState<string | null>(null);
  const [fixedRev, setFixedRev] = useState(0);
  const [fixedStale, setFixedStale] = useState(false);
  const [refInfo, setRefInfo] = useState<RefInfo | null>(null);
  const [attaching, setAttaching] = useState(false);
  const [attachError, setAttachError] = useState<string | null>(null);

  const [tracks, setTracks] = useState<{ index: number; name: string; family: string; strings: number }[]>([]);
  const [visible, setVisible] = useState<number[] | null>(null);
  const [lows, setLows] = useState<Low[]>([]);
  const [base, setBase] = useState<Base | null>(null);
  const [ready, setReady] = useState(false);
  const [playerReady, setPlayerReady] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [speed, setSpeed] = useState(1);
  const [loop, setLoop] = useState(false);
  const loopRef = useRef(loop);
  loopRef.current = loop;
  const [range, setRange] = useState<Range | null>(null);
  const rangeRef = useRef(range);
  rangeRef.current = range;
  const [staff, setStaff] = useState('both');
  const [zoom, setZoom] = useState(0.9);
  const [panel, setPanel] = useState<Panel | null>(null);
  const panelRef = useRef(panel);
  panelRef.current = panel;

  // Manual alignment: what the server has, and the working copy being edited.
  const [saved, setSaved] = useState<Alignment | null>(null);
  const [offset, setOffset] = useState(0);
  const [anchors, setAnchors] = useState<AlignmentAnchor[]>([]);
  const [alignError, setAlignError] = useState<string | null>(null);
  const [alignNote, setAlignNote] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [selTick, setSelTick] = useState<number | null>(null);

  // Note correction: one guitar TAB track (project.score.tracks id) and one 0-based bar at a time.
  const [corrTrack, setCorrTrack] = useState<string | null>(null);
  const [corrBar, setCorrBar] = useState(0);
  const [corrDirty, setCorrDirty] = useState(0);
  const [focus, setFocus] = useState<EditorFocus | null>(null);
  const guitars = useMemo(() => (project.score?.tracks ?? []).filter((t) => t.family === 'guitar' && t.kind === 'tab'),
    [project.score]);

  const map = useMemo(() => (base && view !== 'ref' ? buildMap(base, offset, anchors) : null), [base, offset, anchors, view]);
  const mapRef = useRef(map);
  mapRef.current = map;
  const refInfoRef = useRef(refInfo);
  refInfoRef.current = refInfo;

  const gpFile = project.exports.find((f) => f.endsWith('.gp'));
  const bars = project.score?.bars ?? [];
  const duration = project.waveform.duration;
  const ext = project.meta.filename.slice(project.meta.filename.lastIndexOf('.')).toLowerCase();
  const audioSrc = source === 'synth' ? undefined
    : source === 'input' ? api.fileUrl(pid, `input${ext}`)
      : api.fileUrl(pid, `run/stems/${source}.wav`);
  const reference = project.reference;
  const scoreUrl = view === 'tx' ? (gpFile ? api.fileUrl(pid, `run/exports/${encodeURIComponent(gpFile)}`) : null)
    : view === 'fixed' ? (fixedPath ? `${api.fileUrl(pid, encPath(fixedPath))}?v=${fixedRev}` : null)
      : reference ? api.fileUrl(pid, encPath(reference.path)) : null;

  // Keep the played beat on screen when the recording drives the cursor or after a jump.
  const ensureVisible = useCallback((beat: alphaTab.model.Beat) => {
    const at = atRef.current, w = wrap.current;
    const bounds = at?.boundsLookup?.findBeat(beat)?.barBounds.masterBarBounds.realBounds;
    if (!w || !bounds) return;
    if (bounds.y < w.scrollTop || bounds.y + bounds.h > w.scrollTop + w.clientHeight) {
      w.scrollTo({ top: Math.max(0, bounds.y - 24) });
    }
  }, []);

  const loadAlignment = useCallback(() => {
    setAlignError(null);
    setAlignNote(null);
    api.alignment(pid)
      .then((a) => { setSaved(a); setOffset(a.offset); setAnchors(a.anchors); })
      .catch((e) => { setSaved(null); setAlignError(`讀取對齊失敗：${e.message ?? e}`); });
  }, [pid]);
  useEffect(loadAlignment, [loadAlignment, project.manifest?.stages?.export]);

  // A corrected export from an earlier session lives next to the corrections of this score hash.
  useEffect(() => {
    setFixedPath(null);
    if (!saved?.score_hash) return;
    const path = `corrections/${saved.score_hash}/corrected.gp`;
    let live = true;
    fetch(api.fileUrl(pid, path)).then((r) => { if (live && r.ok) setFixedPath(path); }).catch(() => undefined);
    return () => { live = false; };
  }, [pid, saved?.score_hash]);

  // One alphaTab instance per project run; the selected view decides which file it shows.
  useEffect(() => {
    if (!host.current || !gpFile) return;
    setPlayerReady(false); setView('tx');
    const at = new alphaTab.AlphaTabApi(host.current, {
      core: { fontDirectory: '/font/', tracks: 'all' },
      player: {
        playerMode: alphaTab.PlayerMode.EnabledSynthesizer, soundFont: '/soundfont/sonivox.sf2',
        scrollElement: wrap.current ?? undefined, enableCursor: true,
        enableSeekToClick: true, enablePlaybackRangeSelection: true,
      },
      display: { layoutMode: alphaTab.LayoutMode.Page, scale: zoom,
        staveProfile: STAFF.find((s) => s.id === staff)!.profile },
    } as any);
    atRef.current = at;
    at.scoreLoaded.on((score) => {
      const isRef = viewRef.current === 'ref';
      const summary = project.score?.tracks ?? [];
      const list = score.tracks.map((t, i) => {
        const info = summary.find((s) => s.name === t.name);
        return { index: i, name: t.name, family: isRef || !info ? refFamily(t) : info.family, strings: t.staves[0]?.tuning.length ?? 0 };
      });
      if (isRef) {
        setLows([]);
        setBase(null);
        const mbs = score.masterBars, last = mbs[mbs.length - 1];
        setRefInfo({ barTicks: mbs.map((mb) => mb.start), meters: mbs.map((mb) => [mb.timeSignatureNumerator, mb.timeSignatureDenominator]),
          tempo: score.tempo, end: last ? last.start + last.calculateDuration() : 0 });
      } else {
        setLows(markLowConfidence(score));
        setBase(buildBase(score, bars));
        setRefInfo(null);
      }
      setTracks(list);
      // Open on the guitar part(s) when there are other tracks too.
      const g = list.filter((t) => t.family === 'guitar').map((t) => t.index);
      const initial = g.length && g.length < list.length ? g : null;
      setVisible(initial);
      window.setTimeout(() => at.renderTracks(initial ? initial.map((i) => score.tracks[i]) : score.tracks), 0);
    });
    at.renderStarted.on(() => setReady(false));
    at.renderFinished.on(() => setReady(true));
    at.playerReady.on(() => setPlayerReady(true));
    at.playerStateChanged.on((e: any) => { if (sourceRef.current === 'synth') setPlaying(e.state === 1); });
    at.playerPositionChanged.on((e) => {
      if (sourceRef.current !== 'synth') return;
      curTick.current = e.currentTick;
      const m = mapRef.current;
      setTime(m ? tickToSec(m, e.currentTick) : e.currentTime / 1000);
    });
    at.playbackRangeChanged.on((e) => {
      const r = e.playbackRange;
      setRange(r ? { startTick: r.startTick, endTick: r.endTick } : null);
      const m = mapRef.current, el = audio.current;
      if (r && m && el && sourceRef.current !== 'synth') {
        el.currentTime = Math.max(0, tickToSec(m, r.startTick));
        setTime(el.currentTime);
      }
    });
    at.beatMouseDown.on((beat) => {
      const tick = beat.absolutePlaybackStart;
      curTick.current = tick;
      if (viewRef.current === 'ref') return;
      const m = mapRef.current, el = audio.current;
      setSelTick(tick);
      const bar = beat.voice.bar, track = bar.staff.track;
      const info = (project.score?.tracks ?? []).find((t) => t.name === track.name);
      setCorrBar(bar.index);
      if (info?.family === 'guitar' && info.kind === 'tab') {
        setCorrTrack(info.id);
        setFocus({ track: info.id, bar: bar.index, tick: beat.playbackStart, seq: ++focusSeq.current });
      }
      // While aligning, a click only picks the beat; the recording stays where the ear put it.
      if (panelRef.current === 'align' || !m || !el || sourceRef.current === 'synth') return;
      el.currentTime = Math.max(0, tickToSec(m, tick));
      setTime(el.currentTime);
    });
    at.playedBeatChanged.on((beat) => {
      const bar = beat.voice.bar.index;
      if (bar === lastBar.current) return;
      lastBar.current = bar;
      ensureVisible(beat);
    });
    at.error.on((e: any) => setLoadError(String(e?.message ?? e)));
    setAtGen((g) => g + 1);
    return () => { at.destroy(); atRef.current = null; };
  }, [pid, gpFile, project.manifest?.stages?.export]);  // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const at = atRef.current;
    if (!at || !scoreUrl) return;
    let live = true;
    setReady(false); setLoadError(null); setRange(null); setSelTick(null); setTime(0); curTick.current = 0; lastBar.current = -1;
    fetch(scoreUrl)
      .then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(new Error(`讀取樂譜失敗（${r.status}）`))))
      .then((buf) => { if (live) at.load(new Uint8Array(buf)); })
      .catch((e) => live && setLoadError(String(e.message ?? e)));
    return () => { live = false; };
  }, [atGen, scoreUrl]);

  // The reference disappeared (removed on the import page): fall back to the transcription.
  useEffect(() => { if (view === 'ref' && !reference) switchView('tx'); }, [reference]);  // eslint-disable-line react-hooks/exhaustive-deps

  // The synth plays what is on screen: solo the visible tracks.
  useEffect(() => {
    const at = atRef.current;
    if (!at?.score || !playerReady) return;
    const all = at.score.tracks;
    at.changeTrackSolo(all, false);
    if (visible && visible.length < all.length) at.changeTrackSolo(visible.map((i) => all[i]), true);
  }, [visible, playerReady, tracks]);

  useEffect(() => { if (atRef.current) atRef.current.playbackSpeed = speed; }, [speed, playerReady]);
  useEffect(() => { if (atRef.current) atRef.current.isLooping = loop; }, [loop, playerReady]);

  // While the recording plays it owns the clock: loop it, and drag the score cursor along.
  useEffect(() => {
    if (!playing || source === 'synth') return;
    let raf = 0, lastPush = 0;
    const step = (now: number) => {
      const el = audio.current, m = mapRef.current, r = rangeRef.current;
      if (el && m) {
        if (r && el.currentTime >= tickToSec(m, r.endTick)) {
          if (loopRef.current) el.currentTime = Math.max(0, tickToSec(m, r.startTick));
          else el.pause();
        }
        setTime(el.currentTime);
        if (now - lastPush > 80) {
          lastPush = now;
          const tick = secToTick(m, el.currentTime);
          curTick.current = tick;
          if (atRef.current) atRef.current.tickPosition = tick;
        }
      }
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [playing, source]);

  const seekSec = useCallback((sec: number, moveCursor = true) => {
    const m = mapRef.current, at = atRef.current, el = audio.current;
    const t = Math.max(0, Math.min(duration, sec));
    if (m && at && moveCursor) {
      const tick = secToTick(m, t);
      curTick.current = tick;
      lastBar.current = -1;
      at.tickPosition = tick;
    }
    if (el && sourceRef.current !== 'synth') el.currentTime = t;
    setTime(t);
  }, [duration]);

  const pauseAll = () => { audio.current?.pause(); atRef.current?.pause(); };

  const toggle = useCallback(() => {
    const at = atRef.current, el = audio.current, m = mapRef.current, r = rangeRef.current;
    if (sourceRef.current === 'synth') { if (at && playerReady) at.playPause(); return; }
    if (!el) return;
    if (!el.paused) { el.pause(); return; }
    if (r && m) {
      const s = tickToSec(m, r.startTick), e = tickToSec(m, r.endTick);
      if (el.currentTime < s - 0.05 || el.currentTime >= e) el.currentTime = Math.max(0, s);
    }
    void el.play();
  }, [playerReady]);

  const stop = () => {
    pauseAll();
    if (view === 'ref') { if (atRef.current) atRef.current.tickPosition = range?.startTick ?? 0; setTime(0); return; }
    seekSec(range && map ? tickToSec(map, range.startTick) : 0);
  };

  // Previous / next bar: the first press of "previous" returns to the start of the current bar.
  const stepBar = (dir: 1 | -1) => {
    const at = atRef.current;
    if (view === 'ref') {
      const ri = refInfoRef.current;
      if (!ri || !at) return;
      const i = lastAtOrBefore(ri.barTicks, curTick.current);
      const atStart = curTick.current - ri.barTicks[i] < 120;
      const target = Math.max(0, Math.min(ri.barTicks.length - 1, dir > 0 ? i + 1 : atStart ? i - 1 : i));
      curTick.current = ri.barTicks[target];
      at.tickPosition = ri.barTicks[target];
      return;
    }
    if (!map) return;
    const i = barAtSec(map, time), start = tickToSec(map, map.barTicks[i]);
    const target = Math.max(0, Math.min(map.barTicks.length - 1, dir > 0 ? i + 1 : time - start < 0.3 ? i - 1 : i));
    seekSec(tickToSec(map, map.barTicks[target]));
  };

  const changeSource = (s: string) => {
    pauseAll();
    setPlaying(false);
    pendingSeek.current = time;
    if (s === 'synth') seekSec(time);
    setSource(s);
  };

  const switchView = (v: View) => {
    if (v === view) return;
    pauseAll();
    setPlaying(false);
    if (v === 'ref') {
      // The reference has its own timeline; only its synth playback is meaningful.
      if (sourceRef.current !== 'synth') recSource.current = sourceRef.current;
      setSource('synth');
      setPanel(null);
    } else if (view === 'ref') {
      setSource(recSource.current);
      pendingSeek.current = 0;
    }
    setView(v);
  };

  const changeSpeed = (v: number) => {
    setSpeed(v);
    const el = audio.current;
    if (el) { el.playbackRate = v; el.defaultPlaybackRate = v; }
  };

  const toggleLoop = useCallback(() => setLoop((v) => !v), []);

  const setRangeFromSeconds = (a: number, b: number) => {
    const at = atRef.current, m = mapRef.current;
    if (!at || !m) return;
    const b0 = barAtSec(m, Math.min(a, b)), b1 = barAtSec(m, Math.max(a, b));
    const shown = at.tracks[0]?.staves[0];
    const first = shown?.bars[b0]?.voices[0]?.beats[0];
    const lastBeats = shown?.bars[b1]?.voices[0]?.beats;
    const last = lastBeats?.[lastBeats.length - 1];
    if (first && last) {
      at.highlightPlaybackRange(first, last);
      at.applyPlaybackRangeFromHighlight();
    } else {
      at.playbackRange = { startTick: m.barTicks[b0], endTick: (m.barTicks[b1 + 1] ?? endTickOf(m)) - 50 } as any;
    }
    const r = at.playbackRange;
    if (r) setRange({ startTick: r.startTick, endTick: r.endTick });
    seekSec(tickToSec(m, m.barTicks[b0]));
  };

  const listenBar = (b: number) => {
    const m = mapRef.current;
    if (!m || b < 0 || b >= m.barTicks.length) return;
    const mid = (m.barTicks[b] + (m.barTicks[b + 1] ?? endTickOf(m))) / 2;
    const s = tickToSec(m, mid);
    setRangeFromSeconds(s, s);
    setLoop(true);
  };

  const clearRange = () => {
    const at = atRef.current;
    if (at) { at.playbackRange = null; at.clearPlaybackRangeHighlight(); }
    setRange(null);
  };

  const show = (sel: number[] | null) => {
    const at = atRef.current;
    setVisible(sel);
    if (!at?.score) return;
    at.renderTracks(sel === null ? at.score.tracks : sel.map((i) => at.score!.tracks[i]));
  };

  const applyDisplay = (nextStaff: string, nextZoom: number) => {
    setStaff(nextStaff); setZoom(nextZoom);
    const at = atRef.current;
    if (!at) return;
    at.settings.display.staveProfile = STAFF.find((s) => s.id === nextStaff)!.profile;
    at.settings.display.scale = nextZoom;
    at.updateSettings();
    at.render();
  };
  const zoomStep = (dir: 1 | -1) => {
    const i = ZOOMS.indexOf(zoom), next = ZOOMS[Math.max(0, Math.min(ZOOMS.length - 1, (i < 0 ? 3 : i) + dir))];
    if (next !== zoom) applyDisplay(staff, next);
  };

  const shownLows = useMemo(() => {
    const ticks = lows.filter((l) => visible === null || visible.includes(l.track)).map((l) => l.tick);
    return [...new Set(ticks)];
  }, [lows, visible]);

  const jumpLow = (dir: 1 | -1) => {
    const m = mapRef.current;
    if (!m) return;
    const cur = curTick.current;
    const target = dir > 0 ? shownLows.find((t) => t > cur + 10) : [...shownLows].reverse().find((t) => t < cur - 10);
    if (target !== undefined) seekSec(tickToSec(m, target));
  };

  // --- alignment editing ---
  const canAnchor = selTick != null && source !== 'synth' && !!map && !!saved;
  const addAnchor = useCallback(() => {
    const el = audio.current, m = mapRef.current;
    if (selTick == null || !el || !m || sourceRef.current === 'synth') return;
    const beat = tickToBeat(selTick);
    const seconds = Math.round(el.currentTime * 1000) / 1000;
    const next = anchors.filter((a) => a.beat !== beat).concat({ beat, seconds }).sort((a, b) => a.beat - b.beat);
    const i = next.findIndex((a) => a.beat === beat);
    const before = next[i - 1], after = next[i + 1];
    if ((before && before.seconds >= seconds) || (after && after.seconds <= seconds)) {
      const other = before && before.seconds >= seconds ? before : after!;
      setAlignNote(`順序相反：${describeTick(m, selTick)}（${fmtPrecise(seconds)}）與 ${describeTick(m, other.beat * TICKS_PER_BEAT)}（${fmtPrecise(other.seconds)}）。`);
      return;
    }
    setAnchors(next);
    setAlignNote(null);
    // The selected beat now maps exactly to the current time; put the cursor there.
    if (atRef.current) atRef.current.tickPosition = selTick;
  }, [anchors, selTick]);

  const removeAnchor = (beat: number) => { setAnchors(anchors.filter((a) => a.beat !== beat)); setAlignNote(null); };
  const jumpAnchor = (a: AlignmentAnchor) => { setSelTick(Math.round(a.beat * TICKS_PER_BEAT)); seekSec(a.seconds); };
  const changeOffset = (v: number) => {
    if (!Number.isFinite(v) || Math.abs(v) > 3600) return;
    setOffset(Math.round(v * 1000) / 1000);
  };

  const saveAlignment = async () => {
    if (!saved) return;
    setSaving(true);
    setAlignError(null);
    try {
      const r = await api.saveAlignment(pid, { score_hash: saved.score_hash, revision: saved.revision, offset, anchors });
      setSaved(r); setOffset(r.offset); setAnchors(r.anchors); setAlignNote(null);
    } catch (e: any) {
      setAlignError(`儲存失敗：${e.message ?? e}`);
    } finally {
      setSaving(false);
    }
  };
  const revertAlignment = () => {
    if (!saved) return;
    setOffset(saved.offset); setAnchors(saved.anchors); setAlignNote(null);
  };

  // Space plays/pauses, L toggles the loop, A adds an anchor while aligning.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag && ['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON', 'A'].includes(tag)) return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.code === 'Space') { e.preventDefault(); toggle(); }
      else if (e.key === 'l' || e.key === 'L') toggleLoop();
      else if ((e.key === 'a' || e.key === 'A') && panelRef.current === 'align') addAnchor();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [toggle, toggleLoop, addAnchor]);

  // Memoised so the waveform canvas is not redrawn on every clock tick.
  const lowSecs = useMemo(() => (map ? shownLows.map((t) => tickToSec(map, t)) : []), [map, shownLows]);
  const barSecs = useMemo(() => (map ? map.barTicks.map((t) => tickToSec(map, t)) : []), [map]);
  const anchorSecs = useMemo(() => anchors.map((a) => a.seconds), [anchors]);
  const plainMap = useMemo(() => (base ? buildMap(base, offset, []) : null), [base, offset]);
  const editSpan = useMemo<[number, number] | null>(() => {
    if (panel !== 'edit' || !map || corrBar >= map.barTicks.length) return null;
    return [tickToSec(map, map.barTicks[corrBar]), tickToSec(map, map.barTicks[corrBar + 1] ?? endTickOf(map))];
  }, [panel, map, corrBar]);

  if (!gpFile) {
    return <div className="ws-empty"><I.score size={28} /><p>這次執行沒有產生 .gp 檔。</p></div>;
  }

  const summary = project.score;
  const isRef = view === 'ref';
  const dirty = !!saved && !sameAlignment(saved, { offset, anchors });
  const selPredicted = selTick != null && map ? tickToSec(map, selTick) : null;
  const selAnchor = selTick != null ? anchors.find((a) => a.beat === tickToBeat(selTick)) : undefined;
  const rangeBars = range && map ? [barAtTick(map, range.startTick) + 1, barAtTick(map, range.endTick) + 1]
    : range && refInfo ? [lastAtOrBefore(refInfo.barTicks, range.startTick) + 1, lastAtOrBefore(refInfo.barTicks, range.endTick) + 1] : null;
  const rangeSecs: [number, number] | null = range && map ? [tickToSec(map, range.startTick), tickToSec(map, range.endTick)] : null;
  const sourceLabel = (s: string) => (s === 'input' ? '原曲' : `${familyLabel(s)}分軌`);

  // Position display.
  const barIdx = map ? barAtSec(map, time) : refInfo ? lastAtOrBefore(refInfo.barTicks, curTick.current) : 0;
  const barTotal = map ? map.barTicks.length : refInfo ? refInfo.barTicks.length : bars.length;
  const meter = isRef ? refInfo?.meters[barIdx] : bars[barIdx] ? [bars[barIdx].numerator, bars[barIdx].denominator] : undefined;
  const tempo = isRef ? refInfo?.tempo : bars[barIdx]?.bpm ?? project.analysis?.global_bpm;
  const endSec = isRef ? null : duration;

  const trackMeta = (t: { name: string; family: string; strings: number }) => {
    if (isRef) return [familyLabel(t.family), t.strings && t.family !== 'drums' ? `${t.strings} 弦` : null].filter(Boolean).join(' · ');
    const info = summary?.tracks.find((s) => s.name === t.name);
    if (!info) return familyLabel(t.family);
    const parts = [info.info?.tuning_name ? String(info.info.tuning_name) : familyLabel(info.family), `${info.notes} 音`];
    return parts.join(' · ');
  };
  const trackLow = (name: string) => {
    const info = isRef ? undefined : summary?.tracks.find((s) => s.name === name);
    return info && info.notes > 0 && info.low_conf ? Math.round((info.low_conf / info.notes) * 100) : null;
  };

  const attach = async (file: File | undefined) => {
    if (!file) return;
    setAttaching(true);
    setAttachError(null);
    try { await onAttachReference(file); } catch (e: any) { setAttachError(String(e.message ?? e)); }
    finally { setAttaching(false); if (refInput.current) refInput.current.value = ''; }
  };

  const togglePanel = (p: Panel) => setPanel(panel === p ? null : p);

  return (
    <div className="ws">
      <div className="transport" role="toolbar" aria-label="播放控制">
        <div className="tgroup">
          <button className="tbtn" onClick={() => stepBar(-1)} aria-label="上一小節" title="上一小節"><I.skipBack /></button>
          <button className="tbtn play" onClick={toggle} disabled={source === 'synth' && !playerReady}
            aria-label={playing ? '暫停' : '播放'} title={playing ? '暫停（空白鍵）' : '播放（空白鍵）'}>
            {playing ? <I.pause /> : <I.play />}
          </button>
          <button className="tbtn" onClick={stop} aria-label="停止" title="停止"><I.stop /></button>
          <button className="tbtn" onClick={() => stepBar(1)} aria-label="下一小節" title="下一小節"><I.skipFwd /></button>
          <button className={`tbtn ${loop ? 'on' : ''}`} onClick={toggleLoop} aria-pressed={loop} aria-label="循環" title="循環（L）"><I.loop /></button>
        </div>

        <div className="lcd" aria-live="off">
          <div className="lcd-cell"><span className="lcd-k">小節</span><span className="lcd-v">{barTotal ? barIdx + 1 : '–'}<small>/{barTotal || '–'}</small></span></div>
          <div className="lcd-cell wide"><span className="lcd-k">時間</span>
            <span className="lcd-v">{fmtPrecise(time)}{endSec != null && <small>/{fmtTime(endSec)}</small>}</span></div>
          <div className="lcd-cell"><span className="lcd-k">速度</span>
            <span className="lcd-v"><NoteGlyph value={1} size={13} />{tempo ? Math.round(tempo) : '–'}</span></div>
          <div className="lcd-cell"><span className="lcd-k">拍號</span><span className="lcd-v">{meter ? `${meter[0]}/${meter[1]}` : '–'}</span></div>
          {isRef ? (
            <div className="lcd-cell note"><I.lock size={13} /><span>參考譜唯讀，以合成音播放，不與錄音同步</span></div>
          ) : (
            <div className="lcd-cell"><span className="lcd-k">調</span><span className="lcd-v">{summary?.key?.name ?? '–'}</span></div>
          )}
          {rangeBars && (
            <div className="lcd-cell range">
              <span className="lcd-k">範圍</span>
              <span className="lcd-v">{rangeBars[0] === rangeBars[1] ? rangeBars[0] : `${rangeBars[0]}–${rangeBars[1]}`}</span>
              <button className="lcd-x" onClick={clearRange} aria-label="清除範圍" title="清除範圍"><I.close size={12} /></button>
            </div>
          )}
        </div>

        <div className="tgroup">
          <label className="tfield" title="播放來源">
            <I.headphones />
            <select aria-label="播放來源" value={source} disabled={isRef} onChange={(e) => changeSource(e.target.value)}>
              {!isRef && <option value="input">原曲</option>}
              {!isRef && stemNames.map((n) => <option key={n} value={n}>{sourceLabel(n)}</option>)}
              <option value="synth">{playerReady ? '樂譜合成音' : '合成音載入中'}</option>
            </select>
          </label>
          <select className="tselect" aria-label="播放速度" title="播放速度" value={speed} onChange={(e) => changeSpeed(Number(e.target.value))}>
            {SPEEDS.map((s) => <option key={s} value={s}>{Math.round(s * 100)}%</option>)}
          </select>
        </div>

        <span className="spacer" />

        {!isRef && (
          <div className="tgroup">
            <span className={`lowcount ${shownLows.length ? '' : 'none'}`} title="低信心音（橘色）"><I.warn />{shownLows.length}</span>
            <button className="tbtn" disabled={!shownLows.length} onClick={() => jumpLow(-1)} aria-label="上一個低信心音" title="上一個低信心音"><I.left /></button>
            <button className="tbtn" disabled={!shownLows.length} onClick={() => jumpLow(1)} aria-label="下一個低信心音" title="下一個低信心音"><I.right /></button>
          </div>
        )}
        <div className="tgroup">
          <div className="seg" role="group" aria-label="譜表">
            {STAFF.map((s) => (
              <button key={s.id} className={staff === s.id ? 'on' : ''} aria-pressed={staff === s.id}
                onClick={() => applyDisplay(s.id, zoom)}>{s.label}</button>
            ))}
          </div>
          <div className="zoom">
            <button className="tbtn" onClick={() => zoomStep(-1)} disabled={zoom <= ZOOMS[0]} aria-label="縮小" title="縮小"><I.zoomOut /></button>
            <span className="mono">{Math.round(zoom * 100)}%</span>
            <button className="tbtn" onClick={() => zoomStep(1)} disabled={zoom >= ZOOMS[ZOOMS.length - 1]} aria-label="放大" title="放大"><I.zoomIn /></button>
          </div>
        </div>
        <div className="tgroup">
          <button className={`tbtn text ${panel === 'edit' ? 'on' : ''}`} aria-pressed={panel === 'edit'} disabled={isRef}
            onClick={() => togglePanel('edit')} title={isRef ? '參考譜不可編輯' : '編輯吉他音符'}>
            <I.edit />改音{corrDirty > 0 && <span className="count">{corrDirty}</span>}
          </button>
          <button className={`tbtn text ${panel === 'align' ? 'on' : ''}`} aria-pressed={panel === 'align'} disabled={isRef}
            onClick={() => togglePanel('align')} title={isRef ? '參考譜不與錄音同步' : '樂譜與錄音對齊'}>
            <I.align />對齊{(dirty || saved?.stale || alignError) && <span className={`dot ${alignError || saved?.stale ? 'bad' : ''}`} />}
          </button>
          <a className="tbtn" href={scoreUrl ?? undefined} download={isRef ? reference?.filename : view === 'fixed' ? 'corrected.gp' : gpFile}
            aria-label="下載目前的樂譜" title="下載目前的樂譜"><I.download /></a>
        </div>
      </div>

      {!isRef && (
        <Timeline wave={project.waveform} duration={duration} barSecs={barSecs} lows={lowSecs} anchors={anchorSecs}
          range={rangeSecs} span={editSpan} time={time} onSeek={seekSec} onSelect={setRangeFromSeconds} />
      )}

      <audio ref={audio} src={audioSrc} preload="auto"
        onLoadedMetadata={(e) => {
          const el = e.currentTarget;
          el.playbackRate = speed; el.defaultPlaybackRate = speed;
          (el as any).preservesPitch = true;
          el.currentTime = Math.min(pendingSeek.current, el.duration || pendingSeek.current);
        }}
        onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)}
        onEnded={(e) => {
          if (loopRef.current && !rangeRef.current) { e.currentTarget.currentTime = 0; void e.currentTarget.play(); }
          else setPlaying(false);
        }} />

      <div className={`ws-body ${panel ? 'has-insp' : ''}`}>
        <aside className="rail" aria-label="樂譜與軌道">
          <div className="rail-h">樂譜</div>
          <button className={`rail-row ${view === 'tx' ? 'sel' : ''}`} aria-pressed={view === 'tx'} onClick={() => switchView('tx')}>
            <I.score /><span className="rr-main"><span className="rr-name">轉譜結果</span><span className="rr-meta">{gpFile}</span></span>
          </button>
          {fixedPath && (
            <button className={`rail-row ${view === 'fixed' ? 'sel' : ''}`} aria-pressed={view === 'fixed'} onClick={() => switchView('fixed')}>
              <I.corrected />
              <span className="rr-main"><span className="rr-name">修正版</span>
                <span className={`rr-meta ${fixedStale ? 'warn' : ''}`}>{fixedStale ? '有新儲存的修改尚未匯出' : '最近一次匯出'}</span></span>
            </button>
          )}
          {reference ? (
            <button className={`rail-row ${isRef ? 'sel' : ''}`} aria-pressed={isRef} onClick={() => switchView('ref')} title={reference.filename}>
              <I.book />
              <span className="rr-main"><span className="rr-name">參考譜</span><span className="rr-meta">{reference.filename}</span></span>
              <I.lock size={12} className="rr-end" />
            </button>
          ) : (
            <button className="rail-row add" disabled={attaching} onClick={() => refInput.current?.click()}>
              <I.plus /><span className="rr-main"><span className="rr-name">{attaching ? '上傳中…' : '附加參考譜'}</span>
                <span className="rr-meta">.gp .gp3 .gp4 .gp5 .gpx</span></span>
            </button>
          )}
          <input ref={refInput} type="file" accept=".gp,.gp3,.gp4,.gp5,.gpx" hidden onChange={(e) => void attach(e.target.files?.[0])} />
          {attachError && <div className="msg bad rail-msg" role="alert"><I.error />{attachError}</div>}

          <div className="rail-h">軌道</div>
          <button className={`rail-row ${visible === null ? 'sel' : ''}`} aria-pressed={visible === null} onClick={() => show(null)}>
            <I.layers /><span className="rr-main"><span className="rr-name">全部軌道</span><span className="rr-meta">{tracks.length} 軌</span></span>
          </button>
          {tracks.map((t) => {
            const sel = visible?.length === 1 && visible[0] === t.index;
            const low = trackLow(t.name);
            return (
              <button key={t.index} className={`rail-row ${sel ? 'sel' : ''} ${visible && visible.length > 1 && visible.includes(t.index) ? 'part' : ''}`}
                aria-pressed={sel} onClick={() => show([t.index])} title={t.name}>
                {familyIcon(t.family)}
                <span className="rr-main"><span className="rr-name">{t.name}</span><span className="rr-meta">{trackMeta(t)}</span></span>
                {low != null && <span className="rr-low" title="低信心音比例">{low}%</span>}
              </button>
            );
          })}
        </aside>

        <div className="stage">
          {loadError && <div className="msg bad stage-msg" role="alert"><I.error />{loadError}</div>}
          {!ready && !loadError && <div className="stage-busy" role="status">繪製中…</div>}
          <div className="scorewrap" ref={wrap}><div className="page" ref={host} /></div>
        </div>

        <aside className="insp" hidden={!panel} aria-label={panel === 'edit' ? '改音' : '對齊'}>
          <div className="insp-tabs" role="tablist">
            <button role="tab" aria-selected={panel === 'edit'} className={panel === 'edit' ? 'on' : ''} onClick={() => setPanel('edit')}>
              <I.edit />改音
            </button>
            <button role="tab" aria-selected={panel === 'align'} className={panel === 'align' ? 'on' : ''} onClick={() => setPanel('align')}>
              <I.align />對齊
            </button>
            <span className="spacer" />
            <button className="tbtn" onClick={() => setPanel(null)} aria-label="關閉面板" title="關閉"><I.close /></button>
          </div>
          <div className="insp-body" hidden={panel !== 'edit'}>
            <CorrectionPanel key={`${pid}|${JSON.stringify(project.manifest?.stages?.export ?? null)}`}
              projectId={pid} active={panel === 'edit'} guitars={guitars}
              trackId={corrTrack ?? guitars[0]?.id ?? null} setTrackId={setCorrTrack} bar={corrBar} setBar={setCorrBar}
              bars={bars} playheadBar={map ? barAtSec(map, time) : null} focus={focus}
              onListen={listenBar} onDirty={setCorrDirty} onSaved={() => setFixedStale(true)}
              onExported={(path) => { setFixedPath(path); setFixedStale(false); setFixedRev((n) => n + 1); }} />
          </div>
          <div className="insp-body" hidden={panel !== 'align'}>
            <div className="al">
              <div className="al-state">
                {anchors.length === 0
                  ? <>依節拍偵測推估{offset ? `，整體位移 ${fmtDelta(offset)}` : ''}</>
                  : <>{anchors.length} 個錨點，其餘位置依節奏推估</>}
              </div>
              {saved?.stale && <div className="msg warn"><I.warn />樂譜重新產生過，舊對齊已失效。</div>}
              {alignError && (
                <div className="msg bad" role="alert"><I.error />{alignError}
                  <button className="btn link" onClick={loadAlignment}>重新載入</button></div>
              )}
              {alignNote && <div className="msg warn" role="status"><I.warn />{alignNote}</div>}

              <div className="kv">
                <span className="kv-k">選取的拍</span>
                {selTick != null && map ? (
                  <span className="kv-v">{describeTick(map, selTick)}
                    <small>{selAnchor ? `錨點 ${fmtPrecise(selAnchor.seconds)}` : `推估 ${fmtPrecise(selPredicted!)}`}</small></span>
                ) : <span className="kv-v dim">在樂譜上點選</span>}
              </div>
              <div className="kv">
                <span className="kv-k">錄音位置</span>
                {source === 'synth' ? <span className="kv-v dim">請把播放來源改為原曲或分軌</span> : (
                  <span className="kv-v mono">{fmtPrecise(time)}
                    {selPredicted != null && !selAnchor && <small>{fmtDelta(time - selPredicted)}</small>}</span>
                )}
              </div>
              {source !== 'synth' && (
                <div className="seg nudges" role="group" aria-label="微調錄音位置">
                  {NUDGES.map((d) => (
                    <button key={d} onClick={() => seekSec(time + d, false)} title="只移動錄音，不移動樂譜游標">
                      {d > 0 ? '+' : '−'}{Math.abs(d)}
                    </button>
                  ))}
                </div>
              )}
              <button className="btn primary block" disabled={!canAnchor} onClick={addAnchor} title="A 鍵">
                <I.anchor />{selAnchor ? '更新錨點' : '設為錨點'}<kbd>A</kbd>
              </button>
              <div className="kv">
                <span className="kv-k">整體位移</span>
                <span className="kv-v">
                  <input type="number" step={0.01} value={offset} disabled={anchors.length > 0 || !saved} aria-label="整體位移（秒）"
                    title={anchors.length > 0 ? '有錨點時以錨點為準' : undefined}
                    onChange={(e) => changeOffset(Number(e.target.value))} /> 秒
                </span>
              </div>

              {anchors.length > 0 && map && plainMap && (
                <table className="al-list">
                  <thead><tr><th>位置</th><th>錄音</th><th title="錨點和節奏推估的差">修正</th><th /></tr></thead>
                  <tbody>
                    {anchors.map((a) => (
                      <tr key={a.beat}>
                        <td>{describeTick(map, a.beat * TICKS_PER_BEAT)}</td>
                        <td className="mono">{fmtPrecise(a.seconds)}</td>
                        <td className="mono dim">{fmtDelta(a.seconds - tickToSec(plainMap, a.beat * TICKS_PER_BEAT))}</td>
                        <td className="acts">
                          <button className="tbtn sm" onClick={() => jumpAnchor(a)} aria-label="跳到錨點" title="跳到"><I.target size={14} /></button>
                          <button className="tbtn sm" onClick={() => removeAnchor(a.beat)} aria-label="刪除錨點" title="刪除"><I.trash size={14} /></button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}

              <div className="ne-foot">
                <span className={`ne-status ${dirty ? 'dirty' : ''}`} role="status">
                  {saving ? '儲存中…' : dirty ? '未儲存' : saved ? '已同步' : ''}
                </span>
                <button className="btn" disabled={!dirty || saving} onClick={revertAlignment}><I.undo />還原</button>
                <button className="btn primary" disabled={(!dirty && !saved?.stale) || saving || !saved} onClick={saveAlignment}><I.save />儲存</button>
              </div>
            </div>
          </div>
        </aside>
      </div>

      <div className="statusbar">
        {isRef ? (
          <span>{reference?.filename}</span>
        ) : (
          <>
            {project.analysis?.global_bpm != null && <span>平均 {project.analysis.global_bpm} BPM</span>}
            <span>{bars.length} 小節</span>
            {summary?.key && <span>{summary.key.name}</span>}
            {shownLows.length > 0 && <span className="low"><i />低信心</span>}
          </>
        )}
        <span className="spacer" />
        <span className="keys"><kbd>Space</kbd>播放<kbd>L</kbd>循環{panel === 'align' && <><kbd>A</kbd>錨點</>}</span>
      </div>
    </div>
  );
}

function Timeline({ wave, duration, barSecs, lows, anchors, range, span, time, onSeek, onSelect }: {
  wave: Waveform; duration: number; barSecs: number[]; lows: number[]; anchors: number[];
  range: [number, number] | null; span: [number, number] | null; time: number;
  onSeek: (t: number) => void; onSelect: (a: number, b: number) => void;
}) {
  const box = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const drag = useRef<{ x0: number; t0: number; moved: boolean } | null>(null);
  const [draft, setDraft] = useState<[number, number] | null>(null);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const c = canvas.current;
    if (!c || !width) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = width * dpr;
    c.height = c.clientHeight * dpr;
    const g = c.getContext('2d')!;
    const css = getComputedStyle(c);
    const col = (name: string, fallback: string) => css.getPropertyValue(name).trim() || fallback;
    const W = c.width, H = c.height, RULER = 15 * dpr, LOWH = 3 * dpr;
    g.clearRect(0, 0, W, H);
    const x = (t: number) => (t / Math.max(duration, 1e-6)) * W;
    if (barSecs.length) {
      const px = W / barSecs.length;
      const step = [1, 2, 4, 8, 16, 32].find((s) => s * px >= 6 * dpr) ?? 64;
      const label = [1, 2, 4, 8, 16, 32, 64].find((s) => s * px >= 30 * dpr) ?? 128;
      g.fillStyle = col('--line', '#d6dae1');
      barSecs.forEach((t, i) => { if (i % step === 0) g.fillRect(Math.round(x(t)), RULER, 1, H - RULER); });
      g.fillStyle = col('--faint', '#8b93a1');
      g.font = `${10 * dpr}px "Segoe UI", system-ui, sans-serif`;
      g.textBaseline = 'middle';
      barSecs.forEach((t, i) => {
        if (i % label) return;
        g.fillRect(Math.round(x(t)), RULER - 5 * dpr, 1, 5 * dpr);
        g.fillText(String(i + 1), Math.round(x(t)) + 3 * dpr, RULER / 2);
      });
    }
    g.fillStyle = col('--line', '#d6dae1');
    g.fillRect(0, RULER - 1, W, 1);
    drawWave(g, wave, 0, RULER + 2 * dpr, W, H - RULER - LOWH - 4 * dpr, col('--wave', '#8a95a8'));
    g.fillStyle = col('--low', '#c96a00');
    for (const t of lows) g.fillRect(Math.round(x(t)), H - LOWH, Math.max(1, dpr * 2), LOWH);
    g.fillStyle = col('--accent', '#1f5fd1');
    for (const t of anchors) {
      const ax = x(t);
      g.beginPath(); g.moveTo(ax - 4 * dpr, RULER); g.lineTo(ax + 4 * dpr, RULER); g.lineTo(ax, RULER + 6 * dpr); g.fill();
      g.fillRect(Math.round(ax), RULER, Math.max(1, dpr), H - RULER);
    }
  }, [wave, duration, barSecs, lows, anchors, width]);

  const timeAt = (clientX: number) => {
    const r = box.current!.getBoundingClientRect();
    return Math.max(0, Math.min(1, (clientX - r.left) / r.width)) * duration;
  };
  const shown = draft ?? range;
  const pct = (t: number) => `${(t / Math.max(duration, 1e-6)) * 100}%`;

  return (
    <div className="timeline" ref={box} role="slider" aria-label="播放位置" tabIndex={0}
      aria-valuemin={0} aria-valuemax={Math.round(duration)} aria-valuenow={Math.round(time)} aria-valuetext={fmtTime(time)}
      onKeyDown={(e) => {
        if (e.key === 'ArrowRight') { e.preventDefault(); onSeek(time + 5); }
        else if (e.key === 'ArrowLeft') { e.preventDefault(); onSeek(time - 5); }
      }}
      onPointerDown={(e) => {
        e.currentTarget.setPointerCapture(e.pointerId);
        drag.current = { x0: e.clientX, t0: timeAt(e.clientX), moved: false };
      }}
      onPointerMove={(e) => {
        const d = drag.current;
        if (!d) return;
        if (!d.moved && Math.abs(e.clientX - d.x0) < 5) return;
        d.moved = true;
        const t = timeAt(e.clientX);
        setDraft([Math.min(d.t0, t), Math.max(d.t0, t)]);
      }}
      onPointerUp={(e) => {
        const d = drag.current;
        drag.current = null;
        setDraft(null);
        if (!d) return;
        if (d.moved) onSelect(d.t0, timeAt(e.clientX));
        else onSeek(d.t0);
      }}
      onPointerCancel={() => { drag.current = null; setDraft(null); }}>
      <canvas ref={canvas} />
      {span && <div className="tl-span" style={{ left: pct(span[0]), width: pct(span[1] - span[0]) }} />}
      {shown && <div className="tl-range" style={{ left: pct(shown[0]), width: pct(shown[1] - shown[0]) }} />}
      <div className="tl-head" style={{ left: pct(time) }} />
    </div>
  );
}

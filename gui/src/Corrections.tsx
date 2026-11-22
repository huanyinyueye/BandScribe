import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api, BarInfo, CorrectionBar, CorrectionNote, CorrectionSave, ScoreSummary } from './api';
import { DotGlyph, I, NoteGlyph, TripletGlyph } from './icons';

// All positions and durations are integer ticks, 960 per quarter note, relative to the bar start.
const TPQ = 960;
const WHOLE = TPQ * 4;
type Draft = { string: number; fret: number; pos: number; dur: number };
type Added = Draft & { id: number };
type BarEdit = { changed: Record<number, Draft>; deleted: number[]; added: Added[] };
type Item = { key: string; index: number | null; addId: number | null; orig: Draft | null; v: Draft };
export type EditorFocus = { track: string; bar: number; tick: number; seq: number };

const NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const pc = (p: number) => NAMES[((p % 12) + 12) % 12];
const noteName = (p: number) => `${pc(p)}${Math.floor(p / 12) - 1}`;
const gcd = (a: number, b: number): number => (b ? gcd(b, a % b) : a);
const ratio = (num: number, den: number) => {
  const g = gcd(Math.abs(num), den) || 1;
  return den / g === 1 ? `${num / g}` : `${num / g}/${den / g}`;
};
const toTicks = (q: number) => Math.round(q * TPQ);
const keyOf = (track: string, bar: number) => `${track}/${bar}`;
const parseKey = (k: string) => ({ track: k.slice(0, k.lastIndexOf('/')), bar: Number(k.slice(k.lastIndexOf('/') + 1)) });
const msg = (e: any) => String(e?.message ?? e);
const sameDraft = (a: Draft, b: Draft) => a.string === b.string && a.fret === b.fret && a.pos === b.pos && a.dur === b.dur;
const isEmpty = (e: BarEdit | undefined) => !e || (!Object.keys(e.changed).length && !e.deleted.length && !e.added.length);
const EMPTY: BarEdit = { changed: {}, deleted: [], added: [] };

const DUR_BASES: { ticks: number; glyph: number; label: string }[] = [
  { ticks: WHOLE, glyph: 4, label: '全音符' }, { ticks: WHOLE / 2, glyph: 2, label: '二分音符' },
  { ticks: WHOLE / 4, glyph: 1, label: '四分音符' }, { ticks: WHOLE / 8, glyph: 0.5, label: '八分音符' },
  { ticks: WHOLE / 16, glyph: 0.25, label: '十六分音符' }, { ticks: WHOLE / 32, glyph: 0.125, label: '三十二分音符' },
];
const GRIDS = [
  { ticks: 480, label: '1/8' }, { ticks: 240, label: '1/16' }, { ticks: 120, label: '1/32' },
  { ticks: 320, label: '1/8 三連' }, { ticks: 160, label: '1/16 三連' },
];
type Mod = 'plain' | 'dot' | 'trip';
const applyMod = (base: number, mod: Mod) => (mod === 'dot' ? base * 1.5 : mod === 'trip' ? (base * 2) / 3 : base);
function splitDur(dur: number): { base: number; mod: Mod } | null {
  for (const b of DUR_BASES) for (const mod of ['plain', 'dot', 'trip'] as Mod[]) {
    if (Math.round(applyMod(b.ticks, mod)) === dur) return { base: b.ticks, mod };
  }
  return null;
}

const toDraft = (n: CorrectionNote): Draft => ({ string: n.string ?? 0, fret: n.fret ?? 0, pos: toTicks(n.pos_q), dur: toTicks(n.dur_q) });

// Mini TAB geometry (SVG user units).
const VW = 340, GUT = 26, PADR = 8, TOP = 20, SP = 20;
const innerW = VW - GUT - PADR;

export function CorrectionPanel({ projectId, active, guitars, trackId, setTrackId, bar, setBar, bars,
  playheadBar, focus, onListen, onDirty, onSaved, onExported }: {
  projectId: string; active: boolean; guitars: ScoreSummary['tracks'];
  trackId: string | null; setTrackId: (id: string) => void; bar: number; setBar: (b: number) => void;
  bars: BarInfo[]; playheadBar: number | null; focus: EditorFocus | null;
  onListen: (bar: number) => void; onDirty: (bars: number) => void; onSaved: () => void; onExported: (path: string) => void;
}) {
  const [cache, setCache] = useState<Record<string, CorrectionBar>>({});
  const [edits, setEdits] = useState<Record<string, BarEdit>>({});
  const [pending, setPending] = useState<Record<string, boolean>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [exported, setExported] = useState<string | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);
  const [sel, setSel] = useState<string | null>(null);
  const [cursor, setCursor] = useState<{ string: number; pos: number } | null>(null);
  const [grid, setGrid] = useState(240);
  const [lastDur, setLastDur] = useState(480);
  const nextId = useRef(1);
  const typed = useRef<{ key: string; value: number; at: number } | null>(null);
  const svg = useRef<SVGSVGElement>(null);

  const key = trackId != null ? keyOf(trackId, bar) : null;
  const data = key ? cache[key] : undefined;
  const edit = (key && edits[key]) || EMPTY;
  const dirtyKeys = Object.keys(edits).filter((k) => !isEmpty(edits[k]));
  const info = bars[bar];
  const len = info ? (info.numerator * WHOLE) / info.denominator : 0;
  const beatUnit = info ? WHOLE / info.denominator : TPQ;
  const tuning = data?.tuning ?? [];
  const nStrings = tuning.length;

  useEffect(() => onDirty(dirtyKeys.length), [dirtyKeys.length]);  // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!dirtyKeys.length) return;
    const warn = (e: BeforeUnloadEvent) => { e.preventDefault(); e.returnValue = ''; };
    window.addEventListener('beforeunload', warn);
    return () => window.removeEventListener('beforeunload', warn);
  }, [dirtyKeys.length]);
  useEffect(() => { setSel(null); setCursor(null); typed.current = null; }, [key]);

  const setErr = (k: string, e: string | null) => setErrors((all) => {
    const next = { ...all };
    if (e) next[k] = e; else delete next[k];
    return next;
  });

  const load = useCallback((track: string, b: number) => {
    const k = keyOf(track, b);
    setPending((p) => ({ ...p, [k]: true }));
    setErr(k, null);
    api.corrections(projectId, track, b)
      .then((r) => {
        setCache((c) => ({ ...c, [k]: r }));
        // Keep edits across reloads, but only for notes that still exist.
        setEdits((all) => {
          const e = all[k];
          if (!e) return all;
          const live = new Set(r.notes.map((n) => n.index));
          const changed = Object.fromEntries(Object.entries(e.changed).filter(([i]) => live.has(Number(i))));
          return { ...all, [k]: { changed, deleted: e.deleted.filter((i) => live.has(i)), added: e.added } };
        });
      })
      .catch((e) => setErr(k, `讀取失敗：${msg(e)}`))
      .finally(() => setPending((p) => { const n = { ...p }; delete n[k]; return n; }));
  }, [projectId]);

  useEffect(() => {
    if (active && trackId != null && key && !cache[key] && !pending[key] && !errors[key]) load(trackId, bar);
  }, [active, key]);  // eslint-disable-line react-hooks/exhaustive-deps

  const items: Item[] = useMemo(() => {
    if (!data) return [];
    const out: Item[] = data.notes.filter((n) => !edit.deleted.includes(n.index)).map((n) => {
      const orig = toDraft(n);
      return { key: `n${n.index}`, index: n.index, addId: null, orig, v: edit.changed[n.index] ?? orig };
    });
    for (const a of edit.added) out.push({ key: `a${a.id}`, index: null, addId: a.id, orig: null, v: a });
    return out.sort((a, b) => a.v.pos - b.v.pos || a.v.string - b.v.string);
  }, [data, edit]);

  // Clicking a beat in the score selects its highest note here.
  useEffect(() => {
    if (!focus || !data || focus.track !== trackId || focus.bar !== bar) return;
    const hit = items.filter((it) => Math.abs(it.v.pos - focus.tick) <= 5).sort((a, b) => a.v.string - b.v.string)[0];
    setSel(hit?.key ?? null);
    if (!hit) setCursor({ string: 1, pos: Math.max(0, Math.min(len - grid, Math.round(focus.tick / grid) * grid)) });
  }, [focus?.seq, data]);  // eslint-disable-line react-hooks/exhaustive-deps

  const current = items.find((it) => it.key === sel) ?? null;
  const pitchOf = (d: Draft) => (d.string >= 1 && d.string <= nStrings ? tuning[d.string - 1] + d.fret : null);

  const problems = useMemo(() => {
    const out = new Map<string, string>();
    const seen = new Map<string, string>();
    for (const it of items) {
      const d = it.v, p = pitchOf(d);
      const touched = !it.orig || !sameDraft(it.orig, d);
      if (touched) {
        if (d.string < 1 || d.string > nStrings) out.set(it.key, '請選擇弦');
        else if (!Number.isInteger(d.fret) || d.fret < 0 || d.fret > 36) out.set(it.key, '格數需為 0–36 的整數');
        else if (p != null && p > 127) out.set(it.key, '音高超出範圍');
        else if (d.dur <= 0 || d.pos < 0 || d.pos + d.dur > len) out.set(it.key, '音符超出這一小節');
      }
      const slot = `${d.string}@${d.pos}`;
      if (seen.has(slot)) { out.set(it.key, '同一弦同一時間有兩個音'); out.set(seen.get(slot)!, '同一弦同一時間有兩個音'); }
      seen.set(slot, it.key);
    }
    return out;
  }, [items, nStrings, len]);  // eslint-disable-line react-hooks/exhaustive-deps

  const putEdit = (fn: (e: BarEdit) => BarEdit) => {
    if (!key) return;
    setEdits((all) => ({ ...all, [key]: fn(all[key] ?? EMPTY) }));
  };
  const update = (it: Item, patch: Partial<Draft>) => {
    const v = { ...it.v, ...patch };
    if (patch.dur) setLastDur(patch.dur);
    putEdit((e) => {
      if (it.addId != null) return { ...e, added: e.added.map((a) => (a.id === it.addId ? { ...v, id: a.id } : a)) };
      const changed = { ...e.changed };
      if (it.orig && sameDraft(it.orig, v)) delete changed[it.index!]; else changed[it.index!] = v;
      return { ...e, changed };
    });
  };
  const remove = (it: Item) => {
    const order = items.findIndex((x) => x.key === it.key);
    putEdit((e) => {
      if (it.addId != null) return { ...e, added: e.added.filter((a) => a.id !== it.addId) };
      const changed = { ...e.changed };
      delete changed[it.index!];
      return { ...e, changed, deleted: [...e.deleted, it.index!] };
    });
    const next = items[order + 1] ?? items[order - 1];
    setSel(next && next.key !== it.key ? next.key : null);
    setCursor({ string: it.v.string, pos: it.v.pos });
  };
  const add = (at: { string: number; pos: number }, fret = 0) => {
    if (at.pos >= len) return;
    const id = nextId.current++;
    const dur = Math.min(lastDur, len - at.pos);
    putEdit((e) => ({ ...e, added: [...e.added, { id, string: at.string, fret, pos: at.pos, dur }] }));
    setSel(`a${id}`);
    setCursor(null);
  };
  // Moving to another string keeps the pitch: the usual fix for a wrong TAB position.
  const fretOn = (it: Item, s: number) => {
    const p = pitchOf(it.v);
    return p == null ? it.v.fret : p - tuning[s - 1];
  };
  const moveString = (it: Item, s: number) => {
    if (s < 1 || s > nStrings) return;
    const f = fretOn(it, s);
    if (f >= 0 && f <= 36) update(it, { string: s, fret: f });
  };
  const nudge = (it: Item, dir: 1 | -1) => {
    const pos = Math.max(0, Math.min(len - it.v.dur, it.v.pos + dir * grid));
    if (pos !== it.v.pos) update(it, { pos });
  };
  const revertBar = () => {
    if (!key) return;
    setEdits((all) => { const next = { ...all }; delete next[key]; return next; });
    setSel(null);
  };

  // GP-style keys: digits type the fret, arrows walk notes / move strings, Delete removes.
  useEffect(() => {
    if (!active || !data) return;
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (['INPUT', 'SELECT', 'TEXTAREA'].includes(t.tagName) || t.closest('[role=slider]'))) return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const at = current;
      if (/^[0-9]$/.test(e.key)) {
        e.preventDefault();
        const d = Number(e.key), now = performance.now(), prev = typed.current;
        if (at) {
          const v = prev && prev.key === at.key && now - prev.at < 700 && prev.value * 10 + d <= 36 ? prev.value * 10 + d : d;
          typed.current = { key: at.key, value: v, at: now };
          update(at, { fret: v });
        } else if (cursor) {
          add(cursor, d);
        }
        return;
      }
      const order = at ? items.findIndex((x) => x.key === at.key) : -1;
      switch (e.key) {
        case 'ArrowLeft': case 'ArrowRight': {
          e.preventDefault();
          const dir = e.key === 'ArrowLeft' ? -1 : 1;
          if (at && e.shiftKey) { nudge(at, dir); return; }
          const next = items[order < 0 ? (dir > 0 ? 0 : items.length - 1) : order + dir];
          if (next) { setSel(next.key); setCursor(null); }
          return;
        }
        case 'ArrowUp': case 'ArrowDown': {
          if (!at) return;
          e.preventDefault();
          const dir = e.key === 'ArrowUp' ? -1 : 1;
          if (e.shiftKey) update(at, { fret: Math.max(0, Math.min(36, at.v.fret - dir)) });
          else moveString(at, at.v.string + dir);
          return;
        }
        case 'Delete': case 'Backspace':
          if (at) { e.preventDefault(); remove(at); }
          return;
        case 'Enter':
          if (!at && cursor) { e.preventDefault(); add(cursor); }
          return;
        case 'Escape':
          setSel(null); setCursor(null);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  const save = async () => {
    if (!data || !key || trackId == null) return;
    const k = key, e = edits[k] ?? EMPTY;
    const byIndex = new Map(data.notes.map((n) => [n.index, toDraft(n)]));
    const body: CorrectionSave = {
      score_hash: data.score_hash, revision: data.revision,
      notes: Object.entries(e.changed).map(([i, d]) => {
        const o = byIndex.get(Number(i))!;
        return {
          index: Number(i), string: d.string, fret: d.fret,
          ...(d.pos !== o.pos ? { pos_q: ratio(d.pos, TPQ) } : {}),
          ...(d.dur !== o.dur ? { dur_q: ratio(d.dur, TPQ) } : {}),
        };
      }),
      delete: e.deleted,
      add: e.added.map((a) => ({ string: a.string, fret: a.fret, pos_q: ratio(a.pos, TPQ), dur_q: ratio(a.dur, TPQ) })),
    };
    setSaving(true);
    setErr(k, null);
    try {
      const r = await api.saveCorrections(projectId, trackId, bar, body);
      // Revision is per score, note indices per track: later bars of this track shift by the net insert count.
      const delta = e.added.length - e.deleted.length;
      const later = (ck: string) => { const p = parseKey(ck); return p.track === trackId && p.bar > bar; };
      const shift = (i: number) => i + delta;
      setCache((c) => {
        const next: Record<string, CorrectionBar> = {};
        for (const [ck, v] of Object.entries(c)) {
          if (v.score_hash !== r.score_hash) { next[ck] = v; continue; }
          next[ck] = { ...v, revision: r.revision,
            notes: delta && later(ck) ? v.notes.map((n) => ({ ...n, index: shift(n.index) })) : v.notes };
        }
        next[k] = r;
        return next;
      });
      setEdits((all) => {
        const next: Record<string, BarEdit> = {};
        for (const [ek, v] of Object.entries(all)) {
          if (ek === k) continue;
          next[ek] = delta && later(ek) ? {
            changed: Object.fromEntries(Object.entries(v.changed).map(([i, d]) => [shift(Number(i)), d])),
            deleted: v.deleted.map(shift), added: v.added,
          } : v;
        }
        return next;
      });
      setSel(null);
      setExported(null);
      onSaved();
    } catch (err) {
      setErr(k, `儲存失敗：${msg(err)}`);
    } finally {
      setSaving(false);
    }
  };

  const doExport = async () => {
    setExporting(true);
    setExportError(null);
    try {
      const path = (await api.exportCorrections(projectId)).path;
      setExported(path);
      onExported(path);
    } catch (e) { setExportError(`匯出失敗：${msg(e)}`); }
    finally { setExporting(false); }
  };
  const exportHref = exported ? api.fileUrl(projectId, exported.split('/').map(encodeURIComponent).join('/')) : null;

  if (!guitars.length) return <div className="ne-empty">這份樂譜沒有可編輯的吉他 TAB 軌。</div>;

  const err = key ? errors[key] : undefined;
  const loading = key ? !!pending[key] : false;
  const trackName = (id: string) => guitars.find((t) => t.id === id)?.name ?? id;
  const others = dirtyKeys.filter((k) => k !== key);
  const changes = Object.keys(edit.changed).length + edit.deleted.length + edit.added.length;
  const firstProblem = problems.size ? [...problems.values()][0] : null;

  // --- mini TAB ---
  const H = TOP + Math.max(0, nStrings - 1) * SP + 16;
  const X = (pos: number) => GUT + (len ? (pos / len) * innerW : 0);
  const Y = (s: number) => TOP + (s - 1) * SP;
  const pointAt = (clientX: number, clientY: number) => {
    const r = svg.current!.getBoundingClientRect();
    const vx = ((clientX - r.left) / r.width) * VW, vy = ((clientY - r.top) / r.height) * H;
    const string = Math.max(1, Math.min(nStrings, Math.round((vy - TOP) / SP) + 1));
    const raw = ((vx - GUT) / innerW) * len;
    const pos = Math.max(0, Math.min(len - grid, Math.floor((raw + grid * 0.35) / grid) * grid));
    return { string, pos };
  };
  const beatLines: number[] = [];
  for (let t = 0; t < len; t += beatUnit) beatLines.push(t);
  const subLines: number[] = [];
  for (let t = grid; t < len; t += grid) if (t % beatUnit) subLines.push(t);

  const posLabel = (pos: number) => {
    const beat = Math.floor(pos / beatUnit) + 1, rem = pos % beatUnit;
    return rem ? `${beat} 拍 + ${ratio(rem, WHOLE)}` : `${beat} 拍`;
  };
  const durParts = current ? splitDur(current.v.dur) : null;
  const setDur = (base: number, mod: Mod) => current && update(current, { dur: Math.round(applyMod(base, mod)) });
  const curPitch = current ? pitchOf(current.v) : null;
  const origPitch = current?.orig ? pitchOf(current.orig) : null;

  return (
    <div className="ne" aria-label="改音">
      <div className="ne-nav">
        {guitars.length > 1 && (
          <select aria-label="軌道" value={trackId ?? ''} onChange={(e) => setTrackId(e.target.value)}>
            {guitars.map((t) => <option key={t.id} value={t.id}>{t.name}</option>)}
          </select>
        )}
        <div className="ne-bar">
          <button className="tbtn" disabled={bar <= 0} onClick={() => setBar(bar - 1)} aria-label="上一小節" title="上一小節"><I.left /></button>
          <span className="ne-barnum"><b>{bar + 1}</b><span>/ {bars.length}</span></span>
          <button className="tbtn" disabled={bar >= bars.length - 1} onClick={() => setBar(bar + 1)} aria-label="下一小節" title="下一小節"><I.right /></button>
        </div>
        <span className="spacer" />
        <button className="tbtn" disabled={playheadBar == null || playheadBar === bar} title="跳到播放位置的小節"
          aria-label="跳到播放位置的小節" onClick={() => playheadBar != null && setBar(playheadBar)}><I.target /></button>
        <button className="tbtn" onClick={() => onListen(bar)} title="循環播放這一小節" aria-label="循環播放這一小節"><I.headphones /></button>
        <button className="tbtn" disabled={loading || trackId == null} title="重新載入" aria-label="重新載入"
          onClick={() => trackId != null && load(trackId, bar)}><I.refresh /></button>
      </div>

      {err && <div className="msg bad" role="alert"><I.error />{err}</div>}
      {loading && !data && <div className="ne-empty">讀取中…</div>}

      {data && (
        <>
          <svg ref={svg} className="ne-tab" viewBox={`0 0 ${VW} ${H}`} role="group" aria-label={`第 ${bar + 1} 小節 TAB`}
            onPointerDown={(e) => { if (nStrings && len) { setCursor(pointAt(e.clientX, e.clientY)); setSel(null); } }}
            onDoubleClick={(e) => { if (!(e.target as Element).closest('[data-note]') && nStrings && len) add(pointAt(e.clientX, e.clientY)); }}>
            <title>點音符編輯・雙擊空白新增・0–9 格・↑↓ 換弦・←→ 選音・Shift+方向 移動・Del 刪除</title>
            {beatLines.map((t, i) => (
              <g key={`b${t}`}>
                <line className="ne-beat" x1={X(t)} x2={X(t)} y1={TOP - 6} y2={H - 10} />
                <text className="ne-beatnum" x={X(t) + 3} y={11}>{i + 1}</text>
              </g>
            ))}
            {subLines.map((t) => <line key={`s${t}`} className="ne-sub" x1={X(t)} x2={X(t)} y1={TOP - 4} y2={H - 12} />)}
            <line className="ne-beat" x1={X(len)} x2={X(len)} y1={TOP - 6} y2={H - 10} />
            {tuning.map((p, i) => (
              <g key={`t${i}`}>
                <line className="ne-string" x1={GUT} x2={X(len)} y1={Y(i + 1)} y2={Y(i + 1)} />
                <text className="ne-tuning" x={GUT - 7} y={Y(i + 1) + 4}>{pc(p)}</text>
              </g>
            ))}
            {items.map((it) => it.v.string >= 1 && it.v.string <= nStrings && (
              <rect key={`d${it.key}`} className={`ne-dur ${it.key === sel ? 'sel' : ''}`} x={X(it.v.pos)} y={Y(it.v.string) - 1.5}
                width={Math.max(2, X(it.v.pos + it.v.dur) - X(it.v.pos))} height={3} />
            ))}
            {cursor && <rect className="ne-cursor" x={X(cursor.pos) + 0.5} y={Y(cursor.string) - 7} width={13} height={14} rx={2} />}
            {items.map((it) => {
              const s = Math.max(1, Math.min(nStrings, it.v.string)), label = it.v.string < 1 ? '?' : Number.isFinite(it.v.fret) ? String(it.v.fret) : '·';
              const w = label.length > 1 ? 18 : 13;
              const state = [it.key === sel ? 'sel' : '', !it.orig ? 'added' : !sameDraft(it.orig, it.v) ? 'changed' : '',
                problems.has(it.key) ? 'bad' : ''].join(' ');
              return (
                <g key={it.key} data-note className={`ne-note ${state}`}
                  onPointerDown={(e) => { e.stopPropagation(); setSel(it.key); setCursor(null); }}>
                  <rect x={X(it.v.pos) + 0.5} y={Y(s) - 7} width={w} height={14} rx={2} />
                  <text x={X(it.v.pos) + 0.5 + w / 2} y={Y(s) + 4}>{label}</text>
                </g>
              );
            })}
          </svg>

          {current ? (
            <div className="ne-fields">
              <div className="ne-row">
                <span className="ne-k">音高</span>
                <span className="ne-v mono">{curPitch != null ? noteName(curPitch) : '–'}</span>
                {origPitch != null && curPitch !== origPitch && <span className="ne-was">原為 {noteName(origPitch)}</span>}
                <span className="spacer" />
                <button className="tbtn" title="刪除音符（Delete）" aria-label="刪除音符" onClick={() => remove(current)}><I.trash /></button>
              </div>
              <div className="ne-row">
                <span className="ne-k">弦</span>
                <div className="seg" role="group" aria-label="弦（保持音高）">
                  {tuning.map((p, i) => {
                    const s = i + 1, f = fretOn(current, s);
                    return (
                      <button key={s} className={current.v.string === s ? 'on' : ''} aria-pressed={current.v.string === s}
                        disabled={f < 0 || f > 36} onClick={() => moveString(current, s)}
                        title={f >= 0 && f <= 36 ? `第 ${s} 弦（${pc(p)}）第 ${f} 格` : `第 ${s} 弦彈不到這個音`}>
                        <b>{s}</b><small>{pc(p)}</small>
                      </button>
                    );
                  })}
                </div>
              </div>
              <div className="ne-row">
                <span className="ne-k">格</span>
                <div className="stepper">
                  <button aria-label="降半音" disabled={current.v.fret <= 0} onClick={() => update(current, { fret: current.v.fret - 1 })}><I.minus /></button>
                  <input type="number" aria-label="格" min={0} max={36} step={1}
                    aria-invalid={problems.get(current.key) === '格數需為 0–36 的整數'}
                    value={Number.isFinite(current.v.fret) ? current.v.fret : ''}
                    onChange={(e) => update(current, { fret: e.target.value === '' ? NaN : Math.floor(Number(e.target.value)) })} />
                  <button aria-label="升半音" disabled={current.v.fret >= 36} onClick={() => update(current, { fret: current.v.fret + 1 })}><I.plus /></button>
                </div>
              </div>
              <div className="ne-row">
                <span className="ne-k">位置</span>
                <div className="stepper wide">
                  <button aria-label="往前" title="往前一格（Shift+←）" disabled={current.v.pos <= 0} onClick={() => nudge(current, -1)}><I.left /></button>
                  <span className="mono">{posLabel(current.v.pos)}</span>
                  <button aria-label="往後" title="往後一格（Shift+→）" disabled={current.v.pos + current.v.dur + grid > len} onClick={() => nudge(current, 1)}><I.right /></button>
                </div>
                <select aria-label="格線" value={grid} onChange={(e) => setGrid(Number(e.target.value))}>
                  {GRIDS.map((g) => <option key={g.ticks} value={g.ticks}>{g.label}</option>)}
                </select>
              </div>
              <div className="ne-row">
                <span className="ne-k">時值</span>
                <div className="seg glyphs" role="group" aria-label="時值">
                  {DUR_BASES.map((b) => (
                    <button key={b.ticks} title={b.label} aria-label={b.label} aria-pressed={durParts?.base === b.ticks}
                      className={durParts?.base === b.ticks ? 'on' : ''} onClick={() => setDur(b.ticks, durParts?.mod ?? 'plain')}>
                      <NoteGlyph value={b.glyph} />
                    </button>
                  ))}
                </div>
                <div className="seg glyphs" role="group" aria-label="附點與三連音">
                  <button title="附點" aria-label="附點" aria-pressed={durParts?.mod === 'dot'} className={durParts?.mod === 'dot' ? 'on' : ''}
                    disabled={!durParts} onClick={() => durParts && setDur(durParts.base, durParts.mod === 'dot' ? 'plain' : 'dot')}><DotGlyph /></button>
                  <button title="三連音" aria-label="三連音" aria-pressed={durParts?.mod === 'trip'} className={durParts?.mod === 'trip' ? 'on' : ''}
                    disabled={!durParts} onClick={() => durParts && setDur(durParts.base, durParts.mod === 'trip' ? 'plain' : 'trip')}><TripletGlyph /></button>
                </div>
                {!durParts && <span className="ne-was">{ratio(current.v.dur, WHOLE)}</span>}
              </div>
            </div>
          ) : cursor ? (
            <div className="ne-fields idle">
              <div className="ne-row">
                <span className="ne-k">游標</span>
                <span className="mono">第 {cursor.string} 弦・{posLabel(cursor.pos)}</span>
                <span className="spacer" />
                <button className="btn" onClick={() => add(cursor)}><I.plus />新增音符</button>
              </div>
            </div>
          ) : null}

          <div className="ne-foot">
            <span className={`ne-status ${firstProblem ? 'bad' : changes ? 'dirty' : ''}`} role="status">
              {saving ? '儲存中…' : firstProblem ?? (changes ? `${changes} 處未儲存` : data.revision ? `已儲存・第 ${data.revision} 版` : '未修改')}
            </span>
            <button className="btn" disabled={!changes || saving} onClick={revertBar}><I.undo />還原</button>
            <button className="btn primary" disabled={!changes || saving || !!firstProblem || loading} onClick={save}><I.save />儲存小節</button>
          </div>
        </>
      )}

      {others.length > 0 && (
        <div className="ne-others">
          <span>其他未儲存</span>
          {others.map((k) => {
            const p = parseKey(k);
            return (
              <button key={k} className="btn link" onClick={() => { setTrackId(p.track); setBar(p.bar); }}>
                {guitars.length > 1 ? `${trackName(p.track)} ` : ''}{p.bar + 1}
              </button>
            );
          })}
        </div>
      )}

      <div className="ne-export">
        <button className="btn" disabled={exporting || !data?.revision} onClick={doExport}
          title={data?.revision ? '把已儲存的修改寫成新的 .gp' : '先儲存至少一個小節'}>
          <I.corrected />{exporting ? '匯出中…' : '匯出修正版 .gp'}
        </button>
        {exportHref && exported && (
          <a className="btn" href={exportHref} download={exported.split('/').pop()}><I.download />下載</a>
        )}
        {dirtyKeys.length > 0 && <span className="ne-was">只含已儲存的小節</span>}
        {exportError && <div className="msg bad" role="alert"><I.error />{exportError}</div>}
      </div>
    </div>
  );
}

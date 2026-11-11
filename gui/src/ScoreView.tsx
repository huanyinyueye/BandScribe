import { useEffect, useRef, useState } from 'react';
import * as alphaTab from '@coderline/alphatab';
import { api, Project } from './api';
import { familyLabel, fmtTime } from './App';

const LOW_COLOR = '#d97706';

// Colour the notes of every beat that carries the low-confidence mark ("?").
function colorLowConfidence(score: alphaTab.model.Score): number {
  let count = 0;
  const m = alphaTab.model as any;
  if (!m.NoteStyle || !m.NoteSubElement || !m.Color) return 0;
  const color = m.Color.fromJson(LOW_COLOR);
  for (const track of score.tracks) for (const staff of track.staves) for (const bar of staff.bars)
    for (const voice of bar.voices) for (const beat of voice.beats) {
      if (beat.text !== '?') continue;
      for (const note of beat.notes) {
        const style = new m.NoteStyle();
        for (const key of ['StandardNotationNoteHead', 'GuitarTabFretNumber', 'StandardNotationAccidentals']) {
          if (m.NoteSubElement[key] !== undefined) style.colors.set(m.NoteSubElement[key], color);
        }
        (note as any).style = style;
        count += 1;
      }
    }
  return count;
}

export function ScorePage({ project }: { project: Project }) {
  const host = useRef<HTMLDivElement>(null);
  const wrap = useRef<HTMLDivElement>(null);
  const apiRef = useRef<alphaTab.AlphaTabApi | null>(null);
  const audio = useRef<HTMLAudioElement>(null);
  const [tracks, setTracks] = useState<{ index: number; name: string }[]>([]);
  const [selected, setSelected] = useState<number[] | null>(null);  // null = full score
  const [ready, setReady] = useState(false);
  const [playerReady, setPlayerReady] = useState(false);
  const [synthPlaying, setSynthPlaying] = useState(false);
  const [lowCount, setLowCount] = useState(0);
  const [clicked, setClicked] = useState<number | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const gpFile = project.exports.find((f) => f.endsWith('.gp'));
  const bars = project.score?.bars ?? [];
  const ext = project.meta.filename.slice(project.meta.filename.lastIndexOf('.')).toLowerCase();

  useEffect(() => {
    if (!host.current || !gpFile) return;
    setReady(false);
    setLoadError(null);
    const at = new alphaTab.AlphaTabApi(host.current, {
      core: { fontDirectory: '/font/', tracks: 'all' },
      player: { enablePlayer: true, soundFont: '/soundfont/sonivox.sf2', scrollElement: wrap.current ?? undefined,
        enableCursor: true, enableUserInteraction: true },
      display: { layoutMode: alphaTab.LayoutMode.Page, scale: 0.9 },
    } as any);
    apiRef.current = at;
    at.scoreLoaded.on((score) => {
      setLowCount(colorLowConfidence(score));
      setTracks(score.tracks.map((t, i) => ({ index: i, name: t.name })));
    });
    at.renderFinished.on(() => setReady(true));
    at.playerReady.on(() => setPlayerReady(true));
    at.playerStateChanged.on((e: any) => setSynthPlaying(e.state === 1));
    at.error.on((e: any) => setLoadError(String(e?.message ?? e)));
    // Clicking a beat moves the original recording to the same musical position (A/B compare).
    at.beatMouseDown.on((beat) => {
      const bar = bars[beat.voice.bar.index];
      if (!bar) return;
      const master = beat.voice.bar.masterBar;
      const quarters = (beat.absolutePlaybackStart - master.start) / 960;
      const t = bar.start_time + quarters * (60 / bar.bpm);
      setClicked(t);
      if (audio.current) audio.current.currentTime = Math.max(0, t);
    });
    fetch(api.fileUrl(project.meta.id, `run/exports/${encodeURIComponent(gpFile)}`))
      .then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(new Error(`讀取 .gp 失敗（${r.status}）`))))
      .then((buf) => at.load(new Uint8Array(buf)))
      .catch((e) => setLoadError(String(e.message ?? e)));
    return () => { at.destroy(); apiRef.current = null; };
  }, [project.meta.id, gpFile, project.manifest?.stages?.export]);  // eslint-disable-line react-hooks/exhaustive-deps

  const show = (sel: number[] | null) => {
    setSelected(sel);
    const at = apiRef.current;
    if (!at?.score) return;
    setReady(false);
    at.renderTracks(sel === null ? at.score.tracks : sel.map((i) => at.score!.tracks[i]));
  };
  const playSynth = () => { audio.current?.pause(); apiRef.current?.playPause(); };
  const playOriginal = () => {
    apiRef.current?.pause();
    const el = audio.current;
    if (!el) return;
    if (el.paused) void el.play(); else el.pause();
  };

  if (!gpFile) return (<><h2>樂譜檢視</h2><p className="muted">這次執行沒有產生 .gp 檔。</p></>);
  const summary = project.score;
  return (
    <>
      <h2>樂譜檢視</h2>
      <div className="panel">
        <div className="row">
          <button className={`btn ${selected === null ? 'on' : ''}`} onClick={() => show(null)}>總譜</button>
          {tracks.map((t) => (
            <button key={t.index} className={`btn ${selected?.length === 1 && selected[0] === t.index ? 'on' : ''}`}
              onClick={() => show([t.index])}>{t.name}</button>
          ))}
        </div>
        <div className="row" style={{ marginTop: 10 }}>
          <button className="btn primary" disabled={!playerReady} onClick={playSynth}>
            {synthPlaying ? '暫停樂譜播放' : playerReady ? '播放樂譜（合成音）' : '載入音色中…'}
          </button>
          <button className="btn" onClick={playOriginal}>播放／暫停原音</button>
          <button className="btn" onClick={() => { apiRef.current?.stop(); }}>停止</button>
          {clicked != null && <span className="muted">已將原音定位到 {fmtTime(clicked)}</span>}
        </div>
        <audio ref={audio} controls style={{ width: '100%', marginTop: 8 }}
          src={api.fileUrl(project.meta.id, `input${ext}`)} />
        <div className="small muted" style={{ marginTop: 8 }}>
          <span className="lowkey">橘色音符與「?」</span> 表示低信心（共 {lowCount} 個音符頭）。點擊任一拍可把原音跳到相同位置做 A/B 比對。
        </div>
      </div>
      {loadError && <div className="error">{loadError}</div>}
      {!ready && !loadError && <p className="muted">樂譜繪製中…</p>}
      <div className="scorewrap" ref={wrap}><div ref={host} /></div>
      {summary && (
        <div className="panel" style={{ marginTop: 16 }}>
          <table>
            <thead><tr><th>軌道</th><th>音符數</th><th>低信心</th><th>調弦</th></tr></thead>
            <tbody>
              {summary.tracks.map((t) => (
                <tr key={t.id}><td>{t.name}<span className="muted small">（{familyLabel(t.family)}）</span></td><td>{t.notes}</td>
                  <td className={t.low_conf ? 'lowkey' : ''}>{t.low_conf}（{t.notes ? Math.round((t.low_conf / t.notes) * 100) : 0}%）</td>
                  <td>{String(t.info?.tuning_name ?? '—')}
                    {summary.meta?.tab?.[t.family]?.auto ? <span className="muted small">（自動建議）</span> : null}</td></tr>
              ))}
            </tbody>
          </table>
          <p className="small muted">
            調性（由音符推估）：{summary.key?.name ?? '—'}・小節數：{summary.bars.length}
          </p>
        </div>
      )}
    </>
  );
}

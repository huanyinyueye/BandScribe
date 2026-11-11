import { useEffect, useRef, useState } from 'react';
import { api, Config, Project } from './api';
import { familyLabel, fmtTime } from './App';

const ORDER = ['vocals', 'guitar', 'piano', 'other', 'bass', 'drums'];

// Confidence colour: low = orange, high = green; inactive windows are neutral.
function heatColor(v: number, active: boolean): string {
  if (!active) return 'var(--line)';
  const hue = 30 + 110 * Math.max(0, Math.min(1, v));
  return `hsl(${hue} 70% 45%)`;
}

export function StemsPage({ project, config, running, onOverride, onRerun }: {
  project: Project; config: Config; running: boolean;
  onOverride: (stem: string, state: string | null) => Promise<void>; onRerun: () => void;
}) {
  const stems = project.stems?.stems ?? {};
  const names = ORDER.filter((n) => n in stems);
  const audios = useRef<Record<string, HTMLAudioElement | null>>({});
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [muted, setMuted] = useState<Record<string, boolean>>({});
  const [solo, setSolo] = useState<string | null>(null);
  const duration = project.waveform.duration;
  const dirty = JSON.stringify(config.stem_overrides ?? {}) !== JSON.stringify(project.config?.stem_overrides ?? {});

  // Mute / solo.
  useEffect(() => {
    for (const n of names) {
      const el = audios.current[n];
      if (el) el.muted = solo ? solo !== n : !!muted[n];
    }
  }, [muted, solo, names.join(',')]);  // eslint-disable-line react-hooks/exhaustive-deps

  // Keep every stem on the same clock (the first stem is the reference).
  useEffect(() => {
    const id = window.setInterval(() => {
      const ref = audios.current[names[0]];
      if (!ref) return;
      setTime(ref.currentTime);
      for (const n of names.slice(1)) {
        const el = audios.current[n];
        if (el && Math.abs(el.currentTime - ref.currentTime) > 0.08) el.currentTime = ref.currentTime;
      }
      if (ref.ended) setPlaying(false);
    }, 200);
    return () => window.clearInterval(id);
  }, [names.join(',')]);  // eslint-disable-line react-hooks/exhaustive-deps

  const each = (fn: (el: HTMLAudioElement) => void) => names.forEach((n) => { const el = audios.current[n]; if (el) fn(el); });
  const toggle = () => {
    if (playing) each((el) => el.pause());
    else each((el) => { void el.play(); });
    setPlaying(!playing);
  };
  const seek = (t: number) => { each((el) => { el.currentTime = t; }); setTime(t); };

  return (
    <>
      <h2>分軌檢視</h2>
      <div className="panel">
        <div className="row">
          <button className="btn primary" onClick={toggle}>{playing ? '暫停' : '播放'}</button>
          <span>{fmtTime(time)} / {fmtTime(duration)}</span>
          <input type="range" min={0} max={duration} step={0.1} value={time} style={{ flex: 1 }}
            aria-label="播放位置" onChange={(e) => seek(Number(e.target.value))} />
        </div>
        <div className="small muted" style={{ marginTop: 8 }}>
          <span className="legend"><i style={{ background: heatColor(1, true) }} />兩個分離模型一致</span>
          <span className="legend"><i style={{ background: heatColor(0, true) }} />模型不一致（該段音符會降低信心）</span>
          <span className="legend"><i style={{ background: 'var(--line)' }} />無聲</span>
          點擊色帶可跳到該位置。
        </div>
      </div>
      <div className="panel">
        {names.map((n) => {
          const s = stems[n];
          const absent = config.stem_overrides?.[n] === 'absent';
          const conf = s.confidence;
          return (
            <div className="stem" key={n}>
              <div>
                <div className="name">{familyLabel(n)}</div>
                <div className="small muted">
                  {s.activity ? (s.activity.present ? `有訊號・${s.activity.relative_db} dB` : '幾乎無聲，未轉譜') : ''}
                </div>
                <div className="small muted" title={s.model_id}>{s.model_id.startsWith('ensemble:') ? '雙模型合成' : s.model_id}</div>
              </div>
              <div>
                <div className="row" style={{ marginBottom: 6 }}>
                  <button className={`btn ${solo === n ? 'on' : ''}`} aria-pressed={solo === n}
                    onClick={() => setSolo(solo === n ? null : n)}>Solo</button>
                  <button className={`btn ${muted[n] ? 'on' : ''}`} aria-pressed={!!muted[n]}
                    onClick={() => setMuted({ ...muted, [n]: !muted[n] })}>Mute</button>
                  <label className="check">
                    <input type="checkbox" checked={absent}
                      onChange={(e) => void onOverride(n, e.target.checked ? 'absent' : null)} />
                    這軌其實不是{familyLabel(n)}（不要轉譜）
                  </label>
                </div>
                {conf ? (
                  <div className="heat" role="img" aria-label={`${familyLabel(n)}信心熱圖`}
                    onClick={(e) => {
                      const r = e.currentTarget.getBoundingClientRect();
                      seek(((e.clientX - r.left) / r.width) * duration);
                    }}>
                    {conf.values.map((v, i) => (
                      <i key={i} style={{ background: heatColor(v, conf.active[i]) }}
                        title={`${fmtTime(i * conf.window)}：${conf.active[i] ? `信心 ${v.toFixed(2)}` : '無聲'}`} />
                    ))}
                  </div>
                ) : <div className="small muted">沒有第二個分離模型可比對，無信心資料。</div>}
                <audio ref={(el) => { audios.current[n] = el; }} preload="auto"
                  src={api.fileUrl(project.meta.id, `run/stems/${n}.wav`)} />
              </div>
            </div>
          );
        })}
      </div>
      <div className="row">
        <button className="btn primary" disabled={!dirty || running} onClick={onRerun}>
          套用修正並重跑後段流程
        </button>
        <span className="small muted">{dirty ? '分離結果會沿用快取，只重跑轉譜之後的步驟。' : '勾選上方的修正後可重跑。'}</span>
      </div>
    </>
  );
}

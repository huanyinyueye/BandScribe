import { ReactNode, useCallback, useEffect, useRef, useState } from 'react';
import { api, Config, Info, Job, Project, ProjectMeta, QuickAnalysis, Waveform } from './api';
import { StemsPage } from './Stems';
import { ScorePage } from './ScoreView';
import { familyIcon, I } from './icons';
import { drawWave } from './wave';

type Page = 'import' | 'settings' | 'progress' | 'stems' | 'score' | 'export' | 'about';
const PAGES: { id: Page; label: string; needsProject?: boolean; needsResult?: boolean }[] = [
  { id: 'import', label: '匯入' },
  { id: 'settings', label: '設定', needsProject: true },
  { id: 'progress', label: '處理', needsProject: true },
  { id: 'score', label: '審譜', needsResult: true },
  { id: 'stems', label: '分軌', needsResult: true },
  { id: 'export', label: '匯出', needsResult: true },
];
const STAGE_LABEL: Record<string, string> = {
  load: '載入音訊', beats: '節拍分析', separate: '音源分離', attribute: '樂器歸屬',
  transcribe: '轉譜', tab: 'TAB 指派', notation: '量化與樂譜', qa: '品質檢查', export: '匯出',
};
const FAMILY_LABEL: Record<string, string> = {
  vocals: '人聲', drums: '鼓', bass: '貝斯', guitar: '吉他', piano: '鋼琴', other: '其他',
};
export const familyLabel = (f: string) => FAMILY_LABEL[f] ?? f;

export function fmtTime(s: number): string {
  const m = Math.floor(s / 60);
  return `${m}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
}

const REF_ACCEPT = '.gp,.gp3,.gp4,.gp5,.gpx';
const errText = (e: any) => String(e?.message ?? e);

export function App() {
  const [page, setPage] = useState<Page>('import');
  const [info, setInfo] = useState<Info | null>(null);
  const [projectId, setProjectId] = useState<string | null>(null);
  const [project, setProject] = useState<Project | null>(null);
  const [config, setConfig] = useState<Config | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => { api.info().then(setInfo).catch((e) => setError(errText(e))); }, []);

  const refresh = useCallback(async (id: string) => {
    const p = await api.project(id);
    setProject(p);
    return p;
  }, []);

  const openProject = useCallback(async (id: string, review = false) => {
    setError(null);
    setProjectId(id);
    const p = await refresh(id);
    setJob(p.job ?? null);
    if (review && p.manifest?.status === 'done') setPage('score');
  }, [refresh]);

  // Config: the profile defaults, overlaid with what this project last used.
  useEffect(() => {
    if (!info || !project) return;
    const profile = project.config?.profile ?? 'quality';
    setConfig({ ...info.defaults[profile], ...(project.config ?? {}) } as Config);
  }, [info, project?.meta.id]);  // eslint-disable-line react-hooks/exhaustive-deps

  // Follow the active job.
  useEffect(() => {
    if (!job || ['done', 'failed', 'cancelled'].includes(job.status)) return;
    return api.watchJob(job.id, (j) => {
      setJob(j);
      if (['done', 'failed', 'cancelled'].includes(j.status) && projectId) void refresh(projectId);
    });
  }, [job?.id]);  // eslint-disable-line react-hooks/exhaustive-deps

  const start = async (cfg: Config) => {
    if (!projectId) return;
    setError(null);
    try {
      const { profile, separator_primary, separator_secondary, pitched_transcribers, drum_transcribers,
        mix_vote, low_confidence, strict, guitar_tuning, bass_tuning, instruments, exports, title,
        max_span, stem_overrides } = cfg;
      const j = await api.run(projectId, { profile, separator_primary, separator_secondary,
        pitched_transcribers, drum_transcribers, mix_vote, low_confidence, strict, guitar_tuning,
        bass_tuning, instruments, exports, title, max_span, stem_overrides });
      setJob(j);
      setPage('progress');
    } catch (e) {
      setError(errText(e));
    }
  };

  const attachReference = async (file: File) => {
    if (!projectId) return;
    await api.uploadReference(projectId, file);
    await refresh(projectId);
  };
  const clearReference = async () => {
    if (!projectId) return;
    await api.clearReference(projectId);
    await refresh(projectId);
  };

  const hasResult = project?.manifest?.status === 'done';
  const running = !!job && ['queued', 'running'].includes(job.status);

  const tab = (id: Page, label: ReactNode, disabled = false) => (
    <button key={id} className={`tab ${page === id ? 'on' : ''}`} aria-current={page === id ? 'page' : undefined}
      disabled={disabled} onClick={() => setPage(id)}>
      {label}
      {id === 'progress' && running && <span className="tab-pct">{Math.round((job?.overall ?? 0) * 100)}%</span>}
    </button>
  );

  const banner = error && (
    <div className="app-err msg bad" role="alert">
      <I.error /><span>{error}</span>
      <button className="tbtn sm" onClick={() => setError(null)} aria-label="關閉" title="關閉"><I.close size={14} /></button>
    </div>
  );

  return (
    <div className="app">
      <header className="topbar">
        <span className="brand"><I.logo size={18} />BandScribe</span>
        <span className="song" title={project?.meta.filename}>
          {project ? <><I.song size={14} />{project.meta.name}</> : <span className="dim">尚未匯入</span>}
        </span>
        <nav className="tabs" aria-label="頁面">
          {PAGES.map((p) => tab(p.id, p.label, (p.needsProject && !project) || (p.needsResult && !hasResult)))}
        </nav>
        <span className="spacer" />
        <nav className="tabs aux" aria-label="其他">
          {tab('about', <><I.info />關於</>)}
        </nav>
        <span className="ver" title="伺服器僅綁定 127.0.0.1">v{info?.version ?? '…'}</span>
      </header>
      {page === 'score' && project ? (
        <main className="workspace">
          {banner}
          <ScorePage project={project} onAttachReference={attachReference} />
        </main>
      ) : (
        <main className="doc">
          {banner}
          <div className="view">
            {page === 'import' && (
              <ImportPage project={project} onOpen={openProject} onError={setError} busy={running}
                onAttachReference={attachReference} onClearReference={clearReference}
                onNext={() => setPage('settings')} onReview={() => setPage('score')} />
            )}
            {page === 'settings' && info && config && (
              <SettingsPage info={info} config={config} setConfig={setConfig} running={running}
                onStart={() => start(config)} />
            )}
            {page === 'progress' && (
              <ProgressPage job={job} project={project} onCancel={async () => job && setJob(await api.cancel(job.id))}
                onView={() => setPage('score')} />
            )}
            {page === 'stems' && project && config && (
              <StemsPage project={project} config={config} running={running}
                onOverride={async (stem, state) => {
                  const r = await api.override(project.meta.id, stem, state);
                  setConfig({ ...config, stem_overrides: r.stem_overrides });
                }}
                onRerun={() => start(config)} />
            )}
            {page === 'export' && project && <ExportPage project={project} onError={setError} />}
            {page === 'about' && info && <AboutPage info={info} />}
          </div>
        </main>
      )}
    </div>
  );
}

// Compact player: one play button and a waveform you can click or drag to seek.
function Player({ src, wave }: { src: string; wave: Waveform }) {
  const audio = useRef<HTMLAudioElement>(null);
  const box = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [width, setWidth] = useState(0);
  const dragging = useRef(false);
  const duration = wave.duration;

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    const step = () => { if (audio.current) setTime(audio.current.currentTime); raf = requestAnimationFrame(step); };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [playing]);

  useEffect(() => {
    const c = canvas.current;
    if (!c || !width) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = width * dpr;
    c.height = c.clientHeight * dpr;
    const g = c.getContext('2d')!;
    const css = getComputedStyle(c);
    const W = c.width, H = c.height, split = (time / Math.max(duration, 1e-6)) * W;
    g.clearRect(0, 0, W, H);
    drawWave(g, wave, 0, 2 * dpr, W, H - 4 * dpr, css.getPropertyValue('--wave').trim() || '#8a95a8');
    g.save();
    g.beginPath(); g.rect(0, 0, split, H); g.clip();
    drawWave(g, wave, 0, 2 * dpr, W, H - 4 * dpr, css.getPropertyValue('--accent').trim() || '#1f5fd1');
    g.restore();
  }, [wave, width, time, duration]);

  const seek = (t: number) => {
    const v = Math.max(0, Math.min(duration, t));
    if (audio.current) audio.current.currentTime = v;
    setTime(v);
  };
  const timeAt = (x: number) => {
    const r = box.current!.getBoundingClientRect();
    return (Math.max(0, Math.min(1, (x - r.left) / r.width))) * duration;
  };
  const toggle = () => { const el = audio.current; if (!el) return; if (el.paused) void el.play(); else el.pause(); };

  return (
    <div className="player">
      <button className="tbtn play" onClick={toggle} aria-label={playing ? '暫停' : '播放'} title={playing ? '暫停' : '播放'}>
        {playing ? <I.pause /> : <I.play />}
      </button>
      <span className="pl-time mono">{fmtTime(time)}</span>
      <div className="pl-wave" ref={box} role="slider" tabIndex={0} aria-label="播放位置"
        aria-valuemin={0} aria-valuemax={Math.round(duration)} aria-valuenow={Math.round(time)} aria-valuetext={fmtTime(time)}
        onKeyDown={(e) => {
          if (e.key === 'ArrowRight') { e.preventDefault(); seek(time + 5); }
          else if (e.key === 'ArrowLeft') { e.preventDefault(); seek(time - 5); }
          else if (e.key === ' ') { e.preventDefault(); toggle(); }
        }}
        onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); dragging.current = true; seek(timeAt(e.clientX)); }}
        onPointerMove={(e) => { if (dragging.current) seek(timeAt(e.clientX)); }}
        onPointerUp={() => { dragging.current = false; }}
        onPointerCancel={() => { dragging.current = false; }}>
        <canvas ref={canvas} />
        <div className="pl-head" style={{ left: `${(time / Math.max(duration, 1e-6)) * 100}%` }} />
      </div>
      <span className="pl-time mono dim">{fmtTime(duration)}</span>
      <audio ref={audio} src={src} preload="metadata"
        onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)} onTimeUpdate={(e) => !playing && setTime(e.currentTarget.currentTime)} />
    </div>
  );
}

function Fact({ k, v, sub }: { k: string; v: ReactNode; sub?: string }) {
  return (
    <div className="fact">
      <span className="fact-k">{k}</span>
      <span className="fact-v">{v}</span>
      {sub && <span className="fact-s">{sub}</span>}
    </div>
  );
}

function ImportPage({ project, onOpen, onError, onNext, onReview, busy, onAttachReference, onClearReference }: {
  project: Project | null; onOpen: (id: string, review?: boolean) => Promise<void>; onError: (e: string | null) => void;
  onNext: () => void; onReview: () => void; busy: boolean;
  onAttachReference: (f: File) => Promise<void>; onClearReference: () => Promise<void>;
}) {
  const [over, setOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [quick, setQuick] = useState<QuickAnalysis | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [recent, setRecent] = useState<ProjectMeta[] | null>(null);
  const [refBusy, setRefBusy] = useState<'up' | 'rm' | null>(null);
  const [refError, setRefError] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const refInput = useRef<HTMLInputElement>(null);

  useEffect(() => { api.projects().then(setRecent).catch(() => setRecent([])); }, [project?.meta.id]);
  useEffect(() => { setQuick(null); setRefError(null); }, [project?.meta.id]);

  const send = async (file: File | undefined) => {
    if (!file) return;
    onError(null);
    setUploading(true);
    try {
      const meta = await api.upload(file);
      await onOpen(meta.id);
    } catch (e) {
      onError(`匯入失敗：${errText(e)}`);
    } finally {
      setUploading(false);
      if (input.current) input.current.value = '';
    }
  };
  const analyze = async () => {
    if (!project) return;
    setAnalyzing(true);
    try { setQuick(await api.analyze(project.meta.id)); } catch (e) { onError(errText(e)); }
    setAnalyzing(false);
  };
  const attach = async (file: File | undefined) => {
    if (!file) return;
    setRefBusy('up');
    setRefError(null);
    try { await onAttachReference(file); } catch (e) { setRefError(`上傳失敗：${errText(e)}`); }
    finally { setRefBusy(null); if (refInput.current) refInput.current.value = ''; }
  };
  const detach = async () => {
    setRefBusy('rm');
    setRefError(null);
    try { await onClearReference(); } catch (e) { setRefError(`移除失敗：${errText(e)}`); }
    finally { setRefBusy(null); }
  };

  const a = project?.analysis;
  const done = project?.manifest?.status === 'done';
  const ref = project?.reference ?? null;
  const ext = project ? project.meta.filename.slice(project.meta.filename.lastIndexOf('.')).toLowerCase() : '';

  const drop = (
    <div className={`drop ${over ? 'over' : ''} ${project ? 'compact' : ''}`} role="button" tabIndex={0}
      aria-busy={uploading}
      onClick={() => !uploading && input.current?.click()}
      onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && !uploading && input.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); void send(e.dataTransfer.files[0]); }}>
      <I.upload size={project ? 16 : 28} />
      <span className="drop-main">{uploading ? '匯入中…' : project ? '匯入另一首' : '拖入錄音，或點擊選擇檔案'}</span>
      <span className="drop-sub">MP3 · WAV · FLAC</span>
      <input ref={input} type="file" accept=".mp3,.wav,.flac" hidden onChange={(e) => void send(e.target.files?.[0])} />
    </div>
  );

  return (
    <>
      <div className="view-h">
        <h1>匯入</h1>
        {project && drop}
      </div>
      {!project && drop}

      {project && (
        <>
          <section className="sec">
            <div className="sec-h">
              <I.song /><span className="sec-t" title={project.meta.filename}>{project.meta.filename}</span>
              <span className="spacer" />
              <span className="dim mono">{project.waveform.orig_sr} Hz · {project.waveform.orig_channels === 1 ? '單聲道' : `${project.waveform.orig_channels} 聲道`}</span>
            </div>
            <Player src={api.fileUrl(project.meta.id, `input${ext}`)} wave={project.waveform} />
            <div className="facts">
              <Fact k="長度" v={fmtTime(project.waveform.duration)} />
              {a ? (
                <>
                  <Fact k="速度" v={a.global_bpm} sub="BPM" />
                  <Fact k="拍號" v={`${a.meter.dominant[0]}/${a.meter.dominant[1]}`}
                    sub={a.meter.changes.length > 1 ? `${a.meter.changes.length - 1} 次變拍` : '無變拍'} />
                  <Fact k="調性" v={a.key?.name ?? '–'} sub="由音符推估" />
                  <Fact k="小節" v={a.bars.length} />
                </>
              ) : quick ? (
                <>
                  <Fact k="速度" v={quick.bpm} sub={quick.tempo_changes ? `${quick.tempo_changes} 次變速` : '速度穩定'} />
                  <Fact k="拍號" v={`${quick.meter[0]}/${quick.meter[1]}`} sub={quick.meter_changes ? `${quick.meter_changes} 次變拍` : '無變拍'} />
                  <Fact k="調性" v={quick.key_guess} sub="粗估，轉譜後依音符重估" />
                </>
              ) : (
                <div className="fact act">
                  <button className="btn" disabled={analyzing || busy} onClick={analyze}>
                    <I.target />{analyzing ? '分析中…' : '分析速度、拍號、調性'}
                  </button>
                </div>
              )}
            </div>
          </section>

          <section className="sec">
            <div className="sec-h">
              <I.book /><span className="sec-t">參考譜</span><span className="dim">選用</span>
            </div>
            {ref ? (
              <div className="file-row">
                <I.file />
                <a className="file-name" href={api.fileUrl(project.meta.id, ref.path.split('/').map(encodeURIComponent).join('/'))}
                  download={ref.filename} title="下載原檔">{ref.filename}</a>
                <span className="spacer" />
                <button className="btn" disabled={!!refBusy} onClick={() => refInput.current?.click()}>
                  <I.refresh />{refBusy === 'up' ? '上傳中…' : '更換'}
                </button>
                <button className="btn" disabled={!!refBusy} onClick={detach}>
                  <I.trash />{refBusy === 'rm' ? '移除中…' : '移除'}
                </button>
              </div>
            ) : (
              <div className="file-row">
                <button className="btn" disabled={!!refBusy} onClick={() => refInput.current?.click()}>
                  <I.plus />{refBusy === 'up' ? '上傳中…' : '附加 Guitar Pro 檔'}
                </button>
                <span className="dim mono">.gp .gp3 .gp4 .gp5 .gpx</span>
              </div>
            )}
            <input ref={refInput} type="file" accept={REF_ACCEPT} hidden onChange={(e) => void attach(e.target.files?.[0])} />
            {refError && <div className="msg bad" role="alert"><I.error />{refError}</div>}
            <p className="note"><I.info size={14} />只供對照：不參與轉譜、不能編輯，也不會與錄音同步；在審譜頁以合成音播放。</p>
          </section>

          <div className="actions">
            {done && <button className="btn primary" onClick={onReview}><I.score />開始審譜</button>}
            <button className={`btn ${done ? '' : 'primary'}`} onClick={onNext}>
              <I.settings />{done ? '調整設定重跑' : '下一步：設定'}
            </button>
          </div>
        </>
      )}

      <section className="sec plain">
        <div className="sec-h"><span className="sec-t">最近的專案</span></div>
        {recent === null ? <div className="dim">讀取中…</div>
          : recent.length === 0 ? <div className="dim">還沒有專案。</div> : (
            <ul className="list">
              {recent.slice(0, 10).map((r) => (
                <li key={r.id} className={r.id === project?.meta.id ? 'cur' : ''}>
                  <button className="list-row" onClick={() => void onOpen(r.id, true).catch((e) => onError(errText(e)))}>
                    <I.song />
                    <span className="list-name">{r.name}</span>
                    <span className="list-meta">
                      {r.status === 'done' ? <><I.ok size={14} className="good" />已轉譜</>
                        : r.status === 'new' ? '尚未處理' : r.status}
                    </span>
                    <span className="list-go">{r.status === 'done' ? '審譜' : '開啟'}<I.right size={14} /></span>
                  </button>
                </li>
              ))}
            </ul>
          )}
      </section>
    </>
  );
}

function SettingsPage({ info, config, setConfig, running, onStart }: {
  info: Info; config: Config; setConfig: (c: Config) => void; running: boolean; onStart: () => void;
}) {
  const set = <K extends keyof Config>(k: K, v: Config[K]) => setConfig({ ...config, [k]: v });
  const seps = info.models.filter((m) => m.task === 'separation');
  const model = (id: string) => info.models.find((m) => m.id === id);
  const label = (id: string) => {
    const m = model(id);
    return m ? `${m.name}${m.status === 'disabled' ? '（實驗性，可能極慢或失敗）' : ''}` : id;
  };
  const toggle = (list: string[], v: string) => (list.includes(v) ? list.filter((x) => x !== v) : [...list, v]);
  // Order matters: the first engine is the primary, the second the cross-check.
  const pitchedChoices: { key: string; list: string[]; text: string }[] = [
    { key: 'both', list: ['muscriptor_medium', 'basic_pitch'],
      text: `${label('muscriptor_medium')}＋${label('basic_pitch')}（交叉驗證；吉他只保留 MuScriptor 也抓到的音）` },
    { key: 'muscriptor', list: ['muscriptor_medium'], text: `只用 ${label('muscriptor_medium')}` },
    { key: 'basic_pitch', list: ['basic_pitch'], text: `只用 ${label('basic_pitch')}` },
  ];
  const pitchedKey = pitchedChoices.find((c) => c.list.join() === config.pitched_transcribers.join())?.key ?? 'custom';
  const drumChoices = ['muscriptor_medium', 'adtof'];
  const usesMus = [...config.pitched_transcribers, ...config.drum_transcribers, config.mix_vote ?? '']
    .some((t) => t.startsWith('muscriptor_'));
  return (
    <>
      <div className="view-h"><h1>設定</h1></div>
      <section className="sec">
        <div className="form">
          <label className="field"><span>品質</span>
            <select value={config.profile} onChange={(e) => setConfig({ ...config, ...info.defaults[e.target.value],
              title: config.title, stem_overrides: config.stem_overrides })}>
              <option value="quality">quality（雙分離模型交叉驗證＋混音投票，較慢）</option>
              <option value="balanced">balanced（單一分離模型，較快）</option>
            </select>
          </label>
          <label className="field"><span>標題</span>
            <input type="text" value={config.title ?? ''} placeholder="預設為檔名"
              onChange={(e) => set('title', e.target.value || null)} />
          </label>
        </div>
      </section>

      <section className="sec">
        <div className="sec-h"><I.cpu /><span className="sec-t">模型</span></div>
        <div className="form">
          <label className="field"><span>主分離模型</span>
            <select value={config.separator_primary} onChange={(e) => set('separator_primary', e.target.value)}>
              {seps.map((m) => <option key={m.id} value={m.id}>{label(m.id)}</option>)}
            </select>
            <small>吉他／鋼琴軌的唯一來源</small>
          </label>
          <label className="field"><span>交叉驗證分離模型</span>
            <select value={config.separator_secondary ?? ''} onChange={(e) => set('separator_secondary', e.target.value || null)}>
              <option value="">不使用</option>
              {seps.filter((m) => m.id !== config.separator_primary && m.id !== 'mvsep_mega53')
                .map((m) => <option key={m.id} value={m.id}>{label(m.id)}</option>)}
            </select>
            <small>鼓／貝斯／人聲</small>
          </label>
          <label className="field span2"><span>音高轉譜引擎</span>
            <select value={pitchedKey}
              onChange={(e) => set('pitched_transcribers', pitchedChoices.find((c) => c.key === e.target.value)!.list)}>
              {pitchedChoices.map((c) => <option key={c.key} value={c.key}>{c.text}</option>)}
              {pitchedKey === 'custom' && (
                <option value="custom" disabled>{config.pitched_transcribers.map(label).join(' → ')}（目前設定）</option>
              )}
            </select>
            <small>吉他／貝斯／鋼琴／人聲</small>
          </label>
          <div className="field"><span>鼓轉譜引擎</span>
            <div className="checks">
              {drumChoices.map((id) => {
                const on = config.drum_transcribers.includes(id);
                return (
                  <label key={id} className="check">
                    <input type="checkbox" checked={on} disabled={on && config.drum_transcribers.length === 1}
                      onChange={() => set('drum_transcribers', toggle(config.drum_transcribers, id))} />
                    {label(id)}
                  </label>
                );
              })}
            </div>
          </div>
          <div className="field"><span>樂器歸屬</span>
            <label className="check">
              <input type="checkbox" checked={!!config.mix_vote}
                onChange={(e) => set('mix_vote', e.target.checked ? 'muscriptor_medium' : null)} />
              以原始混音轉譜投票（{label(config.mix_vote ?? 'muscriptor_medium')}）
            </label>
          </div>
        </div>
        {usesMus && <p className="note"><I.lock size={14} />MuScriptor 權重需在 Hugging Face 接受授權；取不到時會略過並在結果中註明。</p>}
      </section>

      <section className="sec">
        <div className="sec-h"><I.guitar /><span className="sec-t">樂器與調弦</span></div>
        <div className="inst">
          {info.defaults.quality.instruments.map((f) => {
            const on = config.instruments.includes(f);
            return (
              <button key={f} className={`inst-b ${on ? 'on' : ''}`} aria-pressed={on}
                onClick={() => set('instruments', toggle(config.instruments, f))}>
                {familyIcon(f, { size: 18 })}<span>{familyLabel(f)}</span>
              </button>
            );
          })}
        </div>
        <div className="form">
          {(['guitar', 'bass'] as const).map((inst) => (
            <label key={inst} className="field"><span>{familyLabel(inst)}調弦</span>
              <select value={inst === 'guitar' ? config.guitar_tuning : config.bass_tuning}
                onChange={(e) => set(inst === 'guitar' ? 'guitar_tuning' : 'bass_tuning', e.target.value)}>
                <option value="auto">自動偵測（依最低音建議）</option>
                {info.tunings.filter((t) => t.instrument === inst).map((t) => <option key={t.key} value={t.key}>{t.name}</option>)}
              </select>
            </label>
          ))}
        </div>
      </section>

      <section className="sec">
        <div className="sec-h"><I.warn /><span className="sec-t">低信心</span></div>
        <div className="form">
          <label className="field"><span>門檻 <b className="mono">{config.low_confidence.toFixed(2)}</b></span>
            <input type="range" min={0} max={1} step={0.05} value={config.low_confidence}
              onChange={(e) => set('low_confidence', Number(e.target.value))} />
            <small>低於門檻的音以「?」與橘色標示</small>
          </label>
          <div className="field"><span>strict</span>
            <label className="check">
              <input type="checkbox" checked={config.strict} onChange={(e) => set('strict', e.target.checked)} />
              直接移除低信心音
            </label>
          </div>
        </div>
      </section>

      <div className="actions">
        <button className="btn primary" disabled={running || config.instruments.length === 0} onClick={onStart}>
          <I.play />{running ? '處理中…' : '開始轉譜'}
        </button>
        {config.instruments.length === 0 && <span className="dim">至少選一種樂器</span>}
      </div>
    </>
  );
}

function ProgressPage({ job, project, onCancel, onView }: {
  job: Job | null; project: Project | null; onCancel: () => void; onView: () => void;
}) {
  const manifest = project?.manifest;
  if (!job && !manifest) {
    return (<><div className="view-h"><h1>處理</h1></div><p className="dim">尚未開始。到「設定」按「開始轉譜」。</p></>);
  }
  const stages = job?.stages ?? Object.keys(STAGE_LABEL);
  const idx = job ? stages.indexOf(job.stage) : stages.length;
  const status = job?.status ?? manifest?.status;
  const statusText: Record<string, string> = {
    queued: '排隊中', running: '處理中', done: '完成', failed: '失敗', cancelled: '已取消',
  };
  const statusIcon = status === 'done' ? <I.ok className="good" /> : status === 'failed' ? <I.error className="bad" />
    : status === 'cancelled' ? <I.stop /> : <I.refresh />;
  return (
    <>
      <div className="view-h"><h1>處理</h1></div>
      <section className="sec">
        <div className="sec-h">
          {statusIcon}<span className="sec-t">{statusText[status] ?? status}</span>
          <span className="spacer" />
          {job && (
            <span className="dim mono">
              {fmtTime(job.elapsed)}{job.eta != null && job.status === 'running' ? ` · 剩 ${fmtTime(job.eta)}` : ''} · VRAM {job.vram_gb.toFixed(2)} GB
            </span>
          )}
        </div>
        {job && (
          <>
            <div className="meter" role="progressbar" aria-valuemin={0} aria-valuemax={100}
              aria-valuenow={Math.round(job.overall * 100)}><i style={{ width: `${job.overall * 100}%` }} /></div>
            <div className="dim job-msg" aria-live="polite">{job.message}</div>
            <ol className="stages">
              {stages.map((s, i) => {
                const done = job.status === 'done' || i < idx;
                const cur = i === idx && job.status === 'running';
                const frac = done ? 1 : i === idx ? job.stage_fraction : 0;
                return (
                  <li key={s} className={done ? 'done' : cur ? 'cur' : ''}>
                    <span className="st-i">{done ? <I.check size={14} /> : <span className="st-dot" />}</span>
                    <span className="st-n">{STAGE_LABEL[s] ?? s}</span>
                    <span className="meter thin"><i style={{ width: `${frac * 100}%` }} /></span>
                    <span className="st-p mono">{Math.round(frac * 100)}%</span>
                  </li>
                );
              })}
            </ol>
          </>
        )}
        {job?.error && <div className="msg bad pre"><I.error />{job.error}</div>}
        <div className="actions">
          {job && ['queued', 'running'].includes(job.status) && <button className="btn danger" onClick={onCancel}><I.stop />取消</button>}
          {status === 'done' && <button className="btn primary" onClick={onView}><I.score />開始審譜</button>}
        </div>
      </section>
      {manifest && (
        <section className="sec">
          <div className="sec-h"><I.cpu /><span className="sec-t">上次執行</span></div>
          {manifest.warnings?.length > 0 && (
            <ul className="warns">{manifest.warnings.map((w: string, i: number) => <li key={i}><I.warn size={14} />{w}</li>)}</ul>
          )}
          <table className="tbl">
            <thead><tr><th>模型</th><th>裝置</th><th className="r">峰值 VRAM</th><th className="r">耗時</th><th className="r">OOM 重試</th></tr></thead>
            <tbody>
              {Object.entries(manifest.separation ?? {}).filter(([k]) => k !== 'cached').map(([id, s]: [string, any]) => (
                <tr key={id}><td>{id}</td><td>{s.device}</td><td className="r mono">{s.peak_vram_gb} GB</td>
                  <td className="r mono">{s.seconds} s</td><td className="r mono">{s.oom_retries}</td></tr>
              ))}
              {Object.entries(manifest.transcription ?? {}).map(([id, t]: [string, any]) => (
                <tr key={id}><td>{id}</td><td>{t.last_run?.device ?? '–'}</td>
                  <td className="r mono">{t.last_run ? `${t.last_run.peak_vram_gb} GB` : '–'}</td><td className="r mono">{t.seconds} s</td>
                  <td className="r mono">{t.last_run?.oom_retries ?? '–'}</td></tr>
              ))}
            </tbody>
          </table>
          {manifest.separation?.cached && <p className="note"><I.info size={14} />分離結果取自快取。</p>}
        </section>
      )}
    </>
  );
}

function ExportPage({ project, onError }: { project: Project; onError: (e: string | null) => void }) {
  const names: Record<string, string> = {
    gp: 'Guitar Pro 8', gp5: 'Guitar Pro 5', mid: 'MIDI', musicxml: 'MusicXML',
  };
  const order = ['gp', 'gp5', 'mid', 'musicxml'];
  const files = [...project.exports].sort((a, b) => order.indexOf(a.split('.').pop()!) - order.indexOf(b.split('.').pop()!));
  const gp = project.manifest?.exports?.gp;
  const pid = project.meta.id;
  const row = (icon: ReactNode, name: string, file: string, href: string, download: string) => (
    <li key={href}>
      <span className="ex-i">{icon}</span>
      <span className="ex-n">{name}</span>
      <span className="ex-f mono">{file}</span>
      <a className="tbtn" href={href} download={download} aria-label={`下載 ${file}`} title="下載"><I.download /></a>
    </li>
  );
  return (
    <>
      <div className="view-h">
        <h1>匯出</h1>
        <button className="btn" onClick={() => api.openFolder(pid).catch((e) => onError(errText(e)))}><I.folder />開啟輸出資料夾</button>
      </div>
      <section className="sec">
        <div className="sec-h"><I.score /><span className="sec-t">樂譜</span></div>
        <ul className="exports">
          {files.map((f) => row(<I.score />, names[f.split('.').pop()!] ?? f, f,
            api.fileUrl(pid, `run/exports/${encodeURIComponent(f)}`), f))}
        </ul>
        {gp && (
          gp.error ? <div className="msg bad"><I.error />.gp 匯出失敗：{gp.error}</div>
            : gp.verified ? <p className="note"><I.ok size={14} className="good" />alphaTab {gp.alphatab} 讀回與樂譜一致（{gp.notes_written} 個音符頭）。尚未用 Guitar Pro 8 本體驗證。</p>
              : <ul className="warns">{gp.mismatches.map((m: string, i: number) => <li key={i}><I.warn size={14} />{m}</li>)}</ul>
        )}
      </section>
      <section className="sec">
        <div className="sec-h"><I.file /><span className="sec-t">紀錄</span></div>
        <ul className="exports">
          {row(<I.cpu />, '執行紀錄', 'manifest.json', api.fileUrl(pid, 'run/manifest.json'), 'manifest.json')}
          {row(<I.ok />, 'QA 結果', 'qa.json', api.fileUrl(pid, 'run/qa.json'), 'qa.json')}
        </ul>
      </section>
      <p className="note"><I.lock size={14} />輸出不可用於商業用途（MuScriptor：CC BY-NC 4.0；ADTOF：CC BY-NC-SA 4.0）。</p>
    </>
  );
}

function AboutPage({ info }: { info: Info }) {
  const [doctor, setDoctor] = useState<any>(null);
  const [doctorError, setDoctorError] = useState<string | null>(null);
  useEffect(() => { api.doctor().then(setDoctor).catch((e) => setDoctorError(errText(e))); }, []);
  const statusIcon = (s: string) => (s === 'ok' ? <I.ok className="good" /> : s === 'warn' ? <I.warn className="warn" /> : <I.error className="bad" />);
  return (
    <>
      <div className="view-h"><h1>關於</h1></div>
      <section className="sec plain">
        <p className="lead"><b>BandScribe {info.version}</b>　本機執行的樂團錄音轉譜工具。所有處理都在這台電腦上，伺服器只綁定 127.0.0.1。</p>
        <p className="note"><I.lock size={14} />僅限個人、非商業使用：預設流程使用 CC BY-NC 4.0（MuScriptor）與 CC BY-NC-SA 4.0（ADTOF）授權的權重。</p>
      </section>
      <section className="sec">
        <div className="sec-h"><I.cpu /><span className="sec-t">模型與授權</span></div>
        <table className="tbl">
          <thead><tr><th>模型</th><th>用途</th><th>狀態</th><th>程式</th><th>權重</th><th /></tr></thead>
          <tbody>
            {info.models.map((m) => (
              <tr key={m.id}>
                <td>{m.name}<div className="dim small">{m.version ?? ''}</div></td>
                <td>{m.task}</td>
                <td>{m.status}{m.gated && <span className="dim" title="需在 Hugging Face 接受授權"> <I.lock size={12} /></span>}</td>
                <td>{m.license_code ?? <span className="dim">未確認</span>}</td>
                <td>{m.license_weights
                  ? <span className={m.non_commercial ? 'warn' : ''}>{m.license_weights}{m.non_commercial ? '（非商用）' : ''}</span>
                  : <span className="dim">未確認</span>}
                  {m.license_note && <div className="dim small">{m.license_note}</div>}</td>
                <td className="r">{m.source_urls.slice(0, 1).map((u) => (
                  <a key={u} className="tbtn sm" href={u} target="_blank" rel="noreferrer" aria-label="來源" title={u}><I.right size={14} /></a>
                ))}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
      <section className="sec">
        <div className="sec-h"><I.settings /><span className="sec-t">環境檢測</span></div>
        {doctorError ? <div className="msg bad"><I.error />{doctorError}</div> : !doctor ? <div className="dim">檢測中…</div> : (
          <ul className="checklist">
            {doctor.checks.map((c: any) => (
              <li key={c.name}>
                <span className="cl-i" title={c.status}>{statusIcon(c.status)}</span>
                <span className="cl-n">{c.name}</span>
                <span className="cl-s">{c.summary}{c.hint && c.status !== 'ok' ? <span className="dim small">{c.hint}</span> : null}</span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </>
  );
}

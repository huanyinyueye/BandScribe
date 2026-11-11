import { useCallback, useEffect, useRef, useState } from 'react';
import { api, Config, Info, Job, Project, QuickAnalysis, Waveform } from './api';
import { StemsPage } from './Stems';
import { ScorePage } from './ScoreView';

type Page = 'import' | 'settings' | 'progress' | 'stems' | 'score' | 'export' | 'about';
const PAGES: { id: Page; label: string; needsProject?: boolean; needsResult?: boolean }[] = [
  { id: 'import', label: '匯入' },
  { id: 'settings', label: '設定', needsProject: true },
  { id: 'progress', label: '處理進度', needsProject: true },
  { id: 'stems', label: '分軌檢視', needsResult: true },
  { id: 'score', label: '樂譜檢視', needsResult: true },
  { id: 'export', label: '匯出', needsResult: true },
  { id: 'about', label: '關於' },
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

export function App() {
  const [page, setPage] = useState<Page>('import');
  const [info, setInfo] = useState<Info | null>(null);
  const [projectId, setProjectId] = useState<string | null>(null);
  const [project, setProject] = useState<Project | null>(null);
  const [config, setConfig] = useState<Config | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => { api.info().then(setInfo).catch((e) => setError(String(e.message ?? e))); }, []);

  const refresh = useCallback(async (id: string) => {
    const p = await api.project(id);
    setProject(p);
    return p;
  }, []);

  const openProject = useCallback(async (id: string) => {
    setError(null);
    setProjectId(id);
    const p = await refresh(id);
    if (p.job) setJob(p.job);
    else setJob(null);
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
    } catch (e: any) {
      setError(String(e.message ?? e));
    }
  };

  const hasResult = project?.manifest?.status === 'done';
  const running = !!job && ['queued', 'running'].includes(job.status);

  return (
    <div className="app">
      <nav aria-label="頁面">
        <h1>BandScribe</h1>
        <div className="song" title={project?.meta.filename}>{project?.meta.name ?? '尚未匯入'}</div>
        {PAGES.map((p) => (
          <button key={p.id} className={page === p.id ? 'active' : ''}
            aria-current={page === p.id ? 'page' : undefined}
            disabled={(p.needsProject && !project) || (p.needsResult && !hasResult)}
            onClick={() => setPage(p.id)}>
            <span>{p.label}</span>
            {p.id === 'progress' && running && <span className="badge">{Math.round((job?.overall ?? 0) * 100)}%</span>}
          </button>
        ))}
        <div className="foot">v{info?.version ?? '…'}<br />僅綁定 127.0.0.1</div>
      </nav>
      <main>
        {error && <div className="error" role="alert">{error}</div>}
        {page === 'import' && (
          <ImportPage project={project} onOpen={openProject} onError={setError}
            onNext={() => setPage('settings')} busy={running} />
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
        {page === 'score' && project && <ScorePage project={project} />}
        {page === 'export' && project && <ExportPage project={project} onError={setError} />}
        {page === 'about' && info && <AboutPage info={info} />}
      </main>
    </div>
  );
}

function WaveCanvas({ wave }: { wave: Waveform }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = c.clientWidth * dpr;
    c.height = c.clientHeight * dpr;
    const g = c.getContext('2d')!;
    g.clearRect(0, 0, c.width, c.height);
    g.fillStyle = getComputedStyle(c).getPropertyValue('--accent') || '#2456c9';
    const n = wave.min.length;
    const mid = c.height / 2;
    const scale = mid / Math.max(wave.peak, 1e-6);
    for (let i = 0; i < n; i++) {
      const x = (i / n) * c.width;
      const top = mid - wave.max[i] * scale;
      const bottom = mid - wave.min[i] * scale;
      g.fillRect(x, top, Math.max(1, c.width / n), Math.max(1, bottom - top));
    }
  }, [wave]);
  return <canvas ref={ref} className="wave" role="img" aria-label="波形" />;
}

function ImportPage({ project, onOpen, onError, onNext, busy }: {
  project: Project | null; onOpen: (id: string) => Promise<void>; onError: (e: string | null) => void;
  onNext: () => void; busy: boolean;
}) {
  const [over, setOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [quick, setQuick] = useState<QuickAnalysis | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [recent, setRecent] = useState<{ id: string; name: string; status?: string }[]>([]);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => { api.projects().then(setRecent).catch(() => undefined); }, [project?.meta.id]);
  useEffect(() => { setQuick(null); }, [project?.meta.id]);

  const send = async (file: File | undefined) => {
    if (!file) return;
    onError(null);
    setUploading(true);
    try {
      const meta = await api.upload(file);
      await onOpen(meta.id);
    } catch (e: any) {
      onError(`匯入失敗：${e.message ?? e}`);
    } finally {
      setUploading(false);
    }
  };
  const analyze = async () => {
    if (!project) return;
    setAnalyzing(true);
    try { setQuick(await api.analyze(project.meta.id)); } catch (e: any) { onError(String(e.message ?? e)); }
    setAnalyzing(false);
  };
  const a = project?.analysis;

  return (
    <>
      <h2>匯入</h2>
      <div className={`drop ${over ? 'over' : ''}`} role="button" tabIndex={0}
        onClick={() => input.current?.click()}
        onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && input.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setOver(true); }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => { e.preventDefault(); setOver(false); void send(e.dataTransfer.files[0]); }}>
        {uploading ? '匯入中…' : '把 MP3 / WAV / FLAC 拖到這裡，或點擊選擇檔案'}
        <input ref={input} type="file" accept=".mp3,.wav,.flac" hidden
          onChange={(e) => void send(e.target.files?.[0])} />
      </div>
      {project && (
        <div className="panel" style={{ marginTop: 16 }}>
          <div className="row" style={{ justifyContent: 'space-between' }}>
            <b>{project.meta.filename}</b>
            <span className="muted">
              {fmtTime(project.waveform.duration)}・{project.waveform.orig_sr} Hz・{project.waveform.orig_channels} 聲道
            </span>
          </div>
          <WaveCanvas wave={project.waveform} />
          <audio controls src={api.fileUrl(project.meta.id, `input${project.meta.filename.slice(project.meta.filename.lastIndexOf('.')).toLowerCase()}`)}
            style={{ width: '100%', marginTop: 8 }} />
          <h3>初步分析</h3>
          {a ? (
            <div className="stats">
              <div className="stat"><span className="muted">BPM</span><b>{a.global_bpm}</b></div>
              <div className="stat"><span className="muted">拍號</span><b>{a.meter.dominant[0]}/{a.meter.dominant[1]}</b>
                <span className="small muted">{a.meter.changes.length > 1 ? `含 ${a.meter.changes.length - 1} 次變拍` : '無變拍'}</span></div>
              <div className="stat"><span className="muted">調性（由音符推估）</span><b>{a.key?.name ?? '—'}</b></div>
              <div className="stat"><span className="muted">小節數</span><b>{a.bars.length}</b></div>
            </div>
          ) : quick ? (
            <div className="stats">
              <div className="stat"><span className="muted">BPM</span><b>{quick.bpm}</b>
                <span className="small muted">{quick.tempo_changes ? `含 ${quick.tempo_changes} 次變速` : '速度穩定'}</span></div>
              <div className="stat"><span className="muted">拍號</span><b>{quick.meter[0]}/{quick.meter[1]}</b>
                <span className="small muted">{quick.meter_changes ? `含 ${quick.meter_changes} 次變拍` : '無變拍'}</span></div>
              <div className="stat"><span className="muted">調性（粗估）</span><b>{quick.key_guess}</b>
                <span className="small muted">完整處理後會依音符重新推估</span></div>
            </div>
          ) : (
            <button className="btn" disabled={analyzing || busy} onClick={analyze}>
              {analyzing ? '分析中…' : '分析 BPM / 拍號 / 調性'}
            </button>
          )}
          <div className="row" style={{ marginTop: 16 }}>
            <button className="btn primary" onClick={onNext}>下一步：設定</button>
          </div>
        </div>
      )}
      {recent.length > 0 && (
        <div className="panel">
          <h3 style={{ marginTop: 0 }}>最近的專案</h3>
          <table>
            <tbody>
              {recent.slice(0, 8).map((r) => (
                <tr key={r.id}>
                  <td>{r.name}</td>
                  <td className="muted">{r.status === 'done' ? '已完成' : r.status === 'new' ? '尚未處理' : r.status}</td>
                  <td style={{ textAlign: 'right' }}><button className="btn" onClick={() => void onOpen(r.id)}>開啟</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
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
  const mus = config.pitched_transcribers.find((t) => t.startsWith('muscriptor_')) ?? '';
  const setMus = (id: string) => {
    const swap = (list: string[]) => {
      const rest = list.filter((t) => !t.startsWith('muscriptor_'));
      return id ? [id, ...rest] : rest;
    };
    setConfig({ ...config, pitched_transcribers: swap(config.pitched_transcribers),
      drum_transcribers: swap(config.drum_transcribers),
      mix_vote: config.mix_vote ? (id || null) : null });
  };
  return (
    <>
      <h2>設定</h2>
      <div className="panel grid2">
        <label className="field">品質設定檔
          <select value={config.profile} onChange={(e) => setConfig({ ...config, ...info.defaults[e.target.value],
            title: config.title, stem_overrides: config.stem_overrides })}>
            <option value="quality">quality（雙分離模型交叉驗證＋混音投票，較慢）</option>
            <option value="balanced">balanced（單一分離模型，較快）</option>
          </select>
        </label>
        <label className="field">標題
          <input type="text" value={config.title ?? ''} placeholder="預設為檔名"
            onChange={(e) => set('title', e.target.value || null)} />
        </label>
      </div>
      <div className="panel">
        <h3 style={{ marginTop: 0 }}>模型</h3>
        <div className="grid2">
          <label className="field">主分離模型（吉他／鋼琴軌的唯一來源）
            <select value={config.separator_primary} onChange={(e) => set('separator_primary', e.target.value)}>
              {seps.map((m) => <option key={m.id} value={m.id}>{label(m.id)}</option>)}
            </select>
          </label>
          <label className="field">交叉驗證分離模型（鼓／貝斯／人聲）
            <select value={config.separator_secondary ?? ''} onChange={(e) => set('separator_secondary', e.target.value || null)}>
              <option value="">不使用</option>
              {seps.filter((m) => m.id !== config.separator_primary && m.id !== 'mvsep_mega53')
                .map((m) => <option key={m.id} value={m.id}>{label(m.id)}</option>)}
            </select>
          </label>
          <label className="field">MuScriptor 模型大小
            <select value={mus} onChange={(e) => setMus(e.target.value)}>
              <option value="">不使用（只用 Basic Pitch / ADTOF）</option>
              <option value="muscriptor_small">small（103M）</option>
              <option value="muscriptor_medium">medium（307M，預設）</option>
              <option value="muscriptor_large">large（1.4B，需較多 VRAM）</option>
            </select>
          </label>
          <label className="check" style={{ alignSelf: 'end' }}>
            <input type="checkbox" checked={!!config.mix_vote} disabled={!mus}
              onChange={(e) => set('mix_vote', e.target.checked ? mus : null)} />
            以原始混音轉譜做樂器歸屬投票
          </label>
        </div>
        <p className="small muted">MuScriptor 權重需在 Hugging Face 逐一接受授權；無法取得時該模型會被略過並在結果中註明。</p>
      </div>
      <div className="panel">
        <h3 style={{ marginTop: 0 }}>樂器與調弦</h3>
        <div className="row">
          {info.defaults.quality.instruments.map((f) => (
            <label key={f} className="check">
              <input type="checkbox" checked={config.instruments.includes(f)}
                onChange={() => set('instruments', toggle(config.instruments, f))} />
              {familyLabel(f)}
            </label>
          ))}
        </div>
        <div className="grid2" style={{ marginTop: 12 }}>
          {(['guitar', 'bass'] as const).map((inst) => (
            <label key={inst} className="field">{familyLabel(inst)}調弦
              <select value={inst === 'guitar' ? config.guitar_tuning : config.bass_tuning}
                onChange={(e) => set(inst === 'guitar' ? 'guitar_tuning' : 'bass_tuning', e.target.value)}>
                <option value="auto">自動偵測（依最低音建議）</option>
                {info.tunings.filter((t) => t.instrument === inst).map((t) => <option key={t.key} value={t.key}>{t.name}</option>)}
              </select>
            </label>
          ))}
        </div>
      </div>
      <div className="panel grid2">
        <label className="field">低信心門檻：{config.low_confidence.toFixed(2)}
          <input type="range" min={0} max={1} step={0.05} value={config.low_confidence}
            onChange={(e) => set('low_confidence', Number(e.target.value))} />
          <span className="small muted">低於門檻的音符會以「?」與顏色標示</span>
        </label>
        <label className="check" style={{ alignSelf: 'center' }}>
          <input type="checkbox" checked={config.strict} onChange={(e) => set('strict', e.target.checked)} />
          strict 模式（直接移除低信心音符）
        </label>
      </div>
      <button className="btn primary" disabled={running || config.instruments.length === 0} onClick={onStart}>
        {running ? '處理中…' : '開始轉譜'}
      </button>
    </>
  );
}

function ProgressPage({ job, project, onCancel, onView }: {
  job: Job | null; project: Project | null; onCancel: () => void; onView: () => void;
}) {
  const manifest = project?.manifest;
  if (!job && !manifest) return (<><h2>處理進度</h2><p className="muted">尚未開始。請到「設定」按「開始轉譜」。</p></>);
  const stages = job?.stages ?? Object.keys(STAGE_LABEL);
  const idx = job ? stages.indexOf(job.stage) : stages.length;
  const status = job?.status ?? manifest?.status;
  const statusText: Record<string, string> = {
    queued: '排隊中', running: '處理中', done: '完成', failed: '失敗', cancelled: '已取消',
  };
  return (
    <>
      <h2>處理進度</h2>
      <div className="panel">
        <div className="row" style={{ justifyContent: 'space-between' }}>
          <b>{statusText[status] ?? status}</b>
          {job && (
            <span className="muted">
              已用 {fmtTime(job.elapsed)}
              {job.eta != null && job.status === 'running' ? `・預估剩餘 ${fmtTime(job.eta)}` : ''}
              ・VRAM {job.vram_gb.toFixed(2)} GB
            </span>
          )}
        </div>
        {job && (
          <>
            <div className="bar" style={{ margin: '10px 0' }} role="progressbar" aria-valuemin={0} aria-valuemax={100}
              aria-valuenow={Math.round(job.overall * 100)}><i style={{ width: `${job.overall * 100}%` }} /></div>
            <div className="muted small" aria-live="polite">{job.message}</div>
            <div className="stages" style={{ marginTop: 14 }}>
              {stages.map((s, i) => {
                const done = job.status === 'done' || i < idx;
                const frac = done ? 1 : i === idx ? job.stage_fraction : 0;
                return [
                  <span key={`${s}l`} className={i === idx && job.status === 'running' ? 'cur' : ''}>{STAGE_LABEL[s] ?? s}</span>,
                  <div key={`${s}b`} className="bar"><i style={{ width: `${frac * 100}%` }} /></div>,
                  <span key={`${s}p`} className="muted small">{Math.round(frac * 100)}%</span>,
                ];
              })}
            </div>
          </>
        )}
        {job?.error && <div className="error" style={{ marginTop: 12 }}>{job.error}</div>}
        <div className="row" style={{ marginTop: 14 }}>
          {job && ['queued', 'running'].includes(job.status) && <button className="btn danger" onClick={onCancel}>取消</button>}
          {status === 'done' && <button className="btn primary" onClick={onView}>檢視樂譜</button>}
        </div>
      </div>
      {manifest && (
        <div className="panel">
          <h3 style={{ marginTop: 0 }}>上次執行</h3>
          {manifest.warnings?.length > 0 && <ul className="warnings">{manifest.warnings.map((w: string, i: number) => <li key={i}>{w}</li>)}</ul>}
          <table>
            <thead><tr><th>模型</th><th>裝置</th><th>峰值 VRAM</th><th>耗時</th><th>OOM 重試</th></tr></thead>
            <tbody>
              {Object.entries(manifest.separation ?? {}).filter(([k]) => k !== 'cached').map(([id, s]: [string, any]) => (
                <tr key={id}><td>{id}</td><td>{s.device}</td><td>{s.peak_vram_gb} GB</td><td>{s.seconds} s</td><td>{s.oom_retries}</td></tr>
              ))}
              {Object.entries(manifest.transcription ?? {}).map(([id, t]: [string, any]) => (
                <tr key={id}><td>{id}</td><td>{t.last_run?.device ?? '—'}</td>
                  <td>{t.last_run ? `${t.last_run.peak_vram_gb} GB` : '—'}</td><td>{t.seconds} s</td><td>{t.last_run?.oom_retries ?? '—'}</td></tr>
              ))}
            </tbody>
          </table>
          {manifest.separation?.cached && <p className="small muted">分離結果取自快取。</p>}
        </div>
      )}
    </>
  );
}

function ExportPage({ project, onError }: { project: Project; onError: (e: string | null) => void }) {
  const names: Record<string, string> = {
    gp: 'Guitar Pro 8（.gp）— 主要格式', gp5: 'Guitar Pro 5（.gp5）— 後備', mid: 'MIDI', musicxml: 'MusicXML',
  };
  const order = ['gp', 'gp5', 'mid', 'musicxml'];
  const files = [...project.exports].sort((a, b) => order.indexOf(a.split('.').pop()!) - order.indexOf(b.split('.').pop()!));
  const gp = project.manifest?.exports?.gp;
  return (
    <>
      <h2>匯出</h2>
      <div className="panel">
        <table>
          <thead><tr><th>格式</th><th>檔案</th><th /></tr></thead>
          <tbody>
            {files.map((f) => (
              <tr key={f}>
                <td>{names[f.split('.').pop()!] ?? f}</td><td>{f}</td>
                <td style={{ textAlign: 'right' }}>
                  <a className="btn" href={api.fileUrl(project.meta.id, `run/exports/${encodeURIComponent(f)}`)} download={f}>下載</a>
                </td>
              </tr>
            ))}
            <tr><td>執行紀錄（模型、sha256、VRAM、耗時）</td><td>manifest.json</td>
              <td style={{ textAlign: 'right' }}><a className="btn" href={api.fileUrl(project.meta.id, 'run/manifest.json')} download>下載</a></td></tr>
            <tr><td>QA 結果</td><td>qa.json</td>
              <td style={{ textAlign: 'right' }}><a className="btn" href={api.fileUrl(project.meta.id, 'run/qa.json')} download>下載</a></td></tr>
          </tbody>
        </table>
        <div className="row" style={{ marginTop: 14 }}>
          <button className="btn primary" onClick={() => api.openFolder(project.meta.id).catch((e) => onError(String(e.message ?? e)))}>
            開啟輸出資料夾
          </button>
        </div>
      </div>
      {gp && (
        <div className="panel">
          <h3 style={{ marginTop: 0 }}>.gp 檔讀回檢查</h3>
          {gp.error ? <div className="error">.gp 匯出失敗：{gp.error}</div>
            : gp.verified ? <p>alphaTab {gp.alphatab} 重新讀回後與樂譜一致（{gp.notes_written} 個音符頭）。尚未以 Guitar Pro 8 本體驗證，請自行開檔確認。</p>
              : <ul className="warnings">{gp.mismatches.map((m: string, i: number) => <li key={i}>{m}</li>)}</ul>}
        </div>
      )}
      <p className="muted small">輸出不可用於商業用途（MuScriptor：CC BY-NC 4.0；ADTOF：CC BY-NC-SA 4.0）。</p>
    </>
  );
}

function AboutPage({ info }: { info: Info }) {
  const [doctor, setDoctor] = useState<any>(null);
  useEffect(() => { api.doctor().then(setDoctor).catch(() => undefined); }, []);
  return (
    <>
      <h2>關於</h2>
      <div className="panel">
        <p><b>BandScribe {info.version}</b>：本機執行的樂團音樂自動轉譜工具。所有處理都在這台電腦上進行，伺服器只綁定 127.0.0.1。</p>
        <p className="lowkey">僅限個人、非商業使用：預設流程使用 CC BY-NC 4.0（MuScriptor）與 CC BY-NC-SA 4.0（ADTOF）授權的權重，輸出不可用於商業用途。</p>
      </div>
      <div className="panel">
        <h3 style={{ marginTop: 0 }}>模型與授權</h3>
        <table>
          <thead><tr><th>模型</th><th>用途</th><th>狀態</th><th>程式授權</th><th>權重授權</th><th>來源</th></tr></thead>
          <tbody>
            {info.models.map((m) => (
              <tr key={m.id}>
                <td>{m.name}<div className="small muted">{m.version ?? ''}</div></td>
                <td>{m.task}</td><td>{m.status}{m.gated ? '・需 HF 授權' : ''}</td>
                <td>{m.license_code ?? '未確認'}</td>
                <td>{m.license_weights
                  ? <span className={`pill ${m.non_commercial ? 'nc' : ''}`}>{m.license_weights}</span>
                  : <span className="muted">未確認</span>}
                  <div className="small muted">{m.license_note}</div></td>
                <td>{m.source_urls.slice(0, 1).map((u) => <a key={u} href={u} target="_blank" rel="noreferrer">連結</a>)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {doctor && (
        <div className="panel">
          <h3 style={{ marginTop: 0 }}>環境檢測</h3>
          <table>
            <tbody>
              {doctor.checks.map((c: any) => (
                <tr key={c.name}><td>{c.name}</td>
                  <td style={{ color: c.status === 'ok' ? 'var(--good)' : c.status === 'warn' ? 'var(--warn)' : 'var(--bad)' }}>{c.status}</td>
                  <td>{c.summary}{c.hint && c.status !== 'ok' ? <div className="small muted">{c.hint}</div> : null}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}

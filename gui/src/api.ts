// Thin client for the BandScribe backend (bandscribe/gui/server.py).

export type Waveform = {
  duration: number; orig_sr: number; orig_channels: number; sha256: string;
  min: number[]; max: number[]; peak: number;
};
export type ProjectMeta = { id: string; name: string; filename: string; created: number; status?: string };
export type Job = {
  id: string; project: string; status: 'queued' | 'running' | 'done' | 'failed' | 'cancelled';
  stage: string; stage_fraction: number; overall: number; message: string; error: string | null;
  elapsed: number; eta: number | null; vram_gb: number; stages: string[]; version: number;
};
export type ModelInfo = {
  id: string; name: string; task: string; status: string; version: string | null;
  license_code: string | null; license_weights: string | null; license_note: string;
  non_commercial: boolean; gated: boolean; source_urls: string[]; vram_note: string;
};
export type Tuning = { key: string; name: string; instrument: string; strings: number[] };
export type Config = {
  profile: string; separator_primary: string; separator_secondary: string | null;
  pitched_transcribers: string[]; drum_transcribers: string[]; mix_vote: string | null;
  low_confidence: number; strict: boolean; guitar_tuning: string; bass_tuning: string;
  instruments: string[]; exports: string[]; title: string | null; max_span: number;
  stem_overrides: Record<string, string>;
};
export type Info = {
  version: string; defaults: Record<string, Config>; tunings: Tuning[]; models: ModelInfo[];
  notice: string;
};
export type StemInfo = {
  model_id: string; weights: Record<string, number> | null;
  confidence: { window: number; values: number[]; active: boolean[]; source: string } | null;
  activity?: { present: boolean; rms_db: number; relative_db: number; active_seconds: number; active_fraction: number };
};
export type BarInfo = {
  index: number; numerator: number; denominator: number; start_time: number; end_time?: number;
  bpm: number; section: string | null; confident: boolean;
};
export type ScoreSummary = {
  title: string; key: { name: string; margin: number } | null; bars: BarInfo[];
  tracks: { id: string; name: string; family: string; kind: string; tuning_key: string | null;
    info: Record<string, unknown>; notes: number; low_conf: number }[];
  meta: Record<string, any>;
};
export type Project = {
  meta: ProjectMeta; waveform: Waveform; config: Partial<Config> | null; manifest: any | null;
  analysis: any | null; stems: { stems: Record<string, StemInfo> } | null; qa: any | null;
  attribution: { removed: any[]; promoted: any[] } | null; score: ScoreSummary | null;
  exports: string[]; job: Job | null;
};
export type QuickAnalysis = {
  bpm: number; tempo_changes: number; meter: [number, number]; meter_changes: number;
  beats: number; key_guess: string; key_note: string;
};

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch { /* not JSON */ }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return res.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
});

export const api = {
  info: () => request<Info>('/api/info'),
  doctor: () => request<any>('/api/doctor'),
  projects: () => request<ProjectMeta[]>('/api/projects'),
  upload: (file: File) => {
    const form = new FormData();
    form.append('file', file);
    return request<ProjectMeta & { waveform: Waveform }>('/api/projects', { method: 'POST', body: form });
  },
  project: (id: string) => request<Project>(`/api/projects/${id}`),
  analyze: (id: string) => request<QuickAnalysis>(`/api/projects/${id}/analyze`, { method: 'POST' }),
  run: (id: string, config: Partial<Config>) => request<Job>(`/api/projects/${id}/run`, json({ config })),
  job: (id: string) => request<Job>(`/api/jobs/${id}`),
  cancel: (id: string) => request<Job>(`/api/jobs/${id}/cancel`, { method: 'POST' }),
  override: (id: string, stem: string, state: string | null) =>
    request<{ stem_overrides: Record<string, string> }>(`/api/projects/${id}/stems/${stem}/override`, json({ state })),
  openFolder: (id: string) => request<{ path: string }>(`/api/projects/${id}/open-folder`, { method: 'POST' }),
  fileUrl: (id: string, path: string) => `/api/projects/${id}/files/${path}`,
  watchJob(id: string, onUpdate: (job: Job) => void): () => void {
    let stopped = false;
    let ws: WebSocket | null = null;
    const poll = async () => {  // fallback when the WebSocket cannot be opened
      while (!stopped) {
        try {
          const j = await api.job(id);
          onUpdate(j);
          if (['done', 'failed', 'cancelled'].includes(j.status)) return;
        } catch { /* keep trying */ }
        await new Promise((r) => setTimeout(r, 700));
      }
    };
    try {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws';
      ws = new WebSocket(`${proto}://${location.host}/api/ws/jobs/${id}`);
      ws.onmessage = (ev) => onUpdate(JSON.parse(ev.data));
      ws.onerror = () => { ws = null; void poll(); };
    } catch {
      void poll();
    }
    return () => { stopped = true; ws?.close(); };
  },
};

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
  bpm: number; section: string | null; confident: boolean; beat_times?: number[];
};
export type ScoreSummary = {
  title: string; key: { name: string; margin: number } | null; bars: BarInfo[];
  tracks: { id: string; name: string; family: string; kind: string; tuning_key: string | null;
    info: Record<string, unknown>; notes: number; low_conf: number }[];
  meta: Record<string, any>;
};
// An existing Guitar Pro score the user attached; path is relative to the project folder.
export type Reference = { filename: string; path: string };
export type Project = {
  meta: ProjectMeta; waveform: Waveform; config: Partial<Config> | null; manifest: any | null;
  analysis: any | null; stems: { stems: Record<string, StemInfo> } | null; qa: any | null;
  attribution: { removed: any[]; promoted: any[] } | null; score: ScoreSummary | null;
  reference: Reference | null; exports: string[]; job: Job | null;
};
export type QuickAnalysis = {
  bpm: number; tempo_changes: number; meter: [number, number]; meter_changes: number;
  beats: number; key_guess: string; key_note: string;
};

// beat = quarter notes from the start of the score (alphaTab tick / 960); seconds = time in the recording.
export type AlignmentAnchor = { beat: number; seconds: number };
export type Alignment = { score_hash: string; offset: number; anchors: AlignmentAnchor[]; revision: number; stale: boolean };

// One bar of one guitar track. pos_q / dur_q are bar-relative quarter notes; string 1 is the highest
// string and tuning[string - 1] its open MIDI pitch. index is the note's index in the whole track.
export type CorrectionNote = {
  index: number; pitch: number; string: number | null; fret: number | null; pos_q: number; dur_q: number;
};
export type CorrectionBar = {
  score_hash: string; revision: number; track_id: string; bar: number; tuning: number[]; notes: CorrectionNote[];
};
// Times are exact rational strings such as "3/2"; omitted times keep the stored value.
export type CorrectionUpdate = { index: number; string: number | null; fret: number | null; pos_q?: string; dur_q?: string };
export type CorrectionAdd = { string: number; fret: number; pos_q: string; dur_q: string };
export type CorrectionSave = {
  score_hash: string; revision: number; notes: CorrectionUpdate[]; delete?: number[]; add?: CorrectionAdd[];
};

export class ApiError extends Error {
  constructor(message: string, readonly status: number) { super(message); }
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch { /* not JSON */ }
    throw new ApiError(typeof detail === 'string' ? detail : JSON.stringify(detail), res.status);
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
  uploadReference: (id: string, file: File) => {
    const form = new FormData();
    form.append('file', file);
    return request<Reference>(`/api/projects/${id}/reference`, { method: 'POST', body: form });
  },
  clearReference: (id: string) => request<{ cleared: boolean }>(`/api/projects/${id}/reference`, { method: 'DELETE' }),
  analyze: (id: string) => request<QuickAnalysis>(`/api/projects/${id}/analyze`, { method: 'POST' }),
  run: (id: string, config: Partial<Config>) => request<Job>(`/api/projects/${id}/run`, json({ config })),
  job: (id: string) => request<Job>(`/api/jobs/${id}`),
  cancel: (id: string) => request<Job>(`/api/jobs/${id}/cancel`, { method: 'POST' }),
  override: (id: string, stem: string, state: string | null) =>
    request<{ stem_overrides: Record<string, string> }>(`/api/projects/${id}/stems/${stem}/override`, json({ state })),
  alignment: (id: string) => request<Alignment>(`/api/projects/${id}/alignment`),
  saveAlignment: (id: string, body: Omit<Alignment, 'stale'>) =>
    request<Alignment>(`/api/projects/${id}/alignment`, { ...json(body), method: 'PUT' }),
  corrections: (id: string, track: string, bar: number) =>
    request<CorrectionBar>(`/api/projects/${id}/corrections/${encodeURIComponent(track)}/${bar}`),
  saveCorrections: (id: string, track: string, bar: number, body: CorrectionSave) =>
    request<CorrectionBar>(`/api/projects/${id}/corrections/${encodeURIComponent(track)}/${bar}`,
      { ...json(body), method: 'PUT' }),
  exportCorrections: (id: string) =>
    request<{ path: string }>(`/api/projects/${id}/corrections/export`, { method: 'POST' }),
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

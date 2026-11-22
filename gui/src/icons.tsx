import type { ReactNode } from 'react';

// One 16px grid, 1.5px strokes, round joins. Glyphs that read better solid set `solid`.
type P = { size?: number; className?: string };
function S({ size = 16, className, solid, children }: P & { solid?: boolean; children: ReactNode }) {
  return (
    <svg viewBox="0 0 16 16" width={size} height={size} className={className} aria-hidden="true" focusable="false"
      fill={solid ? 'currentColor' : 'none'} stroke={solid ? 'none' : 'currentColor'}
      strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
      {children}
    </svg>
  );
}

const head = (filled: boolean) => (
  <ellipse cx="6" cy="12" rx="2.6" ry="1.8" transform="rotate(-22 6 12)"
    fill={filled ? 'currentColor' : 'none'} stroke="currentColor" strokeWidth={filled ? 0 : 1.4} />
);
const stem = <path d="M8.35 11.4V2.5" />;
const flag = (y: number) => <path key={y} d={`M8.35 ${y}c.4 1.6 3 2.1 3 4.1`} />;

export const I = {
  play: (p: P) => <S {...p} solid><path d="M5 3.1v9.8c0 .4.4.6.7.4l7.4-4.9a.5.5 0 0 0 0-.8L5.7 2.7c-.3-.2-.7 0-.7.4z" /></S>,
  pause: (p: P) => <S {...p} solid><rect x="4" y="3" width="2.8" height="10" rx=".6" /><rect x="9.2" y="3" width="2.8" height="10" rx=".6" /></S>,
  stop: (p: P) => <S {...p} solid><rect x="4" y="4" width="8" height="8" rx="1" /></S>,
  skipBack: (p: P) => <S {...p}><path d="M4 3.5v9M12 3.75 6.5 8 12 12.25z" /></S>,
  skipFwd: (p: P) => <S {...p}><path d="M12 3.5v9M4 3.75 9.5 8 4 12.25z" /></S>,
  loop: (p: P) => <S {...p}><path d="M2.75 7.5V7A2.75 2.75 0 0 1 5.5 4.25h7.25M10.75 2.25l2 2-2 2M13.25 8.5V9a2.75 2.75 0 0 1-2.75 2.75H3.25M5.25 13.75l-2-2 2-2" /></S>,
  left: (p: P) => <S {...p}><path d="M10 3.5 5.5 8l4.5 4.5" /></S>,
  right: (p: P) => <S {...p}><path d="M6 3.5 10.5 8 6 12.5" /></S>,
  up: (p: P) => <S {...p}><path d="M4 10l4-4 4 4" /></S>,
  down: (p: P) => <S {...p}><path d="M4 6l4 4 4-4" /></S>,
  warn: (p: P) => <S {...p}><path d="M7.13 2.75a1 1 0 0 1 1.74 0l5.4 9.5a1 1 0 0 1-.87 1.5H2.6a1 1 0 0 1-.87-1.5zM8 6.5v3M8 11.75v.01" /></S>,
  error: (p: P) => <S {...p}><circle cx="8" cy="8" r="6" /><path d="M5.75 5.75l4.5 4.5M10.25 5.75l-4.5 4.5" /></S>,
  ok: (p: P) => <S {...p}><circle cx="8" cy="8" r="6" /><path d="M5.25 8.25l1.9 1.9 3.6-3.9" /></S>,
  align: (p: P) => <S {...p}><path d="M2.75 2.5v11M13.25 2.5v11M5 8h6M7.25 5.75 5 8l2.25 2.25M8.75 5.75 11 8l-2.25 2.25" /></S>,
  anchor: (p: P) => <S {...p}><path d="M8 5.25V14M5 8h6M2.75 9.5a5.25 5.25 0 0 0 10.5 0" /><circle cx="8" cy="3.5" r="1.75" /></S>,
  edit: (p: P) => <S {...p}><path d="M10.25 2.75a1.4 1.4 0 0 1 2 0l1 1a1.4 1.4 0 0 1 0 2L6 13H3v-3zM9 4l3 3" /></S>,
  plus: (p: P) => <S {...p}><path d="M8 3.25v9.5M3.25 8h9.5" /></S>,
  minus: (p: P) => <S {...p}><path d="M3.25 8h9.5" /></S>,
  trash: (p: P) => <S {...p}><path d="M2.75 4.5h10.5M6.25 4.5V3h3.5v1.5M4.25 4.5l.6 8.6a1 1 0 0 0 1 .9h4.3a1 1 0 0 0 1-.9l.6-8.6M6.75 7v4.25M9.25 7v4.25" /></S>,
  undo: (p: P) => <S {...p}><path d="M5.25 3.25 2.5 6l2.75 2.75M2.5 6h7.25a3.5 3.5 0 0 1 0 7H6.5" /></S>,
  save: (p: P) => <S {...p}><path d="M2.5 3.5a1 1 0 0 1 1-1h7.25l2.75 2.75v7.25a1 1 0 0 1-1 1h-9a1 1 0 0 1-1-1z" /><path d="M5 2.5v3h5v-3M5 13.5V9.5h6v4" /></S>,
  check: (p: P) => <S {...p}><path d="M3 8.5l3.25 3.25L13 5" /></S>,
  close: (p: P) => <S {...p}><path d="M4 4l8 8M12 4l-8 8" /></S>,
  download: (p: P) => <S {...p}><path d="M8 2.5v8M4.75 7.25 8 10.5l3.25-3.25M3 13.5h10" /></S>,
  upload: (p: P) => <S {...p}><path d="M8 10.5v-8M4.75 5.75 8 2.5l3.25 3.25M3 13.5h10" /></S>,
  folder: (p: P) => <S {...p}><path d="M2 4.5a1 1 0 0 1 1-1h3l1.5 1.75H13a1 1 0 0 1 1 1V12a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1z" /></S>,
  score: (p: P) => <S {...p}><path d="M3.5 2.5a1 1 0 0 1 1-1h5l3 3v9a1 1 0 0 1-1 1h-7a1 1 0 0 1-1-1zM9.5 1.5v3h3M5.75 8h4.5M5.75 10h4.5M5.75 12h4.5" /></S>,
  corrected: (p: P) => <S {...p}><path d="M3.5 2.5a1 1 0 0 1 1-1h5l3 3v3M9.5 1.5v3h3M5.75 8h3M5.75 10h1.5M3.5 4v9.5a1 1 0 0 0 1 1h2" /><path d="M8.75 12.5l1.6 1.6 3.4-3.6" /></S>,
  book: (p: P) => <S {...p}><path d="M8 4.25C6.75 3.25 4.75 2.75 2.5 2.75v9.5c2.25 0 4.25.5 5.5 1.5 1.25-1 3.25-1.5 5.5-1.5v-9.5c-2.25 0-4.25.5-5.5 1.5zM8 4.25v9.5" /></S>,
  lock: (p: P) => <S {...p}><rect x="3.5" y="7" width="9" height="6.5" rx="1" /><path d="M5.5 7V5a2.5 2.5 0 0 1 5 0v2" /></S>,
  song: (p: P) => <S {...p}><path d="M2.5 7v2M5 5v6M7.5 2.5v11M10 5.5v5M12.5 4v8M15 7.25v1.5" /></S>,
  refresh: (p: P) => <S {...p}><path d="M13 8a5 5 0 1 1-1.46-3.54M13.25 2.5v3h-3" /></S>,
  target: (p: P) => <S {...p}><circle cx="8" cy="8" r="4.25" /><path d="M8 1.5v2.25M8 12.25v2.25M1.5 8h2.25M12.25 8h2.25" /><circle cx="8" cy="8" r=".6" fill="currentColor" /></S>,
  headphones: (p: P) => <S {...p}><path d="M2.75 11V8.5a5.25 5.25 0 0 1 10.5 0V11" /><rect x="2.25" y="9.75" width="2.5" height="4" rx=".8" /><rect x="11.25" y="9.75" width="2.5" height="4" rx=".8" /></S>,
  zoomIn: (p: P) => <S {...p}><circle cx="7" cy="7" r="4.5" /><path d="M10.25 10.25 14 14M5 7h4M7 5v4" /></S>,
  zoomOut: (p: P) => <S {...p}><circle cx="7" cy="7" r="4.5" /><path d="M10.25 10.25 14 14M5 7h4" /></S>,
  layers: (p: P) => <S {...p}><path d="M8 2 14 5 8 8 2 5zM2 8l6 3 6-3M2 11l6 3 6-3" /></S>,
  info: (p: P) => <S {...p}><circle cx="8" cy="8" r="6" /><path d="M8 7.25v4M8 4.9v.01" /></S>,
  settings: (p: P) => <S {...p}><path d="M2.5 4.5h6M11.5 4.5h2M2.5 11.5h2M7.5 11.5h6" /><circle cx="10" cy="4.5" r="1.5" /><circle cx="6" cy="11.5" r="1.5" /></S>,
  file: (p: P) => <S {...p}><path d="M3.5 2.5a1 1 0 0 1 1-1h5l3 3v9a1 1 0 0 1-1 1h-7a1 1 0 0 1-1-1zM9.5 1.5v3h3" /></S>,
  keyboard: (p: P) => <S {...p}><rect x="1.5" y="4" width="13" height="8" rx="1.25" /><path d="M4 6.5h.01M6.5 6.5h.01M9 6.5h.01M11.5 6.5h.01M5 9.5h6" /></S>,
  cpu: (p: P) => <S {...p}><rect x="4" y="4" width="8" height="8" rx="1" /><path d="M6.5 4V2M9.5 4V2M6.5 14v-2M9.5 14v-2M4 6.5H2M4 9.5H2M14 6.5h-2M14 9.5h-2" /></S>,
  // Instrument families.
  guitar: (p: P) => (
    <S {...p}>
      <path d="M13.5 2.5 9 7" />
      <path d="M8.6 6.1a2.6 2.6 0 0 0-3.3.6c-.5.6-.5 1.2-1.3 1.4-1.2.3-2 1.2-2 2.4 0 1.9 1.7 3.4 3.5 3.1 1.2-.2 1.8-1 2.2-2.1.3-.7.8-.8 1.4-1.3a2.6 2.6 0 0 0 .6-3.3" />
      <circle cx="6.1" cy="10" r="1" />
    </S>
  ),
  bass: (p: P) => (
    <S {...p}>
      <path d="M14 2 9.25 6.75" />
      <path d="M8.6 6.1a2.6 2.6 0 0 0-3.3.6c-.5.6-.5 1.2-1.3 1.4-1.2.3-2 1.2-2 2.4 0 1.9 1.7 3.4 3.5 3.1 1.2-.2 1.8-1 2.2-2.1.3-.7.8-.8 1.4-1.3a2.6 2.6 0 0 0 .6-3.3" />
      <path d="M12.25 2.25l1.5 1.5" />
    </S>
  ),
  drums: (p: P) => <S {...p}><ellipse cx="8" cy="6.5" rx="5.5" ry="2.25" /><path d="M2.5 6.5v4.5c0 1.25 2.46 2.25 5.5 2.25s5.5-1 5.5-2.25V6.5M6 1.75l2 3M12.5 1.5 10 4.5" /></S>,
  piano: (p: P) => <S {...p}><rect x="2" y="3" width="12" height="10" rx="1" /><path d="M5.5 3v6M8 3v6M10.5 3v6M5.5 9v4M10.5 9v4" /></S>,
  vocals: (p: P) => <S {...p}><rect x="6" y="2" width="4" height="7.5" rx="2" /><path d="M3.75 8a4.25 4.25 0 0 0 8.5 0M8 12.25V14" /></S>,
  note: (p: P) => <S {...p}><path d="M6.5 12V3.5l6-1.5v8.5" /><ellipse cx="5" cy="12" rx="1.6" ry="1.25" fill="currentColor" stroke="none" /><ellipse cx="11" cy="10.5" rx="1.6" ry="1.25" fill="currentColor" stroke="none" /></S>,
  logo: (p: P) => (
    <S {...p}>
      <path d="M2 4h12M2 8h12M2 12h12" strokeWidth={1.2} />
      <rect x="4" y="2.25" width="3.5" height="3.5" rx=".6" fill="currentColor" stroke="none" />
      <rect x="9" y="6.25" width="3.5" height="3.5" rx=".6" fill="currentColor" stroke="none" />
      <rect x="5.5" y="10.25" width="3.5" height="3.5" rx=".6" fill="currentColor" stroke="none" />
    </S>
  ),
};

// Note-value glyphs for the duration picker: 4 = whole ... 1/8 = thirty-second (quarter notes).
export function NoteGlyph({ value, size = 18 }: { value: number; size?: number }) {
  const flags = value === 0.5 ? [2.5] : value === 0.25 ? [2.5, 5] : value === 0.125 ? [2.5, 4.6, 6.7] : [];
  return (
    <svg viewBox="0 0 16 16" width={size} height={size} aria-hidden="true" focusable="false"
      fill="none" stroke="currentColor" strokeWidth={1.4} strokeLinecap="round">
      {value === 4 ? <ellipse cx="8" cy="9" rx="3.3" ry="2.3" transform="rotate(-18 8 9)" strokeWidth={1.6} /> : (
        <>{head(value <= 1)}{stem}{flags.map(flag)}</>
      )}
    </svg>
  );
}

export function DotGlyph({ size = 18 }: { size?: number }) {
  return (
    <svg viewBox="0 0 16 16" width={size} height={size} aria-hidden="true" focusable="false">
      <ellipse cx="5.5" cy="10" rx="2.4" ry="1.7" transform="rotate(-22 5.5 10)" fill="currentColor" />
      <circle cx="11.5" cy="9.5" r="1.3" fill="currentColor" />
    </svg>
  );
}

export function TripletGlyph({ size = 18 }: { size?: number }) {
  return (
    <svg viewBox="0 0 16 16" width={size} height={size} aria-hidden="true" focusable="false"
      fill="none" stroke="currentColor" strokeWidth={1.3} strokeLinecap="round" strokeLinejoin="round">
      <path d="M2 6.5V5h4M10 5h4v1.5" />
      <path d="M7 3.5h1.6a.9.9 0 0 1 0 1.8H7.6h1a.9.9 0 0 1 0 1.8H7" />
      <path d="M3 14V9M8 14V9M13 14V9M3 9h10" />
    </svg>
  );
}

export function familyIcon(family: string, p: P = {}) {
  const f = (I as Record<string, ((p: P) => ReactNode) | undefined>)[family];
  return f && ['guitar', 'bass', 'drums', 'piano', 'vocals'].includes(family) ? f(p) : I.note(p);
}

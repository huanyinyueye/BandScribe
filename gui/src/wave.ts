import type { Waveform } from './api';

// Min/max envelope centred in the box, scaled so the loudest sample fills it.
export function drawWave(g: CanvasRenderingContext2D, wave: Waveform, x: number, y: number, w: number, h: number, color: string) {
  const n = wave.min.length;
  if (!n || w <= 0 || h <= 0) return;
  const mid = y + h / 2, scale = h / 2 / Math.max(wave.peak, 1e-6);
  const cols = Math.max(1, Math.floor(w));
  g.fillStyle = color;
  for (let c = 0; c < cols; c++) {
    const a = Math.floor((c / cols) * n), b = Math.max(a + 1, Math.floor(((c + 1) / cols) * n));
    let lo = Infinity, hi = -Infinity;
    for (let i = a; i < b && i < n; i++) { if (wave.min[i] < lo) lo = wave.min[i]; if (wave.max[i] > hi) hi = wave.max[i]; }
    if (!Number.isFinite(lo)) continue;
    const top = mid - hi * scale, bottom = mid - lo * scale;
    g.fillRect(x + (c / cols) * w, top, Math.max(1, w / cols), Math.max(1, bottom - top));
  }
}

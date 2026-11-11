"""Self-contained HTML QA report (spec 4.1 step 9, 5.9).

One file, no external assets: summary, per-track table, per-rule counts, models with
version / hash, and a low-confidence timeline per track (one cell per bar, shaded by the
share of low-confidence notes in that bar).
"""

from __future__ import annotations

import html
from collections import defaultdict
from pathlib import Path
from typing import Any

_CSS = """
:root{--bg:#fff;--fg:#1d1d1f;--mut:#6e6e73;--line:#d2d2d7;--ok:#1f8a4c;--warn:#b26b00;
--bad:#c0392b;--cell:#e8e8ed}
@media (prefers-color-scheme:dark){:root{--bg:#161618;--fg:#f5f5f7;--mut:#a1a1a6;
--line:#3a3a3c;--ok:#34c759;--warn:#ff9f0a;--bad:#ff453a;--cell:#2c2c2e}}
body{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;margin:0;
padding:24px 16px;max-width:1100px;margin-inline:auto}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 8px}
.mut{color:var(--mut)}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}
th{font-weight:600;color:var(--mut)}.pass{color:var(--ok)}.warn{color:var(--warn)}
.fail{color:var(--bad)}.tl{display:flex;gap:1px;flex-wrap:wrap;margin:4px 0 12px}
.tl span{width:8px;height:18px;background:var(--cell);display:inline-block}
.scroll{overflow-x:auto}
"""


def _esc(x: Any) -> str:
    return html.escape(str(x))


def _timeline(track, n_bars: int, out_dir_name: str) -> str:
    total = defaultdict(int)
    low = defaultdict(int)
    for n in track.notes:
        total[n.bar] += 1
        low[n.bar] += int(n.low_conf)
    cells = []
    for b in range(n_bars):
        if total[b] == 0:
            cells.append(f'<span title="bar {b + 1}: no notes"></span>')
            continue
        r = low[b] / total[b]
        # green (all confident) -> red (all low-confidence)
        hue = int(120 * (1 - r))
        cells.append(f'<span style="background:hsl({hue} 65% 45%)" title="bar {b + 1}: '
                     f'{low[b]}/{total[b]} low-confidence"></span>')
    return '<div class="tl">' + "".join(cells) + "</div>"


def write_report(path: Path, score, result: dict[str, Any], context: dict[str, Any]) -> Path:
    manifest = context.get("manifest") or {}
    status = result["status"]
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
             f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>BandScribe QA</title><style>{_CSS}</style></head><body>",
             f"<h1>QA report: {_esc(score.title)}</h1>",
             f"<p>Status <b class='{status}'>{status.upper()}</b> · "
             f"{sum(t['notes'] for t in result['tracks'].values())} notes · "
             f"{result['removed']} removed by QA</p>"]

    parts.append("<h2>Tracks</h2><div class='scroll'><table><tr><th>Track</th><th>Kind</th>"
                 "<th>Notes</th><th>Low confidence</th><th>Removed</th><th>Flags</th></tr>")
    for t in result["tracks"].values():
        flags = ", ".join(f"{k} {v}" for k, v in sorted(t["flags"].items())) or "–"
        parts.append(f"<tr><td>{_esc(t['name'])}</td><td>{_esc(t['kind'])}</td>"
                     f"<td>{t['notes']}</td><td>{t['low_conf']} ({t['low_conf_ratio']:.0%})</td>"
                     f"<td>{t['removed_by_qa']}</td><td class='mut'>{_esc(flags)}</td></tr>")
    parts.append("</table></div>")

    parts.append("<h2>Low-confidence timeline</h2><p class='mut'>One cell per bar; green = "
                 "confident, red = all notes low-confidence, grey = no notes.</p>")
    for t in score.tracks:
        parts.append(f"<div>{_esc(t.name)}</div>{_timeline(t, len(score.bars), '')}")

    parts.append("<h2>Rules</h2><table><tr><th>Rule</th><th>Errors</th><th>Warnings</th></tr>")
    for rid, r in result["rules"].items():
        cls = "fail" if r["errors"] else "warn" if r["warnings"] else "pass"
        parts.append(f"<tr><td>{rid}</td><td class='{cls}'>{r['errors']}</td>"
                     f"<td>{r['warnings']}</td></tr>")
    parts.append("</table>")

    if result["stems"]:
        parts.append("<h2>Stems</h2><table><tr><th>Stem</th><th>Present</th><th>RMS dB</th>"
                     "<th>vs mix dB</th><th>Active</th></tr>")
        for name, s in result["stems"].items():
            parts.append(f"<tr><td>{_esc(name)}</td><td>{'yes' if s['present'] else 'no'}</td>"
                         f"<td>{s['rms_db']}</td><td>{s['relative_db']}</td>"
                         f"<td>{s['active_fraction']:.0%}</td></tr>")
        parts.append("</table>")

    models = manifest.get("models") or {}
    if models:
        parts.append("<h2>Models</h2><div class='scroll'><table><tr><th>Model</th>"
                     "<th>Version</th><th>Weights</th><th>License</th></tr>")
        for mid, m in models.items():
            h = m.get("sha256") or ", ".join(f"{k}: {v[:12]}…" for k, v in
                                             (m.get("files_sha256") or {}).items())
            parts.append(f"<tr><td>{_esc(m.get('name', mid))}</td><td>{_esc(m.get('version', ''))}"
                         f"</td><td class='mut'>{_esc(h or 'missing')}</td>"
                         f"<td>{_esc(m.get('license_weights', ''))}</td></tr>")
        parts.append("</table></div>")

    if result["issues"]:
        parts.append("<h2>Issues</h2><div class='scroll'><table><tr><th>Rule</th><th>Severity"
                     "</th><th>Track</th><th>Bar</th><th>Time (s)</th><th>Pitch</th>"
                     "<th>Message</th></tr>")
        for i in result["issues"][:300]:
            bar = i.get("bar")
            parts.append(f"<tr><td>{i['rule']}</td><td class='{'fail' if i['severity'] == 'error' else 'warn'}'>"
                         f"{i['severity']}</td><td>{_esc(i.get('track', ''))}</td>"
                         f"<td>{'' if bar is None else bar + 1}</td><td>{i.get('onset', '')}</td>"
                         f"<td>{i.get('pitch', '')}</td><td>{_esc(i['message'])}</td></tr>")
        parts.append("</table></div>")
        if len(result["issues"]) > 300 or result.get("issues_truncated"):
            parts.append("<p class='mut'>More issues in qa.json.</p>")

    parts.append("</body></html>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(parts), encoding="utf-8")
    return path

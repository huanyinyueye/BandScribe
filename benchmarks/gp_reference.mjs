// Dump the played notes of a Guitar Pro file (repeats expanded) as JSON, for scoring a
// transcription against a reference score.
//
// Usage: node benchmarks/gp_reference.mjs <file.gp> <out.json>
// Output: {tracks: [{index, name, program, isPercussion, tuning, notes: [{tick, length, key,
//          velocity, channel}]}], tempos: [{tick, bpm}], ppq, syncPoints, framePadding}
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const lib = path.join(here, '..', 'exporter', 'node_modules', '@coderline', 'alphatab', 'dist',
  'alphaTab.mjs');
const alphaTab = await import(pathToFileURL(lib).href);

const [src, out] = process.argv.slice(2);
const settings = new alphaTab.Settings();
const score = alphaTab.importer.ScoreLoader.loadScoreFromBytes(new Uint8Array(fs.readFileSync(src)),
  settings);

const notes = [];
const tempos = [];
const timeSignatures = [];
const known = {
  addTempo(tick, bpm) { tempos.push({ tick, bpm }); },
  addTimeSignature(tick, num, den) { timeSignatures.push({ tick, num, den }); },
  addNote(track, start, length, key, velocity, channel) {
    notes.push({ track, tick: start, length, key, velocity, channel });
  },
};
// Every other handler callback (metronome, bends, controllers, ...) is ignored.
const handler = new Proxy(known, { get: (o, k) => o[k] ?? (() => {}) });
const gen = new alphaTab.midi.MidiFileGenerator(score, settings, handler);
gen.generate();

const tracks = score.tracks.map((t, i) => {
  const st = t.staves[0];
  return {
    index: i, name: t.name, shortName: t.shortName, program: t.playbackInfo.program,
    primaryChannel: t.playbackInfo.primaryChannel,
    isPercussion: st.isPercussion, tuning: st.tuning ? Array.from(st.tuning) : [],
    notes: notes.filter((n) => n.track === i).map(({ track, ...n }) => n),
  };
});
const syncPoints = [];
for (const mb of score.masterBars) {
  for (const a of mb.syncPoints ?? []) {
    syncPoints.push({ bar: mb.index, ratio: a.ratioPosition, ...a.syncPointValue });
  }
}
fs.writeFileSync(out, JSON.stringify({
  ppq: 960, tempos, timeSignatures, syncPoints,
  backingTrack: score.backingTrack ? { padding: score.backingTrack.padding } : null,
  bars: score.masterBars.length, tracks,
}, null, 0));
console.log(JSON.stringify(tracks.map((t) => [t.index, t.name, t.isPercussion, t.notes.length])));
console.log(JSON.stringify({ tempos: tempos.slice(0, 5), syncPoints, ppq: 960 }));

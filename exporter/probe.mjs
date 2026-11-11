import * as alphaTab from '@coderline/alphatab';
const m = alphaTab.model;
const settings = new alphaTab.Settings();
const score = new m.Score();
score.title = 'Probe';
const sigs = [[4,4,120,'Intro'],[7,8,150,null]];
for (const [n,d,bpm,sec] of sigs) {
  const mb = new m.MasterBar(); mb.timeSignatureNumerator = n; mb.timeSignatureDenominator = d;
  const a = new m.Automation(); a.type = m.AutomationType.Tempo; a.value = bpm; a.ratioPosition = 0; a.isLinear = false; mb.tempoAutomations.push(a);
  if (sec) { mb.section = new m.Section(); mb.section.text = sec; mb.section.marker = ''; }
  score.addMasterBar(mb);
}
function addTrack(name, program, tuning, perc, tab) {
  const t = new m.Track(); t.name = name; t.shortName = name.slice(0,4);
  t.playbackInfo.program = program; t.playbackInfo.primaryChannel = perc ? 9 : score.tracks.length*2; t.playbackInfo.secondaryChannel = perc ? 9 : score.tracks.length*2+1;
  t.ensureStaveCount(1); const s = t.staves[0];
  if (perc) { s.isPercussion = true; s.showTablature = false; s.showStandardNotation = true; }
  else if (tuning) { s.stringTuning = new m.Tuning('', tuning, false); s.showTablature = tab; s.showStandardNotation = true; }
  else { s.stringTuning = new m.Tuning('', [], false); s.showTablature = false; s.showStandardNotation = true; }
  score.addTrack(t); return t;
}
function addBar(track, beats) {
  const s = track.staves[0]; const bar = new m.Bar(); s.addBar(bar);
  if (s.isPercussion) bar.clef = m.Clef.Neutral;
  const v = new m.Voice(); bar.addVoice(v);
  for (const b of beats) {
    const beat = new m.Beat(); beat.duration = b.dur; beat.dots = b.dots ?? 0;
    if (b.tu) { beat.tupletNumerator = b.tu[0]; beat.tupletDenominator = b.tu[1]; }
    if (b.text) beat.text = b.text;
    v.addBeat(beat);
    for (const n of (b.notes ?? [])) {
      const note = new m.Note();
      if (s.isPercussion) { note.percussionArticulation = n.perc; }
      else if (s.stringTuning.tunings.length) { note.fret = n.fret; note.string = s.stringTuning.tunings.length - n.string + 1; }
      else { note.octave = Math.floor(n.pitch/12); note.tone = n.pitch % 12; }
      if (n.tie) note.isTieDestination = true;
      beat.addNote(note);
    }
  }
}
const D = m.Duration;
const g = addTrack('Guitar', 30, [64,59,55,50,45,38], false, true);
addBar(g, [{dur:D.Eighth, notes:[{fret:0,string:6}]},{dur:D.Eighth, notes:[{fret:3,string:6}], text:'?low-conf'},{dur:D.Quarter, notes:[{fret:5,string:5},{fret:7,string:4}]},{dur:D.Eighth},{dur:D.Quarter, notes:[{fret:5,string:5,tie:true},{fret:7,string:4,tie:true}]},{dur:D.Sixteenth,tu:[3,2],notes:[{fret:5,string:3}]},{dur:D.Sixteenth,tu:[3,2],notes:[{fret:7,string:3}]},{dur:D.Sixteenth,tu:[3,2],notes:[{fret:8,string:3}]}]);
addBar(g, Array.from({length:7},()=>({dur:D.Eighth, notes:[{fret:0,string:6}]})));
const dr = addTrack('Drums', 0, null, true, false);
addBar(dr, [{dur:D.Quarter, notes:[{perc:36},{perc:42}]},{dur:D.Quarter, notes:[{perc:38}]},{dur:D.Quarter, notes:[{perc:46}]},{dur:D.Quarter, notes:[{perc:49},{perc:51}]}]);
addBar(dr, Array.from({length:7},()=>({dur:D.Eighth, notes:[{perc:36}]})));
const p = addTrack('Piano', 0, null, false, false);
addBar(p, [{dur:D.Half, notes:[{pitch:60},{pitch:64}]},{dur:D.Half}]);
addBar(p, Array.from({length:7},()=>({dur:D.Eighth, notes:[{pitch:72}]})));
score.finish(settings);
const data = new alphaTab.exporter.Gp7Exporter().export(score, settings);
console.log('gp bytes', data.length);
const back = alphaTab.importer.ScoreLoader.loadScoreFromBytes(data, settings);
console.log('masterBars', back.masterBars.map(x => [x.timeSignatureNumerator, x.timeSignatureDenominator, x.section?.text, x.tempoAutomations.map(a=>a.value)]));
for (const t of back.tracks) {
  const s = t.staves[0]; const out = [];
  for (const bar of s.bars) { for (const b of bar.voices[0].beats) out.push(`${b.duration}${b.dots?'.':''}${b.tupletNumerator>1?'t'+b.tupletNumerator:''}${b.text?'"'+b.text+'"':''}[` + b.notes.map(n => s.isPercussion ? 'p'+n.percussionArticulation : `${n.realValue}${n.isStringed?'('+n.fret+'@'+n.string+')':''}${n.isTieDestination?'~':''}`).join(',') + ']'); out.push('|'); }
  console.log(t.name, 'tab', s.showTablature, 'std', s.showStandardNotation, 'perc', s.isPercussion, 'tuning', s.stringTuning?.tunings, out.join(' '));
}
import fs from 'node:fs'; fs.writeFileSync(process.env.TEMP + '/probe.gp', data);

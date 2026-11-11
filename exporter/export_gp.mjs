// Build an alphaTab score from BandScribe's layout JSON and write a Guitar Pro 7/8 `.gp` file.
//
// Usage: node export_gp.mjs <input.json> <output.gp>
//        node export_gp.mjs --read <file.gp>          (summary of an existing file)
// Prints a JSON summary of the file as re-read by alphaTab, so the caller can verify that
// bars, time signatures, notes, strings and frets survived the export.
import fs from 'node:fs';
import * as alphaTab from '@coderline/alphatab';

const m = alphaTab.model;
const settings = new alphaTab.Settings();

const DURATIONS = {
  1: m.Duration.Whole, 2: m.Duration.Half, 4: m.Duration.Quarter, 8: m.Duration.Eighth,
  16: m.Duration.Sixteenth, 32: m.Duration.ThirtySecond, 64: m.Duration.SixtyFourth,
};

function buildScore(input) {
  const score = new m.Score();
  score.title = input.title ?? '';
  score.artist = input.artist ?? '';
  score.tab = 'BandScribe (automatic transcription)';
  score.notices = input.notice ?? '';
  let lastBpm = null;
  for (const b of input.bars) {
    const mb = new m.MasterBar();
    mb.timeSignatureNumerator = b.num;
    mb.timeSignatureDenominator = b.den;
    const bpm = Math.round(b.bpm);
    if (lastBpm === null || Math.abs(bpm - lastBpm) >= 1) {
      const a = new m.Automation();
      a.type = m.AutomationType.Tempo;
      a.value = bpm;
      a.ratioPosition = 0;
      a.isLinear = false;
      mb.tempoAutomations.push(a);
      lastBpm = bpm;
    }
    if (b.section) {
      mb.section = new m.Section();
      mb.section.text = b.section;
      mb.section.marker = '';
    }
    score.addMasterBar(mb);
  }

  let channel = 0;
  for (const t of input.tracks) {
    const track = new m.Track();
    track.name = t.name;
    track.shortName = t.short ?? t.name.slice(0, 6);
    const perc = t.kind === 'drums';
    track.playbackInfo.program = perc ? 0 : t.program;
    if (perc) {
      track.playbackInfo.primaryChannel = 9;
      track.playbackInfo.secondaryChannel = 9;
    } else {
      if (channel === 9) channel = 10;
      track.playbackInfo.primaryChannel = channel % 16;
      track.playbackInfo.secondaryChannel = (channel + 1) % 16;
      channel += 2;
    }
    track.ensureStaveCount(1);
    const staff = track.staves[0];
    const stringed = t.kind === 'tab';
    if (perc) {
      staff.isPercussion = true;
      staff.showTablature = false;
      staff.showStandardNotation = true;
    } else if (stringed) {
      staff.stringTuning = new m.Tuning(t.tuning_name ?? '', t.tuning, false);
      staff.showTablature = true;
      staff.showStandardNotation = true;
    } else {
      staff.stringTuning = new m.Tuning('', [], false);
      staff.showTablature = false;
      staff.showStandardNotation = true;
    }
    score.addTrack(track);

    const nStrings = stringed ? t.tuning.length : 0;
    for (const beats of t.bars) {
      const bar = new m.Bar();
      staff.addBar(bar);
      if (perc) bar.clef = m.Clef.Neutral;
      else if (t.clef === 'F4') bar.clef = m.Clef.F4;
      if (!perc && input.key_fifths !== undefined && input.key_fifths !== null) {
        bar.keySignature = input.key_fifths;
        bar.keySignatureType = input.key_minor ? m.KeySignatureType.Minor : m.KeySignatureType.Major;
      }
      const voice = new m.Voice();
      bar.addVoice(voice);
      for (const b of beats) {
        const beat = new m.Beat();
        beat.duration = DURATIONS[b.d];
        beat.dots = b.dots ?? 0;
        if (b.tu) {
          beat.tupletNumerator = b.tu[0];
          beat.tupletDenominator = b.tu[1];
        }
        if (b.text) beat.text = b.text;
        voice.addBeat(beat);
        if (b.rest) continue;
        for (const n of b.notes) {
          const note = new m.Note();
          if (perc) {
            note.percussionArticulation = n.p;
          } else if (stringed) {
            note.fret = n.f;
            note.string = nStrings - n.s + 1; // alphaTab counts strings from the lowest
          } else {
            note.octave = Math.floor(n.p / 12);
            note.tone = n.p % 12;
          }
          if (b.tie) note.isTieDestination = true;
          beat.addNote(note);
        }
      }
    }
  }
  score.finish(settings);
  return score;
}

function summarize(score) {
  const out = {
    title: score.title,
    masterBars: score.masterBars.map((mb) => ({
      num: mb.timeSignatureNumerator, den: mb.timeSignatureDenominator,
      section: mb.section ? mb.section.text : null,
      tempo: mb.tempoAutomations.length ? mb.tempoAutomations[0].value : null,
    })),
    tracks: [],
  };
  for (const t of score.tracks) {
    const staff = t.staves[0];
    const nStrings = staff.stringTuning ? staff.stringTuning.tunings.length : 0;
    const track = {
      name: t.name, percussion: staff.isPercussion, tablature: staff.showTablature,
      tuning: staff.stringTuning ? Array.from(staff.stringTuning.tunings) : [],
      bars: staff.bars.length, beats: 0, rests: 0, notes: [],
    };
    staff.bars.forEach((bar, bi) => {
      const voice = bar.voices[0];
      if (!voice) return;
      for (const beat of voice.beats) {
        track.beats += 1;
        if (beat.notes.length === 0) track.rests += 1;
        for (const n of beat.notes) {
          track.notes.push(staff.isPercussion
            ? [bi, n.percussionArticulation, null, null, n.isTieDestination ? 1 : 0]
            : [bi, n.realValue, n.isStringed ? nStrings - n.string + 1 : null,
              n.isStringed ? n.fret : null, n.isTieDestination ? 1 : 0]);
        }
      }
    });
    out.tracks.push(track);
  }
  return out;
}

const args = process.argv.slice(2);
try {
  if (args[0] === '--read') {
    const data = new Uint8Array(fs.readFileSync(args[1]));
    const score = alphaTab.importer.ScoreLoader.loadScoreFromBytes(data, settings);
    process.stdout.write(JSON.stringify(summarize(score)));
  } else {
    const input = JSON.parse(fs.readFileSync(args[0], 'utf-8'));
    const score = buildScore(input);
    const data = new alphaTab.exporter.Gp7Exporter().export(score, settings);
    fs.writeFileSync(args[1], data);
    const back = alphaTab.importer.ScoreLoader.loadScoreFromBytes(data, settings);
    process.stdout.write(JSON.stringify({ bytes: data.length, alphatab: alphaTab.meta.version,
      ...summarize(back) }));
  }
} catch (e) {
  process.stderr.write(`export_gp failed: ${e && e.stack ? e.stack : e}\n`);
  process.exit(1);
}

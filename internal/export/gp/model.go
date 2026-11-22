// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy
// of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// Translated from alphaTab 1.9.0 (https://github.com/CoderLine/alphaTab), Copyright CoderLine:
// the parts of Score/Track/Staff/Bar/Voice/Beat/Note, Tuning and FingeringAssigner that the
// BandScribe export reaches, plus the score construction of exporter/export_gp.mjs.

package gp

import (
	"fmt"
	"math"
	"strconv"

	"bandscribe/internal/layout"
	"bandscribe/internal/score"
)

type tempoAutomation struct {
	value, ratio float64
	visible      bool
}

type masterBar struct {
	index    int
	num, den int
	tempos   []tempoAutomation
	section  *string
	sentText *string // the payload's section, for verification
}

type articulation struct {
	elementType     string
	id              float64
	staffLine       int
	outputMidi      float64
	elementTypeKind string
}

func (a *articulation) uniqueID() string { return a.elementType + "." + jsString(a.id) }

type gTrack struct {
	index              int
	name, shortName    string
	program            int
	primary, secondary int
	staff              *gStaff
	percArticulations  []*articulation
	sent               *score.Track
	sentBars           [][]layout.Beat
}

type gStaff struct {
	track      *gTrack
	perc       bool
	showTab    bool
	showStd    bool
	tuning     []float64
	tuningName string
	bars       []*gBar
}

type gBar struct {
	id, index int
	staff     *gStaff
	voices    []*gVoice
	clef      string
	ks        float64
	minor     bool
	next      *gBar
}

type gVoice struct {
	id    int
	index int
	bar   *gBar
	beats []*gBeat
}

type gBeat struct {
	id, index         int
	voice             *gVoice
	duration, dots    int
	tupNum, tupDen    int
	text              string
	notes             []*gNote
	prev, next        *gBeat
	strLookup         map[float64]*gNote
	valLookup         map[float64]*gNote
	instrument        bool
	instrumentProgram int
}

type gNote struct {
	id, index      int
	beat           *gBeat
	fret           float64
	fretNull       bool // JS null (score note without fret) until the exporter's JSON clone
	str            float64
	octave, tone   float64
	perc           float64
	tieDest        bool
	tieOrigin      *gNote
	tieDestination *gNote
}

type gScore struct {
	title, artist, tab, notices string
	masterBars                  []*masterBar
	tracks                      []*gTrack
}

func (n *gNote) staff() *gStaff { return n.beat.voice.bar.staff }

func (n *gNote) isStringed() bool { return !isNaN(n.str) && (n.fretNull || !isNaN(n.fret)) }
func (n *gNote) isPiano() bool    { return !n.isStringed() && !isNaN(n.octave) && !isNaN(n.tone) }
func (n *gNote) isPerc() bool     { return !isNaN(n.perc) }

// stringTuning is Note.stringTuning (capo 0): an index outside the tuning reads `undefined`.
func stringTuning(st *gStaff, s float64) float64 {
	if len(st.tuning) > 0 && s >= 0 {
		idx := float64(len(st.tuning)) - (s - 1) - 1
		if idx >= 0 && idx < float64(len(st.tuning)) && idx == math.Trunc(idx) {
			return st.tuning[int(idx)]
		}
		return nan
	}
	return 0
}

// articulationOf is PercussionMapper.getArticulation.
func articulationOf(n *gNote) *articulation {
	idx := n.perc
	if idx < 0 {
		return nil
	}
	list := n.staff().track.percArticulations
	if idx < float64(len(list)) {
		return list[int(idx)]
	}
	return articulationsByID[key(idx)]
}

// realValue is Note.realValue for an untransposed staff without harmonics.
func (n *gNote) realValue() float64 {
	if n.isPerc() {
		if a := articulationOf(n); a != nil {
			return a.outputMidi
		}
		return n.perc
	}
	if n.isStringed() {
		fret := n.fret
		if n.fretNull {
			fret = 0
		}
		return fret + stringTuning(n.staff(), n.str)
	}
	if n.isPiano() {
		return n.octave*12 + n.tone
	}
	return 0
}

func (b *gBeat) addNote(n *gNote) {
	n.beat = b
	n.index = len(b.notes)
	b.notes = append(b.notes, n)
	if n.str >= 0 {
		b.strLookup[key(n.str)] = n
	}
}

var articulationsByID = map[float64]*articulation{}
var defaultArticulations []*articulation

func init() {
	for _, el := range drumKit {
		for _, a := range el.articulations {
			id, err := strconv.Atoi(a.inputMidi)
			if err != nil {
				panic("drum kit articulation without a single input MIDI number: " + a.inputMidi)
			}
			art := &articulation{el.name, float64(id), a.staffLine, float64(a.outputMidi), el.kind}
			defaultArticulations = append(defaultArticulations, art)
			articulationsByID[key(art.id)] = art
		}
	}
}

type drumElement struct {
	name, kind, soundbank string
	articulations         []drumArticulation
}

type drumArticulation struct {
	name                       string
	staffLine                  int
	noteheads, placement, symb string
	inputMidi, rseSound        string
	outputMidi                 int
}

type tuningPreset struct {
	name     string
	tunings  []int
	standard bool
}

var durations = map[int]bool{1: true, 2: true, 4: true, 8: true, 16: true, 32: true, 64: true}

const notice = "Automatic transcription by BandScribe. '?' marks low-confidence notes. " +
	"Non-commercial use only."

type ids struct{ bar, voice, beat, note int }

// build is buildScore of exporter/export_gp.mjs applied to build_payload(score), followed by
// alphaTab's Score.finish and the JSON clone that Gp7Exporter makes before writing.
func build(s *score.Score) (*gScore, error) {
	if len(s.Bars) == 0 {
		return nil, fmt.Errorf("score has no bars")
	}
	if len(s.Tracks) == 0 {
		return nil, fmt.Errorf("score has no tracks")
	}
	g := &gScore{title: s.Title, tab: "BandScribe (automatic transcription)", notices: notice}
	artist, err := jsText(s.Meta["artist"])
	if err != nil {
		return nil, fmt.Errorf("meta.artist: %w", err)
	}
	g.artist = artist
	g.masterBars = tempoMap(s.Bars)

	var fifths *float64
	minor := false
	if s.Key != nil {
		switch v := s.Key["fifths"].(type) {
		case nil:
		case float64:
			fifths = &v
		default:
			return nil, fmt.Errorf("key fifths %v is not a number", v)
		}
		minor = s.Key["mode"] == "minor"
	}

	var id ids
	channel := 0
	for ti := range s.Tracks {
		t := &s.Tracks[ti]
		tr := &gTrack{index: ti, name: t.Name, shortName: utf16Slice(t.Name, 6), sent: t}
		perc := t.Kind == "drums"
		stringed := t.Kind == "tab"
		tr.program = t.MidiProgram
		if perc {
			tr.program = 0
			tr.primary, tr.secondary = 9, 9
		} else {
			if channel == 9 {
				channel = 10
			}
			tr.primary, tr.secondary = channel%16, (channel+1)%16
			channel += 2
		}
		st := &gStaff{track: tr, showStd: true}
		tr.staff = st
		switch {
		case perc:
			st.perc = true
		case stringed:
			for _, p := range t.Tuning {
				st.tuning = append(st.tuning, float64(p))
			}
			st.tuningName = t.TuningName()
			st.showTab = true
		}
		g.tracks = append(g.tracks, tr)

		nStrings := 0
		if stringed {
			nStrings = len(t.Tuning)
		}
		tr.sentBars = layout.Track(t, s.Bars)
		for bi, beats := range tr.sentBars {
			bar := &gBar{id: id.bar, index: bi, staff: st, clef: "G2"}
			id.bar++
			if bi > 0 {
				st.bars[bi-1].next = bar
			}
			st.bars = append(st.bars, bar)
			if perc {
				bar.clef = "Neutral"
			} else if t.Family == "bass" && t.Kind != "tab" {
				bar.clef = "F4"
			}
			if !perc && fifths != nil {
				bar.ks = *fifths
				bar.minor = minor
			}
			v := &gVoice{id: id.voice, bar: bar}
			id.voice++
			bar.voices = []*gVoice{v}
			for _, b := range beats {
				if !durations[b.Duration] {
					return nil, fmt.Errorf("track %s bar %d: note value 1/%d has no Guitar Pro duration", t.ID, bi, b.Duration)
				}
				beat := &gBeat{id: id.beat, index: len(v.beats), voice: v, duration: b.Duration,
					dots: b.Dots, tupNum: -1, tupDen: -1, text: b.Text,
					strLookup: map[float64]*gNote{}, valLookup: map[float64]*gNote{}}
				id.beat++
				if b.Tuplet != nil {
					beat.tupNum, beat.tupDen = b.Tuplet[0], b.Tuplet[1]
				}
				v.beats = append(v.beats, beat)
				if b.Rest {
					continue
				}
				for _, sn := range b.Notes {
					n := &gNote{id: id.note, fret: nan, str: nan, octave: nan, tone: nan, perc: nan}
					id.note++
					switch {
					case perc:
						n.perc = float64(sn.Pitch)
					case stringed:
						if sn.Fret == nil {
							n.fretNull = true
						} else {
							n.fret = float64(*sn.Fret)
						}
						s := 0
						if sn.String != nil {
							s = *sn.String
						}
						n.str = float64(nStrings - s + 1)
					default:
						p := float64(sn.Pitch)
						n.octave = math.Floor(p / 12)
						n.tone = math.Mod(p, 12)
					}
					if b.Tied {
						n.tieDest = true
					}
					beat.addNote(n)
				}
			}
		}
	}
	if err := g.finish(); err != nil {
		return nil, err
	}
	for _, tr := range g.tracks {
		for _, bar := range tr.staff.bars {
			for _, v := range bar.voices {
				for _, b := range v.beats {
					for _, n := range b.notes {
						if n.fretNull { // JSON clone: null is read back as NaN
							n.fret, n.fretNull = nan, false
						}
					}
				}
			}
		}
	}
	return g, nil
}

// jsText is `value ?? ”` for a JSON value that becomes CDATA text.
func jsText(v any) (string, error) {
	switch x := v.(type) {
	case nil:
		return "", nil
	case string:
		return x, nil
	case float64:
		return jsString(x), nil
	case bool:
		return strconv.FormatBool(x), nil
	}
	return "", fmt.Errorf("unsupported value %v", v)
}

// tempoMap is the tempo automation loop of export_gp.mjs: whole BPM values with the rounding
// error carried forward, beat by beat when the bar has per-beat times.
func tempoMap(bars []score.Bar) []*masterBar {
	var out []*masterBar
	var lastBpm, lastVisible *float64
	target, written := 0.0, 0.0
	for i, b := range bars {
		mb := &masterBar{index: i, num: b.Numerator, den: b.Denominator, sentText: b.Section}
		quarters := float64(4*b.Numerator) / float64(b.Denominator)
		var segments []float64
		if len(b.BeatTimes) == b.Numerator+1 {
			for k := 1; k < len(b.BeatTimes); k++ {
				segments = append(segments, b.BeatTimes[k]-b.BeatTimes[k-1])
			}
		} else {
			segments = []float64{float64(60*quarters) / b.BPM}
		}
		segQ := quarters / float64(len(segments))
		for k, duration := range segments {
			target += duration
			remaining := target - written
			d := duration
			if remaining > 0 {
				d = remaining
			}
			bpm := jsMax(1, jsRound(float64(60*segQ)/d))
			written += float64(60*segQ) / bpm
			visible := k == 0 && (lastVisible == nil || math.Abs(b.BPM-*lastVisible) >= 1)
			if !visible && lastBpm != nil && math.Abs(bpm-*lastBpm) < 1 {
				continue
			}
			mb.tempos = append(mb.tempos, tempoAutomation{bpm, float64(k) / float64(len(segments)), visible})
			if visible {
				v := b.BPM
				lastVisible = &v
			}
			lb := bpm
			lastBpm = &lb
		}
		if b.Section != nil && *b.Section != "" {
			sec := *b.Section
			mb.section = &sec
		}
		out = append(out, mb)
	}
	return out
}

// finish is Score.finish for the parts that change what Gp7Exporter writes.
func (g *gScore) finish() error {
	for _, tr := range g.tracks {
		st := tr.staff
		st.finishTuning()
		if st.perc {
			st.tuning = []float64{0, 0, 0, 0, 0, 0}
		}
		if len(st.tuning) == 0 {
			st.showTab = false
		}
		for _, bar := range st.bars {
			for _, v := range bar.voices {
				if err := v.finish(); err != nil {
					return err
				}
			}
		}
		if tr.shortName == "" {
			tr.shortName = utf16Slice(tr.name, 10)
		}
	}
	return nil
}

// finishTuning is Tuning.finish: name an unnamed tuning after the first matching preset.
func (st *gStaff) finishTuning() {
	for _, preset := range tuningPresets[len(st.tuning)] {
		equal := true
		for i, p := range st.tuning {
			if p != float64(preset.tunings[i]) {
				equal = false
				break
			}
		}
		if equal {
			if st.tuningName == "" {
				st.tuningName = preset.name
			}
			return
		}
	}
}

// finish is Voice.finish: chain the beats (and their ties), then Beat.finish.
func (v *gVoice) finish() error {
	for i, b := range v.beats {
		b.index = i
		if i < len(v.beats)-1 {
			b.next = v.beats[i+1]
			b.next.prev = b
		} else if v.bar.next != nil {
			nextVoice := v.bar.next.voices[v.index]
			if len(nextVoice.beats) == 0 {
				return fmt.Errorf("TypeError: Cannot set properties of null (setting 'previousBeat') in bar %d", v.bar.index)
			}
			b.next = nextVoice.beats[0]
			b.next.prev = b
		}
		for _, n := range b.notes {
			b.valLookup[key(n.realValue())] = n
			n.chain()
		}
	}
	for _, b := range v.beats {
		if !b.instrument && b.index == 0 && v.index == 0 && v.bar.index == 0 {
			b.instrument, b.instrumentProgram = true, v.bar.staff.track.program
		}
		for _, n := range b.notes {
			if n.tieDest {
				n.chain()
			}
		}
	}
	return nil
}

// chain is Note.chain for notes without serialized ids: find the tie origin.
func (n *gNote) chain() {
	if !n.tieDest && n.tieOrigin == nil {
		return
	}
	origin := n.tieOrigin
	if origin == nil {
		origin = findTieOrigin(n)
	}
	if origin == nil {
		n.tieDest = false
		return
	}
	origin.tieDestination = n
	n.tieOrigin = origin
	n.fret, n.fretNull = origin.fret, origin.fretNull
	n.octave, n.tone = origin.octave, origin.tone
}

func findTieOrigin(n *gNote) *gNote {
	barIndex := n.beat.voice.bar.index
	for prev := n.beat.prev; prev != nil && prev.voice.bar.index >= barIndex-3; prev = prev.prev {
		if n.isStringed() {
			if o := prev.strLookup[key(n.str)]; o != nil {
				return o
			}
		} else if isNaN(n.octave) && isNaN(n.tone) {
			if n.index < len(prev.notes) {
				return prev.notes[n.index]
			}
		} else if o := prev.valLookup[key(n.realValue())]; o != nil {
			return o
		}
	}
	return nil
}

// staffNotesAreNotStringed is ModelUtils.staffNotesAreNotStringed: decided by the first note.
func staffNotesAreNotStringed(st *gStaff) bool {
	for _, bar := range st.bars {
		for _, v := range bar.voices {
			for _, b := range v.beats {
				for _, n := range b.notes {
					return !n.isStringed()
				}
			}
		}
	}
	return false
}

// fingering is FingeringAssigner with its default options (capo 0, no transposition).
type fingering struct {
	tuning     []float64
	hand       float64
	lastString [128]int
}

func newFingering(tuning []float64) (*fingering, error) {
	if len(tuning) < 1 || len(tuning) > 30 {
		return nil, fmt.Errorf("FingeringAssigner requires 1..30 strings, got tuning.length=%d", len(tuning))
	}
	return &fingering{tuning: tuning, hand: 5}, nil
}

func (f *fingering) assign(b *gBeat) {
	var sorted []int
	for i, n := range b.notes {
		if n.isStringed() {
			continue
		}
		if n.isPerc() {
			if a := articulationOf(n); a != nil {
				if isNaN(n.str) {
					n.str = float64(max(1, min(6, 7-a.staffLine)))
					b.strLookup[key(n.str)] = n
				}
				if isNaN(n.fret) {
					n.fret = a.outputMidi
				}
			}
			continue
		}
		if o := n.tieOrigin; n.tieDest && o != nil && o.isStringed() {
			n.str, n.fret = o.str, o.fret
			b.strLookup[key(n.str)] = n
			continue
		}
		value := n.realValue()
		j := len(sorted)
		sorted = append(sorted, i)
		for j > 0 && b.notes[sorted[j-1]].realValue() > value {
			sorted[j] = sorted[j-1]
			j--
		}
		sorted[j] = i
	}
	if len(sorted) == 0 {
		return
	}
	strings := len(f.tuning)
	used := 0
	newHand := -1.0
	for _, i := range sorted {
		n := b.notes[i]
		realValue := n.realValue()
		target := realValue
		continuity := 0
		if realValue >= 0 && realValue < 128 {
			continuity = f.lastString[int(realValue)]
		}
		bestString, bestFret := 1, 0.0
		if n.tieDest && continuity > 0 && continuity <= strings && used&(1<<(continuity-1)) == 0 {
			bestString = continuity
			bestFret = target - f.tuning[strings-continuity]
		} else {
			bestCost := math.Inf(1)
			for s := 1; s <= strings; s++ {
				fret := target - f.tuning[strings-s]
				cost := math.Abs(fret - f.hand)
				if fret == 0 {
					cost += -1
				} else {
					cost += 0
				}
				if fret < 0 {
					cost += float64(-fret * 3)
				} else {
					cost += 0
				}
				if fret > 12 {
					cost += float64((fret - 12) * 0.5)
				} else {
					cost += 0
				}
				if used&(1<<(s-1)) != 0 {
					cost += 100
				} else {
					cost += 0
				}
				if (s > 1 && used&(1<<(s-2)) != 0) || (s < strings && used&(1<<s) != 0) {
					cost += -1.5
				} else {
					cost += 0
				}
				if s == continuity {
					cost += -0.75
				} else {
					cost += 0
				}
				if cost < bestCost {
					bestCost, bestString, bestFret = cost, s, fret
				}
			}
		}
		n.str, n.fret = float64(bestString), bestFret
		b.strLookup[key(n.str)] = n
		used |= 1 << (bestString - 1)
		if realValue >= 0 && realValue < 128 {
			f.lastString[int(realValue)] = bestString
		}
		if bestFret > 0 && (newHand < 0 || bestFret < newHand) {
			newHand = bestFret
		}
	}
	if newHand >= 0 {
		alpha := 0.7 // a variable: JS evaluates 1-0.7 at run time (0.30000000000000004)
		f.hand = float64(alpha*f.hand) + float64((1-alpha)*newHand)
	}
}

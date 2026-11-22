// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy
// of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// Translated from alphaTab 1.9.0 (https://github.com/CoderLine/alphaTab), Copyright CoderLine:
// GpifWriter, PartConfiguration.writeForScore, LayoutConfiguration.writeForScore and the
// ModelUtils.resolveSpelling tables, for the model features the BandScribe export uses.

package gp

import (
	"encoding/binary"
	"fmt"
	"math"
	"strconv"
	"strings"
)

type spelling struct{ degree, offset int }

var spellingCandidates = [12][]spelling{
	{{0, 0}, {1, -2}, {6, 1}}, {{0, 1}, {1, -1}, {6, 2}}, {{1, 0}, {0, 2}, {2, -2}},
	{{1, 1}, {2, -1}, {3, -2}}, {{2, 0}, {1, 2}, {3, -1}}, {{3, 0}, {2, 1}, {4, -2}},
	{{3, 1}, {4, -1}, {2, 2}}, {{4, 0}, {3, 2}, {5, -2}}, {{4, 1}, {5, -1}},
	{{5, 0}, {4, 2}, {6, -2}}, {{5, 1}, {6, -1}, {0, -2}}, {{6, 0}, {5, 2}, {0, -1}},
}

var sharpPreferred = [12]spelling{{0, 0}, {0, 1}, {1, 0}, {1, 1}, {2, 0}, {3, 0}, {3, 1}, {4, 0},
	{4, 1}, {5, 0}, {5, 1}, {6, 0}}
var flatPreferred = [12]spelling{{0, 0}, {1, -1}, {1, 0}, {2, -1}, {2, 0}, {3, 0}, {4, -1}, {4, 0},
	{5, -1}, {5, 0}, {6, -1}, {6, 0}}
var degreeSemitones = [7]int{0, 2, 4, 5, 7, 9, 11}
var sharpOrder = [7]int{3, 0, 4, 1, 5, 2, 6}
var flatOrder = [7]int{6, 2, 5, 1, 4, 0, 3}

var steps = [7]string{"C", "D", "E", "F", "G", "A", "B"}
var accidentals = [5]string{"bb", "b", "", "#", "x"}

// ksOffset is ModelUtils.getKeySignatureAccidentalOffset; JS throws for a key signature outside
// -7..7 (the lookup row is undefined).
func ksOffset(ks float64, degree int) (int, error) {
	if ks != math.Trunc(ks) || ks < -7 || ks > 7 {
		return 0, fmt.Errorf("TypeError: Cannot read properties of undefined (key signature %s)", jsString(ks))
	}
	k := int(ks)
	for i := 0; i < k; i++ {
		if sharpOrder[i] == degree {
			return 1, nil
		}
	}
	for i := 0; i < -k; i++ {
		if flatOrder[i] == degree {
			return -1, nil
		}
	}
	return 0, nil
}

// resolveSpelling is ModelUtils.resolveSpelling with NoteAccidentalMode.Default.
func resolveSpelling(ks, value float64) (spelling, float64, error) {
	chroma := value - 12*math.Floor(value/12)
	if isNaN(chroma) || chroma != math.Trunc(chroma) {
		return spelling{}, 0, fmt.Errorf("TypeError: Cannot read properties of undefined (reading 'find') for note value %s", jsString(value))
	}
	var sp *spelling
	for i, c := range spellingCandidates[int(chroma)] {
		off, err := ksOffset(ks, c.degree)
		if err != nil {
			return spelling{}, 0, err
		}
		if off == c.offset {
			sp = &spellingCandidates[int(chroma)][i]
			break
		}
	}
	if sp == nil {
		if ks < 0 {
			sp = &flatPreferred[int(chroma)]
		} else {
			sp = &sharpPreferred[int(chroma)]
		}
	}
	base := float64(degreeSemitones[sp.degree] + sp.offset)
	return *sp, math.Floor((value-base)/12) - 1, nil
}

func property(parent *node, name, tag string, value *string) *node {
	p := parent.add("Property").setAttr("name", name)
	t := p.add(tag)
	if value != nil {
		t.setText(*value)
	}
	return p
}

func str(s string) *string { return &s }

func joinNumbers(xs []float64, sep string) string {
	parts := make([]string, len(xs))
	for i, x := range xs {
		parts[i] = jsString(x)
	}
	return strings.Join(parts, sep)
}

type writer struct {
	rhythms       map[string]string
	tuningByStaff map[*gStaff][]float64
}

// writeGPIF is GpifWriter._writeDom.
func writeGPIF(g *gScore) (*node, error) {
	w := &writer{rhythms: map[string]string{}, tuningByStaff: map[*gStaff][]float64{}}
	gpif := &node{name: "GPIF"}
	gpif.add("GPVersion").setText("8.1.3")
	gpif.add("GPRevision").setAttr("required", "12024").setAttr("recommended", "13000").setText("13007")
	enc := gpif.add("Encoding")
	enc.add("EncodingDescription").setText("GP8")
	enc.children = append(enc.children, &node{comment: true, text: alphaTabComment})
	w.score(gpif, g)
	w.masterTrack(gpif, g)
	gpif.add("AudioTracks")
	tracks := gpif.add("Tracks")
	for _, tr := range g.tracks {
		w.track(tracks, tr)
	}
	masterBars := gpif.add("MasterBars")
	for _, mb := range g.masterBars {
		if err := w.masterBar(masterBars, g, mb); err != nil {
			return nil, err
		}
	}
	bars := gpif.add("Bars")
	voices := gpif.add("Voices")
	beats := gpif.add("Beats")
	notes := gpif.add("Notes")
	rhythms := gpif.add("Rhythms")
	for _, tr := range g.tracks {
		st := tr.staff
		needsFingering := staffNotesAreNotStringed(st)
		stringed := w.tuningByStaff[st]
		saved := st.tuning
		if needsFingering && len(saved) == 0 {
			st.tuning = append([]float64(nil), stringed...)
		}
		var assigner *fingering
		for _, bar := range st.bars {
			active := w.bar(bars, bar)
			for _, v := range active {
				if needsFingering && assigner == nil {
					a, err := newFingering(stringed)
					if err != nil {
						return nil, err
					}
					assigner = a
				}
				voices.add("Voice").setAttr("id", strconv.Itoa(v.id)).add("Beats").setText(beatIDs(v))
				for _, b := range v.beats {
					if assigner != nil {
						assigner.assign(b)
					}
					w.beat(beats, b, rhythms)
					for _, n := range b.notes {
						if err := w.note(notes, n); err != nil {
							return nil, err
						}
					}
				}
			}
		}
		st.tuning = saved
	}
	return gpif, nil
}

func beatIDs(v *gVoice) string {
	ids := make([]string, len(v.beats))
	for i, b := range v.beats {
		ids[i] = strconv.Itoa(b.id)
	}
	return strings.Join(ids, " ")
}

func (w *writer) score(parent *node, g *gScore) {
	s := parent.add("Score")
	for _, f := range []struct{ name, value string }{
		{"Title", g.title}, {"SubTitle", ""}, {"Artist", g.artist}, {"Album", ""}, {"Words", ""},
		{"Music", ""}, {"WordsAndMusic", ""}, {"Copyright", ""}, {"Tabber", g.tab},
		{"Instructions", ""}, {"Notices", g.notices}, {"FirstPageHeader", ""},
		{"FirstPageFooter", ""}, {"PageHeader", ""}, {"PageFooter", ""},
	} {
		s.add(f.name).setCData(f.value)
	}
	s.add("ScoreSystemsDefaultLayout").setText("3")
	s.add("ScoreSystemsLayout").setText("")
	s.add("ScoreZoomPolicy").setText("Value")
	s.add("ScoreZoom").setText("1")
	s.add("MultiVoice").setText("1>")
}

func tempoAutomationNode(parent *node, bar int, position string, visible bool, value float64) {
	a := parent.add("Automation")
	a.add("Type").setText("Tempo")
	a.add("Linear").setText("false")
	a.add("Bar").setText(strconv.Itoa(bar))
	a.add("Position").setText(position)
	a.add("Visible").setText(strconv.FormatBool(visible))
	a.add("Value").setText(strconv.FormatInt(toInt32(value), 10) + " 2")
}

func (w *writer) masterTrack(parent *node, g *gScore) {
	m := parent.add("MasterTrack")
	ids := make([]string, len(g.tracks))
	for i := range g.tracks {
		ids[i] = strconv.Itoa(i)
	}
	m.add("Tracks").setText(strings.Join(ids, " "))
	autos := m.add("Automations")
	if len(g.masterBars[0].tempos) == 0 {
		tempoAutomationNode(autos, 0, "0", true, 120)
	}
	for _, mb := range g.masterBars {
		for _, t := range mb.tempos {
			tempoAutomationNode(autos, mb.index, jsString(t.ratio), t.visible, t.value)
		}
	}
}

func isGuitar(program int) bool {
	return program >= 24 && program <= 39 || program == 105 || program == 43
}

func (w *writer) track(parent *node, tr *gTrack) {
	t := parent.add("Track").setAttr("id", strconv.Itoa(tr.index))
	t.add("Name").setCData(tr.name)
	t.add("ShortName").setCData(tr.shortName)
	t.add("Color").setText("200 0 0")
	t.add("SystemsDefautLayout").setText("3")
	t.add("SystemsLayout").setText("")
	t.add("AutoBrush")
	t.add("PalmMute").setText("0")
	style := "Default"
	if isGuitar(tr.program) {
		style = "StringedPick"
	}
	t.add("PlayingStyle").setText(style)
	t.add("UseOneChannelPerString")
	icon := 1
	switch {
	case tr.primary == 9:
		icon = 18
	case tr.program >= 0 && tr.program < 128:
		icon = programInfo[tr.program].icon
	}
	t.add("IconId").setText(strconv.Itoa(icon))
	instrumentSet(t, tr)
	tp := t.add("Transpose")
	tp.add("Chromatic").setText("0")
	tp.add("Octave").setText("0")
	cs := t.add("RSE").add("ChannelStrip").setAttr("version", "E56")
	cs.add("Parameters").setText("0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 1 0.5 0.5 0.9375 0.5 0.5 0.5")
	t.add("ForcedSound").setText("-1")
	mc := t.add("MidiConnection")
	mc.add("Port").setText("1")
	mc.add("PrimaryChannel").setText(strconv.Itoa(tr.primary))
	mc.add("SecondaryChannel").setText(strconv.Itoa(tr.secondary))
	mc.add("ForeOneChannelPerString").setText("false")
	t.add("PlaybackState").setText("Default")
	t.add("AudioEngineState").setText("MIDI")
	t.add("Lyrics").setAttr("dispatched", "true")
	w.staff(t.add("Staves"), tr.staff)
	sounds := t.add("Sounds")
	autos := t.add("Automations")
	for _, bar := range tr.staff.bars {
		for _, v := range bar.voices {
			for _, b := range v.beats {
				if !b.instrument {
					continue
				}
				name := fmt.Sprintf("Track_%d_Initial", tr.index)
				path := fmt.Sprintf("Midi/%d", tr.program)
				snd := sounds.add("Sound")
				snd.add("Name").setCData(name)
				snd.add("Label").setCData(name)
				snd.add("Path").setCData(path)
				snd.add("Role").setCData("Factory")
				midi := snd.add("MIDI")
				midi.add("LSB").setText("0")
				midi.add("MSB").setText("0")
				midi.add("Program").setText(strconv.Itoa(b.instrumentProgram))
				a := autos.add("Automation")
				a.add("Type").setText("Sound")
				a.add("Linear").setText("false")
				a.add("Bar").setText(strconv.Itoa(bar.index))
				a.add("Position").setText("0")
				a.add("Visible").setText("true")
				a.add("Value").setCData(path + ";" + name + ";Factory")
			}
		}
	}
}

func instrumentSet(t *node, tr *gTrack) {
	set := t.add("InstrumentSet")
	if tr.staff.perc {
		set.add("Name").setText("Drums")
		set.add("Type").setText("drumKit")
		set.add("LineCount").setText("5")
		els := set.add("Elements")
		for _, el := range drumKit {
			e := els.add("Element")
			e.add("Name").setText(el.name)
			e.add("Type").setText(el.kind)
			e.add("SoundbankName").setText(el.soundbank)
			arts := e.add("Articulations")
			for _, a := range el.articulations {
				articulationNode(arts, a)
			}
		}
		return
	}
	info := programInfo[0]
	if tr.program >= 0 && tr.program < 128 {
		info = programInfo[tr.program]
	}
	set.add("Name").setText(info.name)
	set.add("Type").setText(info.kind)
	set.add("LineCount").setText("5")
	e := set.add("Elements").add("Element")
	e.add("Name").setText("Pitched")
	e.add("Type").setText("pitched")
	e.add("SoundbankName").setText("")
	articulationNode(e.add("Articulations"), drumArticulation{"", 0,
		"noteheadBlack noteheadHalf noteheadWhole", "outside", "", "", "", 0})
}

func articulationNode(parent *node, a drumArticulation) {
	n := parent.add("Articulation")
	n.add("Name").setText(a.name)
	n.add("StaffLine").setText(strconv.Itoa(a.staffLine))
	n.add("Noteheads").setText(a.noteheads)
	n.add("TechniquePlacement").setText(a.placement)
	n.add("TechniqueSymbol").setText(a.symb)
	n.add("InputMidiNumbers").setText(a.inputMidi)
	n.add("OutputRSESound").setText(a.rseSound)
	n.add("OutputMidiNumber").setText(strconv.Itoa(a.outputMidi))
}

func (w *writer) staff(parent *node, st *gStaff) {
	props := parent.add("Staff").add("Properties")
	property(props, "CapoFret", "Fret", str("0"))
	property(props, "FretCount", "Fret", str("24"))
	tuning, name := st.tuning, st.tuningName
	if len(tuning) == 0 {
		if st.perc {
			tuning, name = []float64{0, 0, 0, 0, 0, 0}, ""
		} else if staffNotesAreNotStringed(st) {
			d := tuningPresets[6][0]
			tuning = nil
			for _, p := range d.tunings {
				tuning = append(tuning, float64(p))
			}
			name = d.name
		}
	}
	w.tuningByStaff[st] = tuning
	if len(tuning) > 0 {
		p := props.add("Property").setAttr("name", "Tuning")
		rev := make([]float64, len(tuning))
		for i, x := range tuning {
			rev[len(tuning)-1-i] = x
		}
		p.add("Pitches").setText(joinNumbers(rev, " "))
		p.add("Label").setCData(name)
		p.add("LabelVisible").setText(strconv.FormatBool(name != ""))
		p.add("Flat")
		p.add("Instrument").setText(tuningInstrument(st, len(tuning)))
	}
	property(props, "PartialCapoFret", "Fret", str("0"))
	property(props, "PartialCapoStringFlags", "Bitset", str(strings.Repeat("0", len(st.tuning))))
	property(props, "TuningFlat", "Enable", nil)
	props.add("Property").setAttr("name", "DiagramCollection").add("Items")
	props.add("Property").setAttr("name", "DiagramWorkingSet").add("Items")
}

func tuningInstrument(st *gStaff, strings int) string {
	program := st.track.program
	if st.perc {
		return "Undefined"
	}
	switch strings {
	case 3:
		return "Shamisen"
	case 4:
		switch program {
		case 105:
			return "Banjo"
		case 42:
			return "Cello"
		case 43:
			return "Contrabass"
		case 40:
			return "Violin"
		case 41:
			return "Viola"
		}
		return "Bass"
	case 5:
		if program == 105 {
			return "Banjo"
		}
		return "Bass"
	case 6:
		if program == 105 {
			return "Banjo"
		}
		if program <= 39 {
			return "Bass"
		}
		return "Guitar"
	case 7:
		if program <= 39 {
			return "Bass"
		}
		return "Guitar"
	}
	return "Guitar"
}

func (w *writer) masterBar(parent *node, g *gScore, mb *masterBar) error {
	m := parent.add("MasterBar")
	key := m.add("Key")
	first := g.tracks[0].staff.bars[mb.index]
	key.add("AccidentalCount").setText(jsString(first.ks))
	mode := "Major"
	if first.minor {
		mode = "Minor"
	}
	key.add("Mode").setText(mode)
	key.add("Sharps").setText("Sharps")
	m.add("Time").setText(fmt.Sprintf("%d/%d", mb.num, mb.den))
	ids := make([]string, len(g.tracks))
	for i, tr := range g.tracks {
		if mb.index >= len(tr.staff.bars) {
			return fmt.Errorf("TypeError: Cannot read properties of undefined (reading 'id'): track %d has no bar %d", i, mb.index)
		}
		ids[i] = strconv.Itoa(tr.staff.bars[mb.index].id)
	}
	m.add("Bars").setText(strings.Join(ids, " "))
	if mb.section != nil {
		s := m.add("Section")
		s.add("Letter").setCData("")
		s.add("Text").setCData(*mb.section)
	}
	return nil
}

func (w *writer) bar(parent *node, bar *gBar) []*gVoice {
	b := parent.add("Bar").setAttr("id", strconv.Itoa(bar.id))
	slots := []string{"-1", "-1", "-1", "-1"}
	var active []*gVoice
	for i, v := range bar.voices {
		if i < 4 && len(v.beats) > 0 {
			slots[i] = strconv.Itoa(v.id)
			active = append(active, v)
		}
	}
	b.add("Voices").setText(strings.Join(slots, " "))
	b.add("Clef").setText(bar.clef)
	return active
}

var noteValues = map[int]string{1: "Whole", 2: "Half", 4: "Quarter", 8: "Eighth", 16: "16th",
	32: "32nd", 64: "64th"}

func (w *writer) beat(parent *node, b *gBeat, rhythms *node) {
	n := parent.add("Beat").setAttr("id", strconv.Itoa(b.id))
	n.add("Dynamic").setText("F")
	if b.text != "" {
		n.add("FreeText").setCData(b.text)
	}
	id := fmt.Sprintf("%d_%d_%d_%d';", b.duration, b.dots, b.tupNum, b.tupDen)
	ref, ok := w.rhythms[id]
	if !ok {
		ref = strconv.Itoa(len(w.rhythms))
		w.rhythms[id] = ref
		r := rhythms.add("Rhythm").setAttr("id", ref)
		if !(b.tupDen == -1 && b.tupNum == -1) && !(b.tupDen == 1 && b.tupNum == 1) {
			r.add("PrimaryTuplet").setAttr("num", strconv.Itoa(b.tupNum)).setAttr("den", strconv.Itoa(b.tupDen))
		}
		if b.dots > 0 {
			r.add("AugmentationDot").setAttr("count", strconv.Itoa(b.dots))
		}
		r.add("NoteValue").setText(noteValues[b.duration])
	}
	n.add("Rhythm").setAttr("ref", ref)
	n.add("ConcertPitchStemOrientation").setText("Undefined")
	if len(b.notes) > 0 {
		ids := make([]string, len(b.notes))
		for i, x := range b.notes {
			ids[i] = strconv.Itoa(x.id)
		}
		n.add("Notes").setText(strings.Join(ids, " "))
	}
	n.add("Properties")
	n.add("XProperties")
}

func pitchProperty(props *node, name, step, octave, accidental string) {
	p := props.add("Property").setAttr("name", name).add("Pitch")
	p.add("Step").setText(step)
	p.add("Accidental").setText(accidental)
	p.add("Octave").setText(octave)
}

func (w *writer) note(parent *node, n *gNote) error {
	el := parent.add("Note").setAttr("id", strconv.Itoa(n.id))
	props := el.add("Properties")
	if n.isPerc() {
		pitchProperty(props, "ConcertPitch", "C", "-1", "")
		pitchProperty(props, "TransposedPitch", "C", "-1", "")
	} else {
		sp, octave, err := resolveSpelling(n.beat.voice.bar.ks, n.realValue())
		if err != nil {
			return err
		}
		for _, name := range []string{"ConcertPitch", "TransposedPitch"} {
			pitchProperty(props, name, steps[sp.degree], jsString(octave+1), accidentals[sp.offset+2])
		}
	}
	switch {
	case n.isPerc():
		midi := 0.0
		if a := articulationOf(n); a != nil {
			midi = a.outputMidi
		}
		property(props, "String", "String", str(jsString(n.str-1)))
		property(props, "Fret", "Fret", str(jsString(midi)))
		property(props, "Midi", "Number", str(jsString(midi)))
	case n.isStringed():
		property(props, "String", "String", str(jsString(n.str-1)))
		property(props, "Fret", "Fret", str(jsString(n.fret)))
		property(props, "Midi", "Number", str(jsString(n.realValue())))
	}
	if n.isPiano() {
		property(props, "Octave", "Number", str(jsString(n.octave)))
		property(props, "Tone", "Step", str(jsString(n.tone)))
		property(props, "Midi", "Number", str(jsString(n.realValue())))
	}
	if n.tieDestination != nil || n.tieDest {
		el.add("Tie").setAttr("origin", strconv.FormatBool(n.tieDestination != nil)).
			setAttr("destination", strconv.FormatBool(n.tieDest))
	}
	art := "0"
	if n.perc >= 0 {
		art = jsString(n.perc)
	}
	el.add("InstrumentArticulation").setText(art)
	return nil
}

// partConfiguration is PartConfiguration.writeForScore: the multi-track view, then one view per
// track, with the standard-notation (1) and tablature (2) flags.
func partConfiguration(g *gScore) []byte {
	flags := make([]byte, len(g.tracks))
	for i, tr := range g.tracks {
		if tr.staff.showStd {
			flags[i] |= 1
		}
		if tr.staff.showTab {
			flags[i] |= 2
		}
	}
	var out []byte
	out = binary.BigEndian.AppendUint32(out, uint32(1+len(g.tracks)))
	out = append(out, 0)
	out = binary.BigEndian.AppendUint32(out, uint32(len(g.tracks)))
	out = append(out, flags...)
	for _, f := range flags {
		out = append(out, 0)
		out = binary.BigEndian.AppendUint32(out, 1)
		out = append(out, f)
	}
	return binary.BigEndian.AppendUint32(out, 1)
}

// layoutConfiguration is LayoutConfiguration.writeForScore for single-voice bars and visible
// tracks.
func layoutConfiguration(g *gScore) []byte {
	out := binary.BigEndian.AppendUint32(nil, 4)
	out = append(out, 0, 0)
	for range g.tracks {
		out = append(out, 0xFF)
	}
	for range g.tracks {
		out = append(out, 0xFF)
	}
	return out
}

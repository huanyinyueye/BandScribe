// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy
// of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// The reading half is translated from alphaTab 1.9.0 (https://github.com/CoderLine/alphaTab),
// Copyright CoderLine: the parts of GpifParser, ModelUtils.consolidate and Score.finish that
// decide what the BandScribe verification sees. The comparison is verify() of
// bandscribe/notation/export_gp.py.

package gp

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

type readMasterBar struct {
	num, den float64
	section  *string
}

type readTrack struct {
	perc   bool
	tuning []float64
	bars   int
	notes  [][5]any // [bar, pitch or articulation, string, fret, tie]; nil = JSON null
}

type readScore struct {
	masterBars []readMasterBar
	tracks     []readTrack
}

func childElements(n *node, name string) []*node {
	var out []*node
	if n == nil {
		return out
	}
	for _, c := range n.children {
		if c.name == name {
			out = append(out, c)
		}
	}
	return out
}

func innerText(n *node) string {
	if n == nil {
		return ""
	}
	return n.innerText()
}

type readNoteData struct {
	str, fret, octave, tone, perc float64
	tieDest                       bool
}

// readGPIF is Gp7To8Importer/GpifParser reading score.gpif, followed by Score.finish.
func readGPIF(xml []byte) (*readScore, error) {
	doc, err := parseXML(string(xml))
	if err != nil {
		return nil, err
	}
	if len(doc.children) == 0 || doc.children[0].name != "GPIF" {
		return nil, fmt.Errorf("UnsupportedFormatError: Root node of XML was not GPIF")
	}
	root := doc.children[0]
	var mapping []string
	tracksByID := map[string]*gTrack{}
	var masterBars []readMasterBar
	var barsOfMaster [][]string
	voicesOfBar := map[string][]string{}
	barsByID := map[string]bool{}
	beatsOfVoice := map[string][]string{}
	voicesByID := map[string]bool{}
	notesOfBeat := map[string][]string{}
	beatsByID := map[string]bool{}
	notesByID := map[string]readNoteData{}
	for _, n := range root.children {
		switch n.name {
		case "MasterTrack":
			for _, c := range n.children {
				if c.name == "Tracks" {
					mapping = splitSafe(c.innerText())
				}
			}
		case "Tracks":
			for _, t := range childElements(n, "Track") {
				id, _ := t.attr("id")
				tracksByID[id] = readTrackNode(t)
			}
		case "MasterBars":
			for _, mb := range childElements(n, "MasterBar") {
				m := readMasterBar{num: 4, den: 4}
				for _, c := range mb.children {
					switch c.name {
					case "Time":
						parts := strings.Split(c.innerText(), "/")
						m.num = parseIntSafe(parts[0], 4)
						if len(parts) > 1 {
							m.den = parseIntSafe(parts[1], 4)
						} else {
							m.den = 4
						}
					case "Section":
						s := innerText(c.child("Text"))
						m.section = &s
					case "Bars":
						barsOfMaster = append(barsOfMaster, splitSafe(c.innerText()))
					}
				}
				masterBars = append(masterBars, m)
			}
		case "Bars":
			for _, b := range childElements(n, "Bar") {
				id, _ := b.attr("id")
				barsByID[id] = true
				for _, c := range b.children {
					if c.name == "Voices" {
						voicesOfBar[id] = splitSafe(c.innerText())
					}
				}
			}
		case "Voices":
			for _, v := range childElements(n, "Voice") {
				id, _ := v.attr("id")
				voicesByID[id] = true
				for _, c := range v.children {
					if c.name == "Beats" {
						beatsOfVoice[id] = splitSafe(c.innerText())
					}
				}
			}
		case "Beats":
			for _, b := range childElements(n, "Beat") {
				id, _ := b.attr("id")
				beatsByID[id] = true
				for _, c := range b.children {
					if c.name == "Notes" {
						notesOfBeat[id] = splitSafe(c.innerText())
					}
				}
			}
		case "Notes":
			for _, el := range childElements(n, "Note") {
				id, _ := el.attr("id")
				notesByID[id] = readNoteNode(el)
			}
		}
	}

	var tracks []*gTrack
	for _, id := range mapping {
		t, ok := tracksByID[id]
		if !ok {
			return nil, fmt.Errorf("TypeError: track %s is not defined", id)
		}
		t.index = len(tracks)
		tracks = append(tracks, t)
	}
	if len(tracks) == 0 {
		return nil, fmt.Errorf("TypeError: the file has no tracks")
	}
	for _, barIDs := range barsOfMaster {
		trackIndex := 0
		for _, barID := range barIDs {
			if trackIndex >= len(tracks) {
				break
			}
			if barID == "-1" {
				trackIndex++
				continue
			}
			if !barsByID[barID] {
				return nil, fmt.Errorf("TypeError: bar %s is not defined", barID)
			}
			st := tracks[trackIndex].staff
			bar := &gBar{index: len(st.bars), staff: st}
			if len(st.bars) > 0 {
				st.bars[len(st.bars)-1].next = bar
			}
			st.bars = append(st.bars, bar)
			for _, vid := range voicesOfBar[barID] {
				if vid == "-1" || len(bar.voices) > 0 {
					continue // one voice per bar: further voices never occur in what we write
				}
				if !voicesByID[vid] {
					return nil, fmt.Errorf("TypeError: voice %s is not defined", vid)
				}
				v := &gVoice{bar: bar}
				bar.voices = append(bar.voices, v)
				for _, bid := range beatsOfVoice[vid] {
					if bid == "-1" {
						continue
					}
					if !beatsByID[bid] {
						return nil, fmt.Errorf("TypeError: beat %s is not defined", bid)
					}
					beat := &gBeat{index: len(v.beats), voice: v, strLookup: map[float64]*gNote{},
						valLookup: map[float64]*gNote{}}
					v.beats = append(v.beats, beat)
					for _, nid := range notesOfBeat[bid] {
						if nid == "-1" {
							continue
						}
						d, ok := notesByID[nid]
						if !ok {
							return nil, fmt.Errorf("TypeError: note %s is not defined", nid)
						}
						n := &gNote{str: d.str, fret: d.fret, octave: d.octave, tone: d.tone,
							perc: d.perc, tieDest: d.tieDest}
						if st.perc {
							n.fret = nan
						} else {
							n.perc = nan
						}
						beat.addNote(n)
						if st.perc && n.perc >= 0 {
							if known := articulationOf(n); known != nil {
								n.perc = float64(st.track.registerArticulation(known))
							}
						}
					}
				}
			}
			if len(bar.voices) == 0 {
				bar.voices = []*gVoice{{bar: bar}}
			}
			trackIndex++
		}
	}
	rs := &readScore{}
	for _, m := range masterBars {
		rs.masterBars = append(rs.masterBars, m)
	}
	for _, tr := range tracks {
		st := tr.staff
		if !st.perc {
			tr.percArticulations = nil
		}
		for len(st.bars) < len(masterBars) {
			bar := &gBar{index: len(st.bars), staff: st, voices: nil}
			bar.voices = []*gVoice{{bar: bar}}
			if len(st.bars) > 0 {
				st.bars[len(st.bars)-1].next = bar
			}
			st.bars = append(st.bars, bar)
		}
		for _, bar := range st.bars {
			for _, v := range bar.voices {
				if len(v.beats) == 0 { // ModelUtils.consolidate: an empty beat
					v.beats = []*gBeat{{voice: v, strLookup: map[float64]*gNote{},
						valLookup: map[float64]*gNote{}}}
				}
			}
		}
		if st.perc {
			st.tuning = []float64{0, 0, 0, 0, 0, 0}
		}
		for _, bar := range st.bars {
			for _, v := range bar.voices {
				if err := v.finish(); err != nil {
					return nil, err
				}
			}
		}
		rt := readTrack{perc: st.perc, tuning: st.tuning, bars: len(st.bars)}
		nStrings := float64(len(st.tuning))
		for bi, bar := range st.bars {
			for _, b := range bar.voices[0].beats {
				for _, n := range b.notes {
					tie := 0
					if n.tieDest {
						tie = 1
					}
					if st.perc {
						rt.notes = append(rt.notes, [5]any{bi, jsonNumber(n.perc), nil, nil, tie})
						continue
					}
					entry := [5]any{bi, jsonNumber(n.realValue()), nil, nil, tie}
					if n.isStringed() {
						entry[2], entry[3] = jsonNumber(nStrings-n.str+1), jsonNumber(n.fret)
					}
					rt.notes = append(rt.notes, entry)
				}
			}
		}
		rs.tracks = append(rs.tracks, rt)
	}
	return rs, nil
}

// jsonNumber is a number after JSON.stringify: NaN and infinities become null.
func jsonNumber(x float64) any {
	if math.IsNaN(x) || math.IsInf(x, 0) {
		return nil
	}
	return x
}

func (t *gTrack) registerArticulation(a *articulation) int {
	for i, x := range t.percArticulations {
		if x.uniqueID() == a.uniqueID() {
			return i
		}
	}
	t.percArticulations = append(t.percArticulations, a)
	return len(t.percArticulations) - 1
}

func readTrackNode(t *node) *gTrack {
	tr := &gTrack{}
	st := &gStaff{track: tr}
	tr.staff = st
	for _, c := range t.children {
		switch c.name {
		case "InstrumentSet":
			for _, x := range c.children {
				switch x.name {
				case "Type":
					if x.innerText() == "drumKit" {
						st.perc = true
					}
				case "Elements":
					for _, el := range childElements(x, "Element") {
						elName := innerText(el.child("Name"))
						for _, arts := range childElements(el, "Articulations") {
							for _, a := range childElements(arts, "Articulation") {
								art := &articulation{elementType: elName, outputMidi: -1}
								for _, f := range a.children {
									switch f.name {
									case "InputMidiNumbers":
										art.id = parseIntSafe(strings.Split(f.innerText(), " ")[0], 0)
									case "OutputMidiNumber":
										art.outputMidi = parseIntSafe(f.innerText(), 0)
									case "StaffLine":
										art.staffLine = int(parseIntSafe(f.innerText(), 0))
									}
								}
								tr.percArticulations = append(tr.percArticulations, art)
							}
						}
					}
				}
			}
		case "Staves":
			staves := childElements(c, "Staff")
			if len(staves) == 0 {
				continue
			}
			for _, props := range childElements(staves[0], "Properties") {
				for _, p := range childElements(props, "Property") {
					if v, _ := p.attr("name"); v != "Tuning" {
						continue
					}
					for _, x := range p.children {
						if x.name != "Pitches" {
							continue
						}
						parts := splitSafe(innerText(p.child("Pitches")))
						tuning := make([]float64, len(parts))
						for i, s := range parts {
							tuning[len(parts)-1-i] = parseIntSafe(s, 0)
						}
						st.tuning = tuning
					}
				}
			}
		}
	}
	return tr
}

func readNoteNode(el *node) readNoteData {
	d := readNoteData{str: nan, fret: nan, octave: nan, tone: nan, perc: nan}
	for _, c := range el.children {
		switch c.name {
		case "Properties":
			for _, p := range childElements(c, "Property") {
				switch name, _ := p.attr("name"); name {
				case "String":
					d.str = parseIntSafe(innerText(p.child("String")), 0) + 1
				case "Fret":
					d.fret = parseIntSafe(innerText(p.child("Fret")), 0)
				case "Octave":
					d.octave = parseIntSafe(innerText(p.child("Number")), 0)
					if isNaN(d.tone) {
						d.tone = 0
					}
				case "Tone":
					d.tone = parseIntSafe(innerText(p.child("Step")), 0)
				}
			}
		case "Tie":
			if v, _ := c.attr("destination"); strings.ToLower(v) == "true" {
				d.tieDest = true
			}
		case "InstrumentArticulation":
			d.perc = parseIntSafe(c.innerText(), 0)
		}
	}
	return d
}

// expectedNotes is _expected_notes of export_gp.py.
func expectedNotes(t *gTrack) [][5]any {
	var out [][5]any
	tab := t.sent.Kind == "tab"
	for bi, beats := range t.sentBars {
		for _, b := range beats {
			tie := 0
			if b.Tied {
				tie = 1
			}
			if b.Rest {
				continue
			}
			for _, n := range b.Notes {
				e := [5]any{bi, float64(n.Pitch), nil, nil, tie}
				if tab {
					if n.String != nil {
						e[2] = float64(*n.String)
					}
					if n.Fret != nil {
						e[3] = float64(*n.Fret)
					}
				}
				out = append(out, e)
			}
		}
	}
	return out
}

func orZero(v any) float64 {
	if f, ok := v.(float64); ok {
		return f
	}
	return 0
}

func numberOf(v any) float64 {
	switch x := v.(type) {
	case int:
		return float64(x)
	case float64:
		return x
	}
	return 0
}

// sortNotes is the stable sort on (bar, pitch, string or 0, fret or 0, tie) of verify().
func sortNotes(notes [][5]any) error {
	for _, n := range notes {
		if n[1] == nil && len(notes) > 1 {
			return fmt.Errorf("TypeError: '<' not supported between instances of 'NoneType' and 'int'")
		}
	}
	sort.SliceStable(notes, func(i, j int) bool {
		a, b := notes[i], notes[j]
		ka := [5]float64{numberOf(a[0]), numberOf(a[1]), orZero(a[2]), orZero(a[3]), numberOf(a[4])}
		kb := [5]float64{numberOf(b[0]), numberOf(b[1]), orZero(b[2]), orZero(b[3]), numberOf(b[4])}
		for k := range ka {
			if ka[k] != kb[k] {
				return ka[k] < kb[k]
			}
		}
		return false
	})
	return nil
}

func sameValue(a, b any) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	return numberOf(a) == numberOf(b)
}

func pyList(xs []float64) string {
	parts := make([]string, len(xs))
	for i, x := range xs {
		parts[i] = jsString(x)
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

// verify is verify() of export_gp.py: the differences between what was sent and what was read.
func verify(g *gScore, rs *readScore) ([]string, error) {
	var issues []string
	same := len(g.masterBars) == len(rs.masterBars)
	for i := 0; same && i < len(g.masterBars); i++ {
		same = float64(g.masterBars[i].num) == rs.masterBars[i].num &&
			float64(g.masterBars[i].den) == rs.masterBars[i].den
	}
	if !same {
		issues = append(issues, fmt.Sprintf("time signatures differ: sent %d bars, read %d",
			len(g.masterBars), len(rs.masterBars)))
	}
	same = len(g.masterBars) == len(rs.masterBars)
	for i := 0; same && i < len(g.masterBars); i++ {
		a, b := g.masterBars[i].sentText, rs.masterBars[i].section
		same = (a == nil) == (b == nil) && (a == nil || *a == *b)
	}
	if !same {
		issues = append(issues, "section markers differ")
	}
	if len(g.tracks) != len(rs.tracks) {
		return append(issues, fmt.Sprintf("track count differs: %d vs %d", len(g.tracks), len(rs.tracks))), nil
	}
	for i, t := range g.tracks {
		ts := rs.tracks[i]
		name := t.sent.Name
		if ts.bars != len(t.sentBars) {
			issues = append(issues, fmt.Sprintf("%s: %d bars read, %d sent", name, ts.bars, len(t.sentBars)))
		}
		if t.sent.Kind == "tab" {
			sent := make([]float64, len(t.sent.Tuning))
			for k, p := range t.sent.Tuning {
				sent[k] = float64(p)
			}
			if pyList(ts.tuning) != pyList(sent) {
				issues = append(issues, fmt.Sprintf("%s: tuning differs: %s vs %s", name, pyList(ts.tuning), pyList(sent)))
			}
		}
		if (t.sent.Kind == "drums") != ts.perc {
			issues = append(issues, fmt.Sprintf("%s: percussion flag differs", name))
		}
		exp := expectedNotes(t)
		got := append([][5]any(nil), ts.notes...)
		if t.sent.Kind != "tab" {
			for k := range got {
				got[k][2], got[k][3] = nil, nil
			}
		}
		if err := sortNotes(exp); err != nil {
			return nil, err
		}
		if err := sortNotes(got); err != nil {
			return nil, err
		}
		diff := -1
		for k := 0; k < min(len(exp), len(got)); k++ {
			for f := 0; f < 5; f++ {
				if !sameValue(exp[k][f], got[k][f]) {
					diff = k
					break
				}
			}
			if diff >= 0 {
				break
			}
		}
		if diff < 0 && len(exp) != len(got) {
			diff = min(len(exp), len(got))
		}
		if diff >= 0 {
			issues = append(issues, fmt.Sprintf("%s: notes differ (%d sent, %d read; first difference at index %d)",
				name, len(exp), len(got), diff))
		}
	}
	return issues, nil
}

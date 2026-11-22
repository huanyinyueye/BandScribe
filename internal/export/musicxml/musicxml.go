// Package musicxml writes MusicXML 4.0 (port of bandscribe/notation/export_musicxml.py),
// byte-for-byte equal to the Python ElementTree output for the same score.
package musicxml

import (
	"math"
	"os"
	"path/filepath"
	"strconv"

	"bandscribe/internal/layout"
	"bandscribe/internal/pyfmt"
	"bandscribe/internal/rat"
	"bandscribe/internal/score"
	x "bandscribe/internal/xmltree"
)

const Divisions = 960

var types = map[int]string{1: "whole", 2: "half", 4: "quarter", 8: "eighth", 16: "16th", 32: "32nd", 64: "64th"}

var steps = [12]struct {
	step  string
	alter int
}{{"C", 0}, {"C", 1}, {"D", 0}, {"D", 1}, {"E", 0}, {"F", 0}, {"F", 1}, {"G", 0}, {"G", 1}, {"A", 0}, {"A", 1}, {"B", 0}}

type drumDisplay struct {
	step   string
	octave int
	head   string
}

var drums = map[int]drumDisplay{36: {"F", 4, ""}, 38: {"C", 5, ""}, 42: {"G", 5, "x"}, 44: {"D", 4, "x"},
	46: {"G", 5, "circle-x"}, 43: {"A", 4, ""}, 47: {"D", 5, ""}, 50: {"E", 5, ""}, 49: {"A", 5, "x"},
	51: {"F", 5, "x"}, 52: {"B", 5, "x"}, 55: {"B", 5, "x"}}

func itoa(i int) *string { return x.Str(strconv.Itoa(i)) }

func pyDiv(a, b int) int { return int(math.Floor(float64(a) / float64(b))) }

func clef(t *score.Track) (sign string, line int, octaveChange int) {
	switch {
	case t.Kind == "drums":
		return "percussion", 2, 0
	case t.Family == "bass":
		if t.Kind == "tab" {
			return "F", 4, -1
		}
		return "F", 4, 0
	case t.Family == "guitar":
		return "G", 2, -1
	}
	return "G", 2, 0
}

func duration(b layout.Beat) *string {
	return itoa(int(b.LengthQ.Mul(rat.Int(Divisions)).Trunc()))
}

func rhythm(el *x.Element, b layout.Beat) {
	el.Sub("type", x.Str(types[b.Duration]))
	for i := 0; i < b.Dots; i++ {
		el.Sub("dot", nil)
	}
	if b.Tuplet != nil {
		tm := el.Sub("time-modification", nil)
		tm.Sub("actual-notes", itoa(b.Tuplet[0]))
		tm.Sub("normal-notes", itoa(b.Tuplet[1]))
	}
}

func note(measure *x.Element, t *score.Track, b layout.Beat, n *score.Note, chord, tieStart bool, octaveShift int) {
	el := measure.Sub("note", nil)
	if chord {
		el.Sub("chord", nil)
	}
	head := ""
	if t.Kind == "drums" {
		d, ok := drums[n.Pitch]
		if !ok {
			d = drumDisplay{"C", 5, ""}
		}
		head = d.head
		up := el.Sub("unpitched", nil)
		up.Sub("display-step", x.Str(d.step))
		up.Sub("display-octave", itoa(d.octave))
	} else {
		st := steps[((n.Pitch%12)+12)%12]
		p := el.Sub("pitch", nil)
		p.Sub("step", x.Str(st.step))
		if st.alter != 0 {
			p.Sub("alter", itoa(st.alter))
		}
		p.Sub("octave", itoa(pyDiv(n.Pitch, 12)-1+octaveShift))
	}
	el.Sub("duration", duration(b))
	if b.Tied {
		el.Sub("tie", nil, "type", "stop")
	}
	if tieStart {
		el.Sub("tie", nil, "type", "start")
	}
	el.Sub("voice", itoa(1))
	rhythm(el, b)
	if head != "" {
		el.Sub("notehead", x.Str(head))
	}
	technical := t.Kind == "tab" && n.String != nil
	if b.Tied || tieStart || technical {
		notations := el.Sub("notations", nil)
		if b.Tied {
			notations.Sub("tied", nil, "type", "stop")
		}
		if tieStart {
			notations.Sub("tied", nil, "type", "start")
		}
		if technical {
			tech := notations.Sub("technical", nil)
			tech.Sub("string", itoa(*n.String))
			tech.Sub("fret", optInt(n.Fret))
		}
	}
}

// optInt mirrors _sub(parent, tag, value): None leaves the element empty.
func optInt(v *int) *string {
	if v == nil {
		return nil
	}
	return itoa(*v)
}

// Build returns the score-partwise root element (not yet indented).
func Build(s *score.Score) *x.Element {
	root := x.New("score-partwise", "version", "4.0")
	work := root.Sub("work", nil)
	title := s.Title
	if title == "" {
		title = "Untitled"
	}
	work.Sub("work-title", x.Str(title))
	ident := root.Sub("identification", nil)
	enc := ident.Sub("encoding", nil)
	enc.Sub("software", x.Str("BandScribe"))
	ident.Sub("rights", x.Str("Automatic transcription. '?' marks low-confidence notes. Non-commercial use only."))
	partList := root.Sub("part-list", nil)
	for i := 1; i <= len(s.Tracks); i++ {
		t := &s.Tracks[i-1]
		id := "P" + strconv.Itoa(i)
		sp := partList.Sub("score-part", nil, "id", id)
		sp.Sub("part-name", x.Str(t.Name))
		mi := sp.Sub("midi-instrument", nil, "id", id+"-I1")
		ch := 10
		prog := 1
		if t.Kind != "drums" {
			if i < 10 {
				ch = min(16, i)
			} else {
				ch = min(16, i+1)
			}
			prog = t.MidiProgram + 1
		}
		mi.Sub("midi-channel", itoa(ch))
		mi.Sub("midi-program", itoa(prog))
	}
	fifths, hasFifths := s.KeyFifths()
	for i := 1; i <= len(s.Tracks); i++ {
		t := &s.Tracks[i-1]
		part := root.Sub("part", nil, "id", "P"+strconv.Itoa(i))
		lay := layout.Track(t, s.Bars)
		sign, line, octaveChange := clef(t)
		octaveShift := 0
		if octaveChange == -1 {
			octaveShift = 1
		}
		type pos struct{ bi, k int }
		var flat []pos
		for bi, beats := range lay {
			for k := range beats {
				flat = append(flat, pos{bi, k})
			}
		}
		nxtTied := map[pos]bool{}
		for idx := 0; idx+1 < len(flat); idx++ {
			nb := flat[idx+1]
			nxtTied[flat[idx]] = lay[nb.bi][nb.k].Tied
		}
		var lastSig [2]int
		haveSig := false
		var lastBPM, lastSoundBPM float64
		haveBPM, haveSound := false, false
		for bi, bar := range s.Bars {
			beats := lay[bi]
			measure := part.Sub("measure", nil, "number", strconv.Itoa(bar.Index+1))
			sig := [2]int{bar.Numerator, bar.Denominator}
			if bar.Index == 0 || !haveSig || sig != lastSig {
				attrs := measure.Sub("attributes", nil)
				if bar.Index == 0 {
					attrs.Sub("divisions", itoa(Divisions))
					if hasFifths && t.Kind != "drums" {
						key := attrs.Sub("key", nil)
						key.Sub("fifths", itoa(fifths))
						mode := s.KeyMode()
						if _, ok := s.Key["mode"]; !ok {
							mode = "major"
						}
						key.Sub("mode", x.Str(mode))
					}
				}
				tm := attrs.Sub("time", nil)
				tm.Sub("beats", itoa(bar.Numerator))
				tm.Sub("beat-type", itoa(bar.Denominator))
				if bar.Index == 0 {
					c := attrs.Sub("clef", nil)
					c.Sub("sign", x.Str(sign))
					if sign != "percussion" {
						c.Sub("line", itoa(line))
					}
					if octaveChange != 0 {
						c.Sub("clef-octave-change", itoa(octaveChange))
					}
				}
				lastSig, haveSig = sig, true
			}
			if i == 1 {
				visible := !haveBPM || math.Abs(bar.BPM-lastBPM) >= 1
				if visible {
					d := measure.Sub("direction", nil, "placement", "above")
					met := d.Sub("direction-type", nil).Sub("metronome", nil)
					met.Sub("beat-unit", x.Str("quarter"))
					met.Sub("per-minute", x.Str(strconv.FormatInt(pyfmt.RoundInt(bar.BPM), 10)))
					d.Sub("sound", nil, "tempo", pyfmt.Repr(pyfmt.Round(bar.BPM, 2)))
					lastBPM, haveBPM = bar.BPM, true
					lastSoundBPM, haveSound = bar.BPM, true
				} else if !haveSound || math.Abs(bar.BPM-lastSoundBPM) > 0.005 {
					measure.Sub("sound", nil, "tempo", pyfmt.Repr(pyfmt.Round(bar.BPM, 2)))
					lastSoundBPM, haveSound = bar.BPM, true
				}
			}
			if bar.Section != nil && *bar.Section != "" && i == 1 {
				d := measure.Sub("direction", nil, "placement", "above")
				d.Sub("direction-type", nil).Sub("rehearsal", x.Str(*bar.Section))
			}
			for k, b := range beats {
				if b.Text != "" {
					d := measure.Sub("direction", nil, "placement", "above")
					d.Sub("direction-type", nil).Sub("words", x.Str(b.Text))
				}
				if b.Rest {
					el := measure.Sub("note", nil)
					el.Sub("rest", nil)
					el.Sub("duration", duration(b))
					el.Sub("voice", itoa(1))
					rhythm(el, b)
					continue
				}
				tieStart := nxtTied[pos{bar.Index, k}]
				for j, n := range b.Notes {
					note(measure, t, b, n, j > 0, tieStart, octaveShift)
				}
			}
		}
	}
	return root
}

const header = `<?xml version="1.0" encoding="UTF-8"?>` + "\n" +
	`<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">` + "\n"

func Bytes(s *score.Score) []byte {
	root := Build(s)
	x.Indent(root)
	return []byte(header + root.String())
}

func Write(s *score.Score, path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, Bytes(s), 0o644)
}

// Package gp5 writes the Guitar Pro 5.10 `.gp5` fallback export (port of
// bandscribe/notation/export_gp5.py). The byte layout was derived clean-room from the GP5 file
// format and from observing the files the Python exporter wrote; the output is byte-identical.
//
// GP5 only knows fretted and percussion tracks. Staff tracks (vocals, piano, other) are written
// on a synthetic 7-string tuning in octaves so every pitch is kept; they are for reference only.
package gp5

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/bits"
	"os"
	"path/filepath"
	"sort"

	"bandscribe/internal/layout"
	"bandscribe/internal/score"
)

var staffTuning = []int{96, 84, 72, 60, 48, 36, 24} // string 1..7, an octave apart
var drumTuning = []int{64, 59, 55, 50, 45, 40}

const (
	defaultVelocity = 95 // forte; notes at this velocity carry no dynamic byte
	tabText         = "BandScribe (automatic transcription)"
)

var notice = []string{"'?' marks low-confidence notes.", "Non-commercial use only."}

type Report struct {
	Path         string   `json:"path"`
	NotesWritten int      `json:"notes_written"`
	NotesDropped int      `json:"notes_dropped"`
	Warnings     []string `json:"warnings"`
}

type position struct{ str, value, velocity int }

type track struct {
	src        *score.Track
	index      int // position in score.tracks; only index 0 carries tempo changes
	name       string
	tuning     []int
	drums      bool
	channel    int
	effect     int
	instrument int
}

// positions is _positions: (string, value, velocity) for each note of a beat, strings distinct.
func positions(t *score.Track, b *layout.Beat) ([]position, error) {
	var out []position
	used := map[int]bool{}
	switch t.Kind {
	case "tab":
		for _, n := range b.Notes {
			if n.String != nil && !used[*n.String] {
				if n.Fret == nil {
					return nil, errors.New("tab note without a fret")
				}
				used[*n.String] = true
				out = append(out, position{*n.String, *n.Fret, n.Velocity})
			}
		}
	case "drums":
		for i, n := range b.Notes {
			if i == 6 {
				break
			}
			out = append(out, position{i + 1, n.Pitch, n.Velocity})
		}
	default:
		notes := append([]*score.Note(nil), b.Notes...)
		sort.SliceStable(notes, func(i, j int) bool { return notes[i].Pitch > notes[j].Pitch })
		for _, n := range notes {
			for s, open := range staffTuning {
				fret := n.Pitch - open
				if fret >= 0 && fret <= 24 && !used[s+1] {
					used[s+1] = true
					out = append(out, position{s + 1, fret, n.Velocity})
					break
				}
			}
		}
	}
	return out, nil
}

// tempos rounds each bar's tempo to whole BPM, carrying the timing error into the next bar.
func tempos(bars []score.Bar) ([]int, error) {
	var out []int
	var target, written float64
	for _, b := range bars {
		if b.BPM == 0 {
			return nil, errors.New("float division by zero")
		}
		length := b.LengthQ().Float()
		duration := 60 * length / b.BPM
		target += duration
		remaining := target - written
		if !(remaining > 0) {
			remaining = duration
		}
		x := math.RoundToEven(60 * length / remaining)
		if math.IsNaN(x) || math.IsInf(x, 0) || math.Abs(x) > math.MaxInt32 {
			return nil, fmt.Errorf("bar %d: tempo out of range", b.Index)
		}
		bpm := max(1, int(x))
		out = append(out, bpm)
		written += 60 * length / float64(bpm)
	}
	return out, nil
}

// Write exports the score to path and returns what was written.
func Write(s *score.Score, path string) (*Report, error) {
	rep := &Report{Path: path, Warnings: []string{}}
	data, err := encode(s, rep)
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	return rep, os.WriteFile(path, data, 0o644)
}

func encode(s *score.Score, rep *Report) ([]byte, error) {
	tempoList, err := tempos(s.Bars)
	if err != nil {
		return nil, err
	}
	songTempo := 120
	if len(tempoList) > 0 {
		songTempo = tempoList[0]
	}

	var tracks []track
	channel := 0
	for k := range s.Tracks {
		t := &s.Tracks[k]
		tr := track{src: t, index: k, name: t.Name, tuning: staffTuning}
		switch t.Kind {
		case "tab":
			tr.tuning = t.Tuning
		case "drums":
			tr.tuning = drumTuning
		}
		if len(tr.tuning) > 7 {
			rep.Warnings = append(rep.Warnings, t.Name+": GP5 supports at most 7 strings; track skipped (it is in the .gp file)")
			continue
		}
		if t.Kind == "drums" {
			tr.drums, tr.channel, tr.effect, tr.instrument = true, 9, 9, 0
		} else {
			if channel == 9 || channel == 10 {
				channel = 11
			}
			tr.channel, tr.effect, tr.instrument = channel%16, (channel+1)%16, t.MidiProgram
			channel += 2
		}
		tracks = append(tracks, tr)
	}
	if len(s.Bars) == 0 || len(tracks) == 0 {
		return nil, errors.New("list index out of range")
	}

	w := &writer{}
	w.byteSizeString("FICHIER GUITAR PRO v5.10", 30)
	title, artist := "", ""
	if a, ok := s.Meta["artist"]; ok {
		artist = pyStr(a)
	}
	title = s.Title
	for _, v := range []string{title, "", artist, "", "", "", "", tabText, ""} {
		w.intByteSizeString(v)
	}
	w.i32(int64(len(notice)))
	for _, v := range notice {
		w.intByteSizeString(v)
	}
	w.i32(0) // lyrics track
	for range 5 {
		w.i32(1)
		w.i32(0)
	}
	w.i32(100) // RSE master volume
	w.i32(0)
	w.raw(make([]byte, 11)) // master equalizer
	for _, v := range []int64{210, 297, 10, 10, 15, 10, 100} {
		w.i32(v) // page size, margins, score size
	}
	w.raw([]byte{0xff, 0x01}) // header and footer flags
	for _, v := range []string{"%title%", "%subtitle%", "%artist%", "%album%", "Words by %words%",
		"Music by %music%", "Words & Music by %WORDSMUSIC%", "Copyright %copyright%",
		"All Rights Reserved - International Copyright Secured", "Page %N%/%P%"} {
		w.intByteSizeString(v)
	}
	w.intByteSizeString("Moderate")
	w.i32(int64(songTempo))
	w.raw([]byte{0, 0, 0, 0, 0, 0}) // hide tempo, key, octave

	instruments := make([]int, 64)
	claimed := make([]bool, 64)
	for i := range instruments {
		instruments[i] = 25
		if i%16 == 9 {
			instruments[i] = -1
		}
	}
	for _, tr := range tracks {
		for _, c := range []int{tr.channel, tr.effect} {
			if !claimed[c] {
				claimed[c] = true
				if tr.drums {
					instruments[c] = -1
				} else {
					instruments[c] = tr.instrument
				}
			}
		}
	}
	for _, inst := range instruments {
		w.i32(int64(inst))
		w.raw([]byte{13, 8, 0, 0, 0, 0, 0, 0}) // volume 104, balance 64, effects off
	}
	for range 19 {
		w.raw([]byte{0xff, 0xff}) // musical directions
	}
	w.i32(0) // master reverb
	w.i32(int64(len(s.Bars)))
	w.i32(int64(len(tracks)))

	for i, b := range s.Bars {
		var flags byte
		if i == 0 {
			flags = 0x01 | 0x02 | 0x40
		} else {
			w.raw([]byte{0})
			prev := s.Bars[i-1]
			if b.Numerator != prev.Numerator {
				flags |= 0x01
			}
			if b.Denominator != prev.Denominator {
				flags |= 0x02
			}
		}
		marker := b.Section != nil && *b.Section != ""
		if marker {
			flags |= 0x20
		}
		w.raw([]byte{flags})
		if flags&0x01 != 0 {
			if err := w.i8(b.Numerator); err != nil {
				return nil, err
			}
		}
		if flags&0x02 != 0 {
			if err := w.i8(b.Denominator); err != nil {
				return nil, err
			}
		}
		if marker {
			w.intByteSizeString(*b.Section)
			w.raw([]byte{0xff, 0, 0, 0})
		}
		if flags&0x40 != 0 {
			w.raw([]byte{0, 0})
		}
		if flags&0x03 != 0 {
			w.raw([]byte{2, 2, 2, 2})
		}
		w.raw([]byte{0, 0}) // no alternate ending, no triplet feel
	}

	for k, tr := range tracks {
		if k == 0 {
			w.raw([]byte{0})
		}
		flags := byte(0x08)
		if tr.drums {
			flags |= 0x01
		}
		w.raw([]byte{flags})
		w.byteSizeString(tr.name, 40)
		w.i32(int64(len(tr.tuning)))
		for i := range 7 {
			v := 0
			if i < len(tr.tuning) {
				v = tr.tuning[i]
			}
			w.i32(int64(v))
		}
		w.i32(1)
		w.i32(int64(tr.channel + 1))
		w.i32(int64(tr.effect + 1))
		w.i32(24) // frets
		w.i32(0)  // capo
		w.raw([]byte{0xff, 0, 0, 0, 0x43, 0, 0, 0, 0})
		if len(tr.tuning) == 0 {
			return nil, fmt.Errorf("writing track %d: list index out of range", k+1)
		}
		low := int64(0)
		if tr.tuning[len(tr.tuning)-1] < 35 {
			low = 12
		}
		w.i32(low)
		w.i32(low)
		w.i32(100)
		w.raw(make([]byte, 12))
		w.raw(bytes.Repeat([]byte{0xff}, 16)) // RSE instrument
		w.raw(make([]byte, 4))                // track equalizer
		w.intByteSizeString("")
		w.intByteSizeString("")
	}
	w.raw([]byte{0})

	layouts := make([][][]layout.Beat, len(tracks))
	for k, tr := range tracks {
		layouts[k] = layout.Track(tr.src, s.Bars)
	}
	lastBPM := songTempo
	for i := range s.Bars {
		bpm := tempoList[i]
		for k, tr := range tracks {
			beats := layouts[k][i]
			w.i32(int64(len(beats)))
			for j := range beats {
				lb := &beats[j]
				mix := j == 0 && tr.index == 0 && abs(bpm-lastBPM) >= 1
				if mix {
					lastBPM = bpm
				}
				if err := w.beat(tr, lb, mix, bpm, rep); err != nil {
					return nil, fmt.Errorf("writing track %d, measure %d: %w", k+1, i+1, err)
				}
			}
			w.i32(0)         // second voice
			w.raw([]byte{0}) // line break
		}
	}
	return w.buf.Bytes(), nil
}

func (w *writer) beat(tr track, lb *layout.Beat, mix bool, bpm int, rep *Report) error {
	var flags byte
	if lb.Dots != 0 {
		flags |= 0x01
	}
	if lb.Text != "" {
		flags |= 0x04
	}
	if mix {
		flags |= 0x10
	}
	if lb.Tuplet != nil {
		flags |= 0x20
	}
	if lb.Rest {
		flags |= 0x40
	}
	w.raw([]byte{flags})
	if lb.Rest {
		w.raw([]byte{0x02})
	}
	if err := w.i8(bits.Len(uint(lb.Duration)) - 3); err != nil {
		return err
	}
	if lb.Tuplet != nil {
		w.i32(int64(lb.Tuplet[0]))
	}
	if lb.Text != "" {
		w.intByteSizeString(lb.Text)
	}
	if mix {
		w.raw(bytes.Repeat([]byte{0xff}, 23)) // instrument, RSE instrument, volume..tremolo unchanged
		w.intByteSizeString("")
		w.i32(int64(bpm))
		w.raw([]byte{0, 1, 0, 0xff})
		w.intByteSizeString("")
		w.intByteSizeString("")
	}
	var placed []position
	if !lb.Rest {
		var err error
		if placed, err = positions(tr.src, lb); err != nil {
			return err
		}
		rep.NotesDropped += len(lb.Notes) - len(placed)
		rep.NotesWritten += len(placed)
	}
	sort.SliceStable(placed, func(i, j int) bool { return placed[i].str < placed[j].str })
	strings := 0
	for _, p := range placed {
		if p.str > 7 {
			return errors.New("negative shift count")
		}
		strings |= 1 << (7 - p.str)
	}
	if strings > 0xff {
		return errors.New("ubyte format requires 0 <= number <= 255")
	}
	w.raw([]byte{byte(strings)})
	for _, p := range placed {
		vel := min(127, max(15, p.velocity))
		flags := byte(0x20)
		if vel != defaultVelocity {
			flags |= 0x10
		}
		typ := byte(1)
		if lb.Tied {
			typ = 2
		}
		w.raw([]byte{flags, typ})
		if vel != defaultVelocity {
			w.raw([]byte{byte((vel-15)/16 + 1)})
		}
		value := p.value
		if lb.Tied {
			value = 0 // tied notes carry no fret of their own
		}
		if err := w.i8(value); err != nil {
			return err
		}
		w.raw([]byte{0})
	}
	w.raw([]byte{0, 0})
	return nil
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// pyStr is str(value) for the JSON scalars score meta can hold.
func pyStr(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case nil:
		return "None"
	case bool:
		if x {
			return "True"
		}
		return "False"
	default:
		return fmt.Sprint(x)
	}
}

type writer struct{ buf bytes.Buffer }

func (w *writer) raw(b []byte) { w.buf.Write(b) }

func (w *writer) i32(v int64) {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], uint32(int32(v)))
	w.buf.Write(b[:])
}

func (w *writer) i8(v int) error {
	if v < -128 || v > 127 {
		return errors.New("byte format requires -128 <= number <= 127")
	}
	w.buf.WriteByte(byte(int8(v)))
	return nil
}

func (w *writer) intByteSizeString(s string) {
	b := cp1252(s)
	if len(b) > 255 {
		b = b[:255]
	}
	w.i32(int64(len(b) + 1))
	w.buf.WriteByte(byte(len(b)))
	w.buf.Write(b)
}

func (w *writer) byteSizeString(s string, size int) {
	b := cp1252(s)
	if len(b) > size {
		b = b[:size]
	}
	w.buf.WriteByte(byte(len(b)))
	w.buf.Write(b)
	w.buf.Write(make([]byte, size-len(b)))
}

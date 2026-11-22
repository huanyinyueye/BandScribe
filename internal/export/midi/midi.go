// Package midi writes the BandScribe MIDI export (port of bandscribe/notation/export_midi.py),
// byte-for-byte equal to what mido 1.3 wrote for the same score.
package midi

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"

	"bandscribe/internal/rat"
	"bandscribe/internal/score"
)

const Ticks = 960

func ticks(q rat.Q) int { return int(q.Mul(rat.Int(Ticks)).RoundHalfEven()) }

type event struct {
	tick, order int
	data        []byte // message bytes without the delta time
	status      byte   // channel status byte, 0 for meta
}

// latin1 is mido's default meta-text charset. Characters outside it are written as UTF-8
// (mido raised UnicodeEncodeError and the MIDI export failed).
func latin1(s string) []byte {
	out := make([]byte, 0, len(s))
	for _, r := range s {
		if r > 0xff {
			return []byte(s)
		}
		out = append(out, byte(r))
	}
	return out
}

func varint(v int) []byte {
	if v < 0 {
		panic("midi: negative variable-length value")
	}
	buf := []byte{byte(v & 0x7f)}
	for v >>= 7; v > 0; v >>= 7 {
		buf = append([]byte{byte(v&0x7f | 0x80)}, buf...)
	}
	return buf
}

func meta(typ byte, data []byte) []byte {
	out := []byte{0xff, typ}
	out = append(out, varint(len(data))...)
	return append(out, data...)
}

// bpm2tempo is mido.bpm2tempo: int(round(60 * 1e6 / bpm)).
func bpm2tempo(bpm float64) int { return int(math.RoundToEven(60 * 1e6 / bpm)) }

func log2(den int) byte {
	return byte(math.Log2(float64(den)))
}

func writeTrack(buf *bytes.Buffer, events []event) {
	var data bytes.Buffer
	running := byte(0)
	prev := 0
	for _, e := range events {
		data.Write(varint(e.tick - prev))
		prev = e.tick
		if e.status == 0 {
			data.Write(e.data)
			running = 0
			continue
		}
		if e.status == running {
			data.Write(e.data[1:])
		} else {
			data.Write(e.data)
		}
		running = e.status
	}
	data.Write(varint(0))
	data.Write(meta(0x2f, nil))
	buf.WriteString("MTrk")
	binary.Write(buf, binary.BigEndian, uint32(data.Len()))
	buf.Write(data.Bytes())
}

func clamp7(v int, what string) byte {
	if v < 0 || v > 127 {
		panic(fmt.Sprintf("midi: %s %d out of range", what, v))
	}
	return byte(v)
}

// Bytes returns the MIDI file for the score.
func Bytes(s *score.Score) []byte {
	var buf bytes.Buffer
	buf.WriteString("MThd")
	binary.Write(&buf, binary.BigEndian, uint32(6))
	binary.Write(&buf, binary.BigEndian, [3]int16{1, int16(1 + len(s.Tracks)), Ticks})

	title := s.Title
	if title == "" {
		title = "BandScribe"
	}
	head := []event{{0, -1, meta(0x03, latin1(title)), 0}}
	var events []event
	var lastSig [2]int
	haveSig, haveBPM := false, false
	lastBPM := 0.0
	for _, b := range s.Bars {
		t := ticks(b.StartQ)
		sig := [2]int{b.Numerator, b.Denominator}
		if !haveSig || sig != lastSig {
			events = append(events, event{t, 0, meta(0x58, []byte{clamp7(sig[0], "numerator"), log2(sig[1]), 24, 8}), 0})
			lastSig, haveSig = sig, true
		}
		if !haveBPM || math.Abs(b.BPM-lastBPM) > 0.05 {
			tempo := bpm2tempo(b.BPM)
			events = append(events, event{t, 1, meta(0x51, []byte{byte(tempo >> 16), byte(tempo >> 8), byte(tempo)}), 0})
			lastBPM, haveBPM = b.BPM, true
		}
		if b.Section != nil && *b.Section != "" {
			events = append(events, event{t, 2, meta(0x06, latin1(*b.Section)), 0})
		}
	}
	sort.SliceStable(events, func(i, j int) bool {
		if events[i].tick != events[j].tick {
			return events[i].tick < events[j].tick
		}
		return events[i].order < events[j].order
	})
	writeTrack(&buf, append(head, events...))

	channel := 0
	for ti := range s.Tracks {
		track := &s.Tracks[ti]
		evs := []event{{0, -1, meta(0x03, latin1(track.Name)), 0}}
		var ch int
		if track.Kind == "drums" {
			ch = 9
		} else {
			if channel == 9 {
				channel++
			}
			ch = channel % 16
			channel++
			st := byte(0xc0 | ch)
			evs = append(evs, event{0, -1, []byte{st, clamp7(track.MidiProgram, "program")}, st})
		}
		var msgs []event
		for _, n := range track.Notes {
			at := s.Bars[n.Bar].StartQ.Add(n.PosQ)
			start := ticks(at)
			end := max(start+1, ticks(at.Add(n.DurQ)))
			vel := min(127, max(1, n.Velocity))
			on, off := byte(0x90|ch), byte(0x80|ch)
			p := clamp7(n.Pitch, "pitch")
			msgs = append(msgs, event{start, 1, []byte{on, p, byte(vel)}, on})
			msgs = append(msgs, event{end, 0, []byte{off, p, 0}, off})
		}
		sort.SliceStable(msgs, func(i, j int) bool {
			if msgs[i].tick != msgs[j].tick {
				return msgs[i].tick < msgs[j].tick
			}
			return msgs[i].order < msgs[j].order
		})
		writeTrack(&buf, append(evs, msgs...))
	}
	return buf.Bytes()
}

func Write(s *score.Score, path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, Bytes(s), 0o644)
}

// Package score reads score.json (bandscribe/notation/score_model.py, schema 1).
package score

import (
	"encoding/json"
	"fmt"
	"os"

	"bandscribe/internal/rat"
)

const SchemaVersion = 1

type Bar struct {
	Index       int
	Numerator   int
	Denominator int
	StartBeat   int
	StartQ      rat.Q
	BPM         float64
	StartTime   float64
	EndTime     float64
	Section     *string
	Confident   bool
	BeatTimes   []float64
}

// BeatQ is the length of one tracked beat in quarter notes.
func (b Bar) BeatQ() rat.Q   { return rat.New(4, int64(b.Denominator)) }
func (b Bar) LengthQ() rat.Q { return b.BeatQ().Mul(rat.Int(int64(b.Numerator))) }
func (b Bar) EndQ() rat.Q    { return b.StartQ.Add(b.LengthQ()) }

type Note struct {
	Bar      int
	PosQ     rat.Q
	DurQ     rat.Q
	Pitch    int
	Velocity int
	Conf     float64
	LowConf  bool
	Sources  []string
	String   *int // 1 = highest string
	Fret     *int
	Flags    []string
	Onset    float64
	Offset   float64
}

type Track struct {
	ID          string
	Name        string
	Family      string
	Kind        string // tab | staff | drums
	MidiProgram int
	TuningKey   *string
	Tuning      []int // open strings, string 1 (highest) first
	Notes       []Note
	Grid        map[[2]int]int // (bar, beat) -> subdivisions per beat
	Info        map[string]any
}

// TuningName is info["tuning_name"], or "" when absent.
func (t Track) TuningName() string {
	if s, ok := t.Info["tuning_name"].(string); ok {
		return s
	}
	return ""
}

type Score struct {
	Title  string
	Bars   []Bar
	Tracks []Track
	Key    map[string]any // nil when the score has no key
	Meta   map[string]any
}

// KeyFifths returns key["fifths"] when present and not null.
func (s Score) KeyFifths() (int, bool) {
	if s.Key == nil {
		return 0, false
	}
	f, ok := s.Key["fifths"].(float64)
	return int(f), ok
}

// KeyMode returns key["mode"], or "" when absent.
func (s Score) KeyMode() string {
	if s.Key == nil {
		return ""
	}
	m, _ := s.Key["mode"].(string)
	return m
}

type rawBar struct {
	Index       int       `json:"index"`
	Numerator   int       `json:"numerator"`
	Denominator int       `json:"denominator"`
	StartBeat   int       `json:"start_beat"`
	StartQ      string    `json:"start_q"`
	BPM         float64   `json:"bpm"`
	StartTime   float64   `json:"start_time"`
	EndTime     float64   `json:"end_time"`
	Section     *string   `json:"section"`
	Confident   *bool     `json:"confident"`
	BeatTimes   []float64 `json:"beat_times"`
}

type rawNote struct {
	Bar      int      `json:"bar"`
	PosQ     string   `json:"pos_q"`
	DurQ     string   `json:"dur_q"`
	Pitch    int      `json:"pitch"`
	Velocity int      `json:"velocity"`
	Conf     float64  `json:"conf"`
	LowConf  bool     `json:"low_conf"`
	Sources  []string `json:"sources"`
	String   *int     `json:"string"`
	Fret     *int     `json:"fret"`
	Flags    []string `json:"flags"`
	Onset    float64  `json:"onset"`
	Offset   float64  `json:"offset"`
}

type rawTrack struct {
	ID          string         `json:"id"`
	Name        string         `json:"name"`
	Family      string         `json:"family"`
	Kind        string         `json:"kind"`
	MidiProgram int            `json:"midi_program"`
	TuningKey   *string        `json:"tuning_key"`
	Tuning      []int          `json:"tuning"`
	Info        map[string]any `json:"info"`
	Grid        [][3]int       `json:"grid"`
	Notes       []rawNote      `json:"notes"`
}

type rawScore struct {
	SchemaVersion *int           `json:"schema_version"`
	Title         string         `json:"title"`
	Key           map[string]any `json:"key"`
	Meta          map[string]any `json:"meta"`
	Bars          []rawBar       `json:"bars"`
	Tracks        []rawTrack     `json:"tracks"`
}

var kinds = map[string]bool{"tab": true, "staff": true, "drums": true}

// Parse decodes score.json bytes.
func Parse(data []byte) (*Score, error) {
	var r rawScore
	if err := json.Unmarshal(data, &r); err != nil {
		return nil, err
	}
	if r.SchemaVersion == nil || *r.SchemaVersion != SchemaVersion {
		return nil, fmt.Errorf("unsupported score schema version")
	}
	s := &Score{Title: r.Title, Key: r.Key, Meta: r.Meta}
	if s.Meta == nil {
		s.Meta = map[string]any{}
	}
	for _, b := range r.Bars {
		q, err := rat.Parse(b.StartQ)
		if err != nil {
			return nil, fmt.Errorf("bar %d start_q: %w", b.Index, err)
		}
		if b.Numerator <= 0 || b.Denominator <= 0 {
			return nil, fmt.Errorf("bar %d: bad time signature %d/%d", b.Index, b.Numerator, b.Denominator)
		}
		confident := true
		if b.Confident != nil {
			confident = *b.Confident
		}
		s.Bars = append(s.Bars, Bar{b.Index, b.Numerator, b.Denominator, b.StartBeat, q, b.BPM,
			b.StartTime, b.EndTime, b.Section, confident, b.BeatTimes})
	}
	for _, t := range r.Tracks {
		if !kinds[t.Kind] {
			return nil, fmt.Errorf("unknown track kind %q", t.Kind)
		}
		tr := Track{ID: t.ID, Name: t.Name, Family: t.Family, Kind: t.Kind,
			MidiProgram: t.MidiProgram, TuningKey: t.TuningKey, Tuning: t.Tuning,
			Grid: map[[2]int]int{}, Info: t.Info}
		if tr.Info == nil {
			tr.Info = map[string]any{}
		}
		for _, g := range t.Grid {
			tr.Grid[[2]int{g[0], g[1]}] = g[2]
		}
		for i, n := range t.Notes {
			pos, err := rat.Parse(n.PosQ)
			if err != nil {
				return nil, fmt.Errorf("track %s note %d pos_q: %w", t.ID, i, err)
			}
			dur, err := rat.Parse(n.DurQ)
			if err != nil {
				return nil, fmt.Errorf("track %s note %d dur_q: %w", t.ID, i, err)
			}
			if n.Bar < 0 || n.Bar >= len(s.Bars) {
				return nil, fmt.Errorf("track %s note %d: bar %d out of range", t.ID, i, n.Bar)
			}
			tr.Notes = append(tr.Notes, Note{n.Bar, pos, dur, n.Pitch, n.Velocity, n.Conf,
				n.LowConf, n.Sources, n.String, n.Fret, n.Flags, n.Onset, n.Offset})
		}
		s.Tracks = append(s.Tracks, tr)
	}
	return s, nil
}

func Load(path string) (*Score, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return Parse(data)
}

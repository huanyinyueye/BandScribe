// Command bandscribe-export writes the score exports (.gp, .gp5, MIDI, MusicXML) from score.json
// and prints a JSON report on stdout.
//
//	bandscribe-export --score score.json [--gp out.gp] [--gp5 out.gp5] [--midi out.mid] [--musicxml out.musicxml]
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"bandscribe/internal/export/gp"
	"bandscribe/internal/export/gp5"
	"bandscribe/internal/export/midi"
	"bandscribe/internal/export/musicxml"
	"bandscribe/internal/score"
)

type result struct {
	Format string `json:"format"`
	Path   string `json:"path"`
	OK     bool   `json:"ok"`
	Error  string `json:"error,omitempty"`
	Report any    `json:"report,omitempty"`
}

func run(format, path string, fn func() (any, error)) (r result) {
	r = result{Format: format, Path: path}
	defer func() {
		if p := recover(); p != nil {
			r.OK, r.Error = false, fmt.Sprint(p)
		}
	}()
	rep, err := fn()
	if err != nil {
		r.Error = err.Error()
		return r
	}
	r.OK, r.Report = true, rep
	return r
}

func main() {
	scorePath := flag.String("score", "", "score.json")
	gpPath := flag.String("gp", "", "Guitar Pro 7/8 output")
	gp5Path := flag.String("gp5", "", "Guitar Pro 5 output")
	midiPath := flag.String("midi", "", "MIDI output")
	xmlPath := flag.String("musicxml", "", "MusicXML output")
	flag.Parse()
	if *scorePath == "" {
		fmt.Fprintln(os.Stderr, "--score is required")
		os.Exit(2)
	}
	s, err := score.Load(*scorePath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	var results []result
	if *gpPath != "" {
		results = append(results, run("gp", *gpPath, func() (any, error) { return gp.Write(s, *gpPath) }))
	}
	if *gp5Path != "" {
		results = append(results, run("gp5", *gp5Path, func() (any, error) { return gp5.Write(s, *gp5Path) }))
	}
	if *midiPath != "" {
		results = append(results, run("midi", *midiPath, func() (any, error) { return nil, midi.Write(s, *midiPath) }))
	}
	if *xmlPath != "" {
		results = append(results, run("musicxml", *xmlPath, func() (any, error) { return nil, musicxml.Write(s, *xmlPath) }))
	}
	enc := json.NewEncoder(os.Stdout)
	if err := enc.Encode(map[string]any{"results": results}); err != nil {
		os.Exit(1)
	}
}

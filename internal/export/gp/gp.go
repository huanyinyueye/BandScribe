// Package gp writes Guitar Pro 7/8 `.gp` files exactly as bandscribe/notation/export_gp.py does
// with alphaTab 1.9.0's Gp7Exporter, then verifies the file the way alphaTab reads it back.
package gp

import (
	"archive/zip"
	"bytes"
	"fmt"
	"os"
	"path/filepath"

	"bandscribe/internal/score"
)

// Report is GpExportReport without the summary.
type Report struct {
	Path         string   `json:"path"`
	Bytes        int      `json:"bytes"`
	AlphaTab     string   `json:"alphatab"`
	Verified     bool     `json:"verified"`
	Mismatches   []string `json:"mismatches"`
	NotesWritten int      `json:"notes_written"`
}

// Write exports s to path and reports whether the file reads back as sent. An unverified file is
// still written; callers decide whether to use it.
func Write(s *score.Score, path string) (*Report, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	g, err := build(s)
	if err != nil {
		return nil, err
	}
	root, err := writeGPIF(g)
	if err != nil {
		return nil, err
	}
	gpif := serialize(root)
	if _, err := readGPIF(gpif); err != nil { // export_gp.mjs reads the bytes back at once
		return nil, fmt.Errorf("export_gp failed: %w", err)
	}
	files := [][2]any{
		{"VERSION", []byte("7.0")},
		{"Content/", []byte{}},
		{"Content/BinaryStylesheet", binaryStylesheet},
		{"Content/PartConfiguration", partConfiguration(g)},
		{"Content/LayoutConfiguration", layoutConfiguration(g)},
		{"Content/score.gpif", gpif},
	}
	fixed, err := restorePitchedStaves(root, g)
	if err != nil {
		return nil, err
	}
	if fixed != nil {
		gpif = fixed
		files[len(files)-1][1] = fixed
	}
	data, err := zipFiles(files)
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
		return nil, err
	}
	if err := os.WriteFile(abs, data, 0o644); err != nil {
		return nil, err
	}
	rs, err := readGPIF(gpif)
	if err != nil {
		return nil, fmt.Errorf("reading %s back: %w", abs, err)
	}
	issues, err := verify(g, rs)
	if err != nil {
		return nil, err
	}
	notes := 0
	for _, t := range rs.tracks {
		notes += len(t.notes)
	}
	if issues == nil {
		issues = []string{}
	}
	return &Report{abs, len(data), alphaTabVersion, len(issues) == 0, issues, notes}, nil
}

func zipFiles(files [][2]any) ([]byte, error) {
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	for _, f := range files {
		w, err := zw.CreateHeader(&zip.FileHeader{Name: f[0].(string), Method: zip.Deflate})
		if err != nil {
			return nil, err
		}
		if _, err := w.Write(f[1].([]byte)); err != nil {
			return nil, err
		}
	}
	if err := zw.Close(); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

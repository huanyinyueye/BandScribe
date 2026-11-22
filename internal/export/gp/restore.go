package gp

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

// restorePitchedStaves is _restore_pitched_staves of bandscribe/notation/export_gp.py: alphaTab
// gives piano and vocal staves a synthetic six-string fingering, which large tied chords can
// turn into wrong pitches. Drop the tuning and store each note's pitch as octave and tone.
//
// Python round-trips the file through ElementTree, so the result is what ElementTree writes:
// comments dropped, CDATA written as escaped text, line ends normalised. It returns the new
// score.gpif, or nil when the score has no staff tracks (the file stays as alphaTab wrote it).
func restorePitchedStaves(root *node, g *gScore) ([]byte, error) {
	var staff []int
	for i, tr := range g.tracks {
		if tr.sent.Kind == "staff" {
			staff = append(staff, i)
		}
	}
	if len(staff) == 0 {
		return nil, nil
	}
	if err := checkWellFormed(root); err != nil {
		return nil, fmt.Errorf("ParseError: %w", err)
	}
	et := etTree(root)
	tracks := elements(et.child("Tracks"))
	for _, index := range staff {
		for _, st := range elements(tracks[index].child("Staves")) {
			if st.name != "Staff" {
				continue
			}
			props := st.child("Properties")
			for _, p := range elements(props) {
				if v, _ := p.attr("name"); v == "Tuning" {
					props.remove(p)
				}
			}
		}
	}
	lookup := func(name string) map[string]*node {
		out := map[string]*node{}
		for _, el := range elements(et.child(name)) {
			id, _ := el.attr("id")
			out[id] = el
		}
		return out
	}
	bars, voices, beats, notes := lookup("Bars"), lookup("Voices"), lookup("Beats"), lookup("Notes")
	ids := map[string]bool{}
	for _, mb := range elements(et.child("MasterBars")) {
		barIDs := strings.Fields(childText(mb, "Bars"))
		for _, index := range staff {
			for _, vid := range strings.Fields(childText(bars[barIDs[index]], "Voices")) {
				v, ok := voices[vid]
				if !ok {
					continue
				}
				for _, bid := range strings.Fields(childText(v, "Beats")) {
					for _, nid := range strings.Fields(childText(beats[bid], "Notes")) {
						ids[nid] = true
					}
				}
			}
		}
	}
	sorted := make([]string, 0, len(ids))
	for id := range ids {
		sorted = append(sorted, id)
	}
	sort.Strings(sorted)
	for _, id := range sorted {
		props := notes[id].child("Properties")
		var midi *node
		for _, p := range elements(props) {
			if v, _ := p.attr("name"); v == "Midi" {
				midi = p.child("Number")
				break
			}
		}
		if midi == nil || midi.text == "" {
			return nil, fmt.Errorf("pitched GP note %s has no MIDI pitch", id)
		}
		pitch, err := strconv.Atoi(strings.TrimSpace(midi.text))
		if err != nil {
			return nil, fmt.Errorf("ValueError: invalid literal for int() with base 10: '%s'", midi.text)
		}
		for _, p := range elements(props) {
			switch v, _ := p.attr("name"); v {
			case "String", "Fret", "Octave", "Tone":
				props.remove(p)
			}
		}
		octave := int(math.Floor(float64(pitch) / 12))
		props.add("Property").setAttr("name", "Octave").add("Number").setText(strconv.Itoa(octave))
		props.add("Property").setAttr("name", "Tone").add("Step").setText(strconv.Itoa(pitch - 12*octave))
	}
	var b strings.Builder
	b.WriteString("<?xml version='1.0' encoding='utf-8'?>\n")
	etWrite(&b, et)
	return []byte(b.String()), nil
}

func elements(n *node) []*node {
	if n == nil {
		return nil
	}
	out := make([]*node, 0, len(n.children))
	for _, c := range n.children {
		if !c.comment {
			out = append(out, c)
		}
	}
	return out
}

func childText(n *node, name string) string {
	if c := n.child(name); c != nil {
		return c.text
	}
	return ""
}

// etTree is what ElementTree.fromstring builds from the serialized tree: no comments, CDATA as
// plain text (an empty text is None), "\r\n" and "\r" read as "\n", whitespace in attribute
// values read as spaces.
func etTree(n *node) *node {
	lines := strings.NewReplacer("\r\n", "\n", "\r", "\n")
	space := strings.NewReplacer("\r\n", " ", "\r", " ", "\n", " ", "\t", " ")
	out := &node{name: n.name}
	for _, a := range n.attrs {
		out.attrs = append(out.attrs, [2]string{a[0], space.Replace(a[1])})
	}
	if t := lines.Replace(n.text); t != "" {
		out.hasText, out.text = true, t
	}
	for _, c := range n.children {
		if !c.comment {
			out.children = append(out.children, etTree(c))
		}
	}
	return out
}

// etWrite is ElementTree's XML serialization (short empty elements, UTF-8).
func etWrite(b *strings.Builder, n *node) {
	b.WriteString("<" + n.name)
	for _, a := range n.attrs {
		b.WriteString(" " + a[0] + `="`)
		b.WriteString(etAttrEscaper.Replace(a[1]))
		b.WriteString(`"`)
	}
	if n.text == "" && len(n.children) == 0 {
		b.WriteString(" />")
		return
	}
	b.WriteString(">")
	b.WriteString(etTextEscaper.Replace(n.text))
	for _, c := range n.children {
		etWrite(b, c)
	}
	b.WriteString("</" + n.name + ">")
}

var etTextEscaper = strings.NewReplacer("&", "&amp;", "<", "&lt;", ">", "&gt;")
var etAttrEscaper = strings.NewReplacer("&", "&amp;", "<", "&lt;", ">", "&gt;", `"`, "&quot;",
	"\r", "&#13;", "\n", "&#10;", "\t", "&#09;")

// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy
// of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// Translated from alphaTab 1.9.0 (https://github.com/CoderLine/alphaTab), Copyright CoderLine:
// XmlParser, which reads score.gpif when alphaTab verifies the written file.

package gp

import (
	"fmt"
	"strings"
	"unicode/utf8"
)

type xmlParser struct {
	xml  string
	pos  int
	open []*node
}

func isXMLSpace(c byte) bool { return c == ' ' || c == '\n' || c == '\r' || c == '\t' }

func isNameEnd(c byte) bool {
	return isXMLSpace(c) || c == '>' || c == '/' || c == '=' || c == '<'
}

// parseXML is XmlParser.parse: element names, attributes, character data with leading and
// trailing whitespace removed (CDATA kept as is); comments and processing instructions skipped.
func parseXML(xml string) (*node, error) {
	p := &xmlParser{xml: xml}
	doc := &node{name: "#document"}
	if strings.HasPrefix(xml, "\uFEFF") {
		p.pos = len("\uFEFF")
	}
	current := doc
	var err error
	for p.pos < len(xml) && err == nil {
		if xml[p.pos] != '<' {
			p.characterData(current, doc)
			continue
		}
		next := byte(0)
		if p.pos+1 < len(xml) {
			next = xml[p.pos+1]
		}
		switch next {
		case '/':
			current, err = p.endTag(current, doc)
		case '?':
			p.pos, err = p.indexAfter("?>", p.pos+2, "processing instruction")
		case '!':
			err = p.declaration(current, doc)
		default:
			current, err = p.startTag(current)
		}
	}
	if err != nil {
		return nil, err
	}
	if current != doc {
		return nil, fmt.Errorf("XmlError: Unexpected end of document, missing </%s>", current.name)
	}
	return doc, nil
}

func (p *xmlParser) characterData(current, doc *node) {
	first, last := -1, -1
	ref := false
	i := p.pos
	for ; i < len(p.xml) && p.xml[i] != '<'; i++ {
		if !isXMLSpace(p.xml[i]) {
			if first == -1 {
				first = i
			}
			last = i
			if p.xml[i] == '&' {
				ref = true
			}
		}
	}
	p.pos = i
	if first == -1 || current == doc {
		return
	}
	text := p.xml[first : last+1]
	if ref {
		text = decodeXML(text)
	}
	appendText(current, text)
}

func appendText(n *node, text string) {
	n.text += text
	n.hasText = true
}

func (p *xmlParser) readName(kind string) (string, error) {
	start := p.pos
	for p.pos < len(p.xml) && !isNameEnd(p.xml[p.pos]) {
		p.pos++
	}
	if p.pos == start {
		return "", fmt.Errorf("XmlError: Expected %s at %d", kind, start)
	}
	return p.xml[start:p.pos], nil
}

func (p *xmlParser) skipSpace() {
	for p.pos < len(p.xml) && isXMLSpace(p.xml[p.pos]) {
		p.pos++
	}
}

func (p *xmlParser) expect(pos int, c byte, what string) error {
	if pos >= len(p.xml) || p.xml[pos] != c {
		return fmt.Errorf("XmlError: Expected %s at %d", what, pos)
	}
	return nil
}

func (p *xmlParser) startTag(parent *node) (*node, error) {
	p.pos++
	name, err := p.readName("element name")
	if err != nil {
		return nil, err
	}
	el := &node{name: name}
	parent.children = append(parent.children, el)
	for {
		p.skipSpace()
		if p.pos >= len(p.xml) {
			break
		}
		switch p.xml[p.pos] {
		case '>':
			p.pos++
			p.open = append(p.open, parent)
			return el, nil
		case '/':
			if err := p.expect(p.pos+1, '>', "'>' after '/'"); err != nil {
				return nil, err
			}
			p.pos += 2
			return parent, nil
		}
		attr, err := p.readName("attribute name")
		if err != nil {
			return nil, err
		}
		p.skipSpace()
		if err := p.expect(p.pos, '=', "'=' after attribute "+attr); err != nil {
			return nil, err
		}
		p.pos++
		p.skipSpace()
		value, err := p.attributeValue(attr)
		if err != nil {
			return nil, err
		}
		el.setAttr(attr, value)
	}
	return nil, fmt.Errorf("XmlError: Unexpected end of document in tag <%s>", el.name)
}

func (p *xmlParser) attributeValue(name string) (string, error) {
	if p.pos >= len(p.xml) || (p.xml[p.pos] != '"' && p.xml[p.pos] != '\'') {
		return "", fmt.Errorf("XmlError: Expected quoted value for attribute %s", name)
	}
	quote := p.xml[p.pos]
	start := p.pos + 1
	ref := false
	for i := start; i < len(p.xml); i++ {
		switch p.xml[i] {
		case quote:
			p.pos = i + 1
			if ref {
				return decodeXML(p.xml[start:i]), nil
			}
			return p.xml[start:i], nil
		case '&':
			ref = true
		}
	}
	return "", fmt.Errorf("XmlError: Unterminated value for attribute %s", name)
}

func (p *xmlParser) endTag(current, doc *node) (*node, error) {
	start := p.pos + 2
	i := start
	for i < len(p.xml) && !isNameEnd(p.xml[i]) {
		i++
	}
	if current == doc || p.xml[start:i] != current.name {
		return nil, fmt.Errorf("XmlError: Unexpected </%s>", p.xml[start:i])
	}
	p.pos = i
	p.skipSpace()
	if err := p.expect(p.pos, '>', "'>' to end </"+current.name+">"); err != nil {
		return nil, err
	}
	p.pos++
	parent := p.open[len(p.open)-1]
	p.open = p.open[:len(p.open)-1]
	return parent, nil
}

func (p *xmlParser) indexAfter(term string, from int, what string) (int, error) {
	if from > len(p.xml) {
		from = len(p.xml)
	}
	i := strings.Index(p.xml[from:], term)
	if i < 0 {
		return 0, fmt.Errorf("XmlError: Unterminated %s", what)
	}
	return from + i + len(term), nil
}

func (p *xmlParser) declaration(current, doc *node) error {
	rest := p.xml[p.pos:]
	switch {
	case strings.HasPrefix(rest, "<!--"):
		end, err := p.indexAfter("-->", p.pos+4, "comment")
		p.pos = end
		return err
	case strings.HasPrefix(rest, "<![CDATA["):
		start := p.pos + 9
		end, err := p.indexAfter("]]>", start, "CDATA section")
		if err != nil {
			return err
		}
		p.pos = end
		if current != doc {
			appendText(current, p.xml[start:end-3])
			current.cdata = true
		}
		return nil
	case len(rest) >= 9 && strings.EqualFold(rest[:9], "<!DOCTYPE"):
		p.pos += 9
		p.skipSpace()
		depth := 0
		for ; p.pos < len(p.xml); p.pos++ {
			c := p.xml[p.pos]
			if c == '[' {
				depth++
			} else if c == ']' {
				depth--
			} else if c == '>' && depth <= 0 {
				break
			}
		}
		if p.pos >= len(p.xml) {
			return fmt.Errorf("XmlError: Unterminated document type declaration")
		}
		p.pos++
		return nil
	}
	return fmt.Errorf("XmlError: Unsupported markup declaration at %d", p.pos)
}

// decodeXML is XmlParser._decode: the five predefined entities and character references; an
// unknown reference stays as written.
func decodeXML(s string) string {
	var b strings.Builder
	seg := 0
	for i := 0; i < len(s); {
		if s[i] != '&' {
			i++
			continue
		}
		start := i + 1
		semi := start
		maxEnd := min(len(s), start+10)
		for semi < maxEnd && s[semi] != ';' {
			semi++
		}
		resolved, ok := "", false
		if semi < maxEnd {
			resolved, ok = resolveReference(s[start:semi])
		}
		if !ok {
			i++
			continue
		}
		b.WriteString(s[seg:i])
		b.WriteString(resolved)
		i = semi + 1
		seg = i
	}
	b.WriteString(s[seg:])
	return b.String()
}

func resolveReference(name string) (string, bool) {
	switch name {
	case "lt":
		return "<", true
	case "gt":
		return ">", true
	case "amp":
		return "&", true
	case "quot":
		return "\"", true
	case "apos":
		return "'", true
	}
	if len(name) < 2 || name[0] != '#' {
		return "", false
	}
	digits, base := name[1:], 10
	if name[1] == 'x' {
		digits, base = name[2:], 16
	}
	cp, ok := jsParseInt(digits, base)
	if !ok || cp < 0 || cp > 0x10FFFF {
		return "", false
	}
	if cp >= 0xD800 && cp <= 0xDFFF {
		return string(utf8.RuneError), true
	}
	return string(rune(cp)), true
}

// jsParseInt is Number.parseInt(text, base) for the integer prefix of text; ok is false for NaN.
func jsParseInt(text string, base int) (int64, bool) {
	s := strings.TrimLeft(text, " \t\n\r\v\f\u00a0\ufeff")
	neg := false
	if s != "" && (s[0] == '+' || s[0] == '-') {
		neg = s[0] == '-'
		s = s[1:]
	}
	if base == 16 && len(s) >= 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X') {
		s = s[2:]
	}
	var v int64
	n := 0
	for ; n < len(s); n++ {
		c := s[n]
		var d int
		switch {
		case c >= '0' && c <= '9':
			d = int(c - '0')
		case c >= 'a' && c <= 'z':
			d = int(c-'a') + 10
		case c >= 'A' && c <= 'Z':
			d = int(c-'A') + 10
		default:
			d = base
		}
		if d >= base {
			break
		}
		if v < 1<<53 {
			v = v*int64(base) + int64(d)
		}
	}
	if n == 0 {
		return 0, false
	}
	if neg {
		v = -v
	}
	return v, true
}

// parseIntSafe is GpifParser._parseIntSafe.
func parseIntSafe(text string, fallback float64) float64 {
	if text == "" {
		return fallback
	}
	v, ok := jsParseInt(text, 10)
	if !ok {
		return fallback
	}
	return float64(v)
}

// splitSafe is GpifParser._splitSafe.
func splitSafe(text string) []string {
	var out []string
	for _, part := range strings.Split(text, " ") {
		if t := strings.Trim(part, " \t\n\r\v\f\u00a0\ufeff\u2028\u2029"); t != "" {
			out = append(out, t)
		}
	}
	return out
}

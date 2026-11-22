// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy
// of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// Translated from alphaTab 1.9.0 (https://github.com/CoderLine/alphaTab), Copyright CoderLine.

package gp

import (
	"fmt"
	"strings"
	"unicode/utf8"
)

// node is alphaTab's XmlNode as far as the GPIF writer uses it: elements and comments.
type node struct {
	name     string
	attrs    [][2]string
	hasText  bool
	text     string
	cdata    bool
	comment  bool
	children []*node
}

func (n *node) add(name string) *node {
	c := &node{name: name}
	n.children = append(n.children, c)
	return c
}

func (n *node) setText(s string) *node {
	n.hasText, n.text, n.cdata, n.children = true, s, false, nil
	return n
}

func (n *node) setCData(s string) *node {
	n.setText(s)
	n.cdata = true
	return n
}

func (n *node) setAttr(name, value string) *node {
	for i := range n.attrs {
		if n.attrs[i][0] == name {
			n.attrs[i][1] = value
			return n
		}
	}
	n.attrs = append(n.attrs, [2]string{name, value})
	return n
}

func (n *node) attr(name string) (string, bool) {
	for _, a := range n.attrs {
		if a[0] == name {
			return a[1], true
		}
	}
	return "", false
}

func (n *node) child(name string) *node {
	for _, c := range n.children {
		if !c.comment && c.name == name {
			return c
		}
	}
	return nil
}

// innerText is the element text, or the concatenated text of its children.
func (n *node) innerText() string {
	if len(n.children) == 0 {
		return n.text
	}
	var b strings.Builder
	b.WriteString(n.text)
	for _, c := range n.children {
		if !c.comment {
			b.WriteString(c.innerText())
		}
	}
	return b.String()
}

func (n *node) remove(c *node) {
	for i, x := range n.children {
		if x == c {
			n.children = append(n.children[:i], n.children[i+1:]...)
			return
		}
	}
}

// serialize is XmlDocument.toFormattedString("", true).
func serialize(root *node) []byte {
	var b strings.Builder
	b.WriteString(`<?xml version="1.0" encoding="utf-8"?>`)
	writeNode(&b, root)
	return []byte(b.String())
}

func writeNode(b *strings.Builder, n *node) {
	if n.comment {
		b.WriteString("<!-- " + n.text + " -->")
		return
	}
	b.WriteString("<" + n.name)
	for _, a := range n.attrs {
		b.WriteString(" " + a[0] + `="`)
		escape(b, a[1], true)
		b.WriteString(`"`)
	}
	switch {
	case len(n.children) > 0:
		b.WriteString(">")
		for _, c := range n.children {
			writeNode(b, c)
		}
		b.WriteString("</" + n.name + ">")
	case !n.hasText:
		b.WriteString("/>")
	default:
		b.WriteString(">")
		if n.cdata {
			b.WriteString("<![CDATA[" + n.text + "]]>")
		} else {
			escape(b, n.text, false)
		}
		b.WriteString("</" + n.name + ">")
	}
}

func escape(b *strings.Builder, s string, attr bool) {
	for _, c := range s {
		switch {
		case c == '<':
			b.WriteString("&lt;")
		case c == '&':
			b.WriteString("&amp;")
		case c == '>' && attr:
			b.WriteString("&gt;")
		case c == '\'' && attr:
			b.WriteString("&apos;")
		case c == '"' && attr:
			b.WriteString("&quot;")
		default:
			b.WriteRune(c)
		}
	}
}

// checkWellFormed reports what makes an XML 1.0 parser (ElementTree) reject the document:
// characters outside the XML Char production and "]]>" inside a CDATA section.
func checkWellFormed(n *node) error {
	check := func(s string) error {
		for i, r := range s {
			if r == utf8.RuneError {
				continue
			}
			if !(r == 0x9 || r == 0xA || r == 0xD || (r >= 0x20 && r <= 0xD7FF) ||
				(r >= 0xE000 && r <= 0xFFFD) || r >= 0x10000) {
				return fmt.Errorf("not well-formed (invalid token): character %U at %d", r, i)
			}
		}
		return nil
	}
	if n.cdata && strings.Contains(n.text, "]]>") {
		return fmt.Errorf("not well-formed: CDATA contains ']]>'")
	}
	if err := check(n.text); err != nil {
		return err
	}
	for _, a := range n.attrs {
		if err := check(a[1]); err != nil {
			return err
		}
	}
	for _, c := range n.children {
		if err := checkWellFormed(c); err != nil {
			return err
		}
	}
	return nil
}

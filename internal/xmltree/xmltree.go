// Package xmltree is a minimal element tree that serialises exactly like Python 3.11's
// xml.etree.ElementTree (write with short empty elements, and ET.indent).
package xmltree

import (
	"strings"
)

type Element struct {
	Tag      string
	Attrs    [][2]string
	Text     *string
	Tail     *string
	Children []*Element
}

func New(tag string, attrs ...string) *Element {
	e := &Element{Tag: tag}
	for i := 0; i+1 < len(attrs); i += 2 {
		e.Attrs = append(e.Attrs, [2]string{attrs[i], attrs[i+1]})
	}
	return e
}

// Sub appends a child, like ET.SubElement; text is set unless nil.
func (e *Element) Sub(tag string, text *string, attrs ...string) *Element {
	c := New(tag, attrs...)
	c.Text = text
	e.Children = append(e.Children, c)
	return c
}

func Str(s string) *string { return &s }

var cdata = strings.NewReplacer("&", "&amp;", "<", "&lt;", ">", "&gt;")
var attrib = strings.NewReplacer("&", "&amp;", "<", "&lt;", ">", "&gt;", `"`, "&quot;",
	"\r", "&#13;", "\n", "&#10;", "\t", "&#09;")

func (e *Element) write(b *strings.Builder) {
	b.WriteString("<" + e.Tag)
	for _, a := range e.Attrs {
		b.WriteString(" " + a[0] + `="` + attrib.Replace(a[1]) + `"`)
	}
	if (e.Text != nil && *e.Text != "") || len(e.Children) > 0 {
		b.WriteString(">")
		if e.Text != nil {
			b.WriteString(cdata.Replace(*e.Text))
		}
		for _, c := range e.Children {
			c.write(b)
		}
		b.WriteString("</" + e.Tag + ">")
	} else {
		b.WriteString(" />")
	}
	if e.Tail != nil {
		b.WriteString(cdata.Replace(*e.Tail))
	}
}

func (e *Element) String() string {
	var b strings.Builder
	e.write(&b)
	return b.String()
}

// Indent is ET.indent(tree, space="  ").
func Indent(root *Element) {
	if len(root.Children) == 0 {
		return
	}
	indentations := []string{"\n"}
	var rec func(e *Element, level int)
	rec = func(e *Element, level int) {
		child := level + 1
		if child >= len(indentations) {
			indentations = append(indentations, indentations[level]+"  ")
		}
		ci := indentations[child]
		if e.Text == nil || strings.TrimSpace(*e.Text) == "" {
			e.Text = Str(ci)
		}
		var last *Element
		for _, c := range e.Children {
			if len(c.Children) > 0 {
				rec(c, child)
			}
			if c.Tail == nil || strings.TrimSpace(*c.Tail) == "" {
				c.Tail = Str(ci)
			}
			last = c
		}
		if strings.TrimSpace(*last.Tail) == "" {
			last.Tail = Str(indentations[level])
		}
	}
	rec(root, 0)
}

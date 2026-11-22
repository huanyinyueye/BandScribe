// Package layout turns a track's notes into per-bar sequences of notated beats.
// It is a line-by-line port of bandscribe/notation/layout.py; every exporter uses it.
package layout

import (
	"sort"

	"bandscribe/internal/rat"
	"bandscribe/internal/score"
)

const LowConfText = "?"

type Beat struct {
	Duration int // 1 whole, 2 half, 4 quarter, 8, 16, 32, 64
	Dots     int
	Tuplet   *[2]int // {3, 2} = three in the time of two
	Rest     bool
	Tied     bool // every note continues the previous beat's notes
	Notes    []*score.Note
	Text     string // "" = none
	LengthQ  rat.Q
}

// DefaultGrid is quantize.default_grid.
func DefaultGrid(denominator int) int {
	if denominator >= 8 {
		return 2
	}
	return 4
}

func valueLength(duration, dots int, tuplet *[2]int) rat.Q {
	length := rat.New(4, int64(duration))
	if dots != 0 {
		length = length.Mul(rat.New(3, 2))
	}
	if tuplet != nil {
		length = length.Mul(rat.New(int64(tuplet[1]), int64(tuplet[0])))
	}
	return length
}

type value struct{ duration, dots int }

func splitCells(k, offset int, bar score.Bar) []value {
	den := bar.Denominator
	compound := den == 8 && bar.Numerator%3 == 0
	var out []value
	for k > 0 {
		size, v, dots := 1, den, 0
		if compound {
			if offset%6 == 0 && k >= 6 {
				size, v, dots = 6, 2, 1
			} else if offset%3 == 0 && k >= 3 {
				size, v, dots = 3, 4, 1
			}
		} else if den == 4 && bar.Numerator == 3 && offset == 0 && k == 3 {
			size, v, dots = 3, 2, 1
		} else {
			for _, cand := range []int{8, 4, 2} {
				if cand <= k && offset%cand == 0 && den%cand == 0 {
					size, v = cand, den/cand
					break
				}
			}
		}
		out = append(out, value{v, dots})
		k -= size
		offset += size
	}
	return out
}

var binary = [][4]int{{8, 4, 0, 8}, {6, 8, 1, 2}, {4, 8, 0, 2}, {3, 16, 1, 1}, {2, 16, 0, 2}, {1, 32, 0, 1}}

func splitBinary(start, length int) []value {
	var out []value
	for length > 0 {
		for _, b := range binary {
			size, v, dots, align := b[0], b[1], b[2], b[3]
			if size <= length && pyMod(start, align) == 0 {
				out = append(out, value{v, dots})
				start += size
				length -= size
				break
			}
		}
	}
	return out
}

func splitTuplet(start, length, unitValue int) []int {
	var out []int
	for length > 0 {
		progressed := false
		for _, size := range []int{4, 2, 1} {
			if size <= length && pyMod(start, size) == 0 && unitValue/size >= 1 {
				out = append(out, unitValue/size)
				start += size
				length -= size
				progressed = true
				break
			}
		}
		if !progressed {
			panic("layout: tuplet span cannot be split") // Python loops forever here
		}
	}
	return out
}

func pyMod(a, b int) int {
	m := a % b
	if m != 0 && (m < 0) != (b < 0) {
		m += b
	}
	return m
}

type segment struct {
	start, end rat.Q
	notes      []*score.Note // nil = rest
	isRest     bool
}

type chunk struct {
	k    int
	a, b rat.Q
}

// Track returns, for every bar, the beats of the track. Durations in a bar sum to its length.
func Track(track *score.Track, bars []score.Bar) [][]Beat {
	drums := track.Kind == "drums"
	startsQ := make([]rat.Q, len(bars))
	for i, b := range bars {
		startsQ[i] = b.StartQ
	}
	totalEnd := bars[len(bars)-1].EndQ()

	cellEnd := func(absQ rat.Q) rat.Q {
		// bisect_right(starts_q, abs_q) - 1
		i := sort.Search(len(startsQ), func(j int) bool { return absQ.Less(startsQ[j]) }) - 1
		if i < 0 {
			i = 0
		}
		bar := bars[i]
		c := absQ.Sub(bar.StartQ).Div(bar.BeatQ()).Floor()
		return bar.StartQ.Add(rat.Int(c + 1).Mul(bar.BeatQ()))
	}

	type group struct {
		at    rat.Q
		notes []*score.Note
	}
	var groups []*group
	index := map[rat.Q]*group{}
	for i := range track.Notes {
		n := &track.Notes[i]
		at := bars[n.Bar].StartQ.Add(n.PosQ)
		g, ok := index[at]
		if !ok {
			g = &group{at: at}
			index[at] = g
			groups = append(groups, g)
		}
		g.notes = append(g.notes, n)
	}
	sort.SliceStable(groups, func(i, j int) bool { return groups[i].at.Less(groups[j].at) })

	var segments []segment
	cursor := bars[0].StartQ
	for i, g := range groups {
		s := g.at
		if !s.Less(totalEnd) {
			break
		}
		nxt := totalEnd
		if i+1 < len(groups) {
			nxt = groups[i+1].at
		}
		longest := g.notes[0].DurQ
		for _, n := range g.notes[1:] {
			longest = rat.Max(longest, n.DurQ)
		}
		end := rat.Min(rat.Min(nxt, s.Add(longest)), totalEnd)
		if drums {
			end = rat.Min(end, cellEnd(s))
		}
		if cursor.Less(s) {
			segments = append(segments, segment{cursor, s, nil, true})
		}
		notes := append([]*score.Note(nil), g.notes...)
		sort.SliceStable(notes, func(a, b int) bool {
			sa, sb := strOr0(notes[a]), strOr0(notes[b])
			if sa != sb {
				return sa < sb
			}
			return notes[a].Pitch < notes[b].Pitch
		})
		segments = append(segments, segment{s, end, notes, false})
		cursor = end
	}
	if cursor.Less(totalEnd) {
		segments = append(segments, segment{cursor, totalEnd, nil, true})
	}

	out := make([][]Beat, 0, len(bars))
	segI := 0
	emitted := map[int]bool{}
	zero := rat.Int(0)
	for _, bar := range bars {
		var beats []Beat
		den, bq := bar.Denominator, bar.BeatQ()
		cells := make([][]chunk, 0, bar.Numerator)
		for c := 0; c < bar.Numerator; c++ {
			c0 := bar.StartQ.Add(rat.Int(int64(c)).Mul(bq))
			c1 := bar.StartQ.Add(rat.Int(int64(c + 1)).Mul(bq))
			for segI < len(segments) && !c0.Less(segments[segI].end) {
				segI++
			}
			var chunks []chunk
			for k := segI; k < len(segments) && segments[k].start.Less(c1); k++ {
				a, b := rat.Max(segments[k].start, c0), rat.Min(segments[k].end, c1)
				if a.Less(b) {
					chunks = append(chunks, chunk{k, a.Sub(c0), b.Sub(c0)})
				}
			}
			cells = append(cells, chunks)
		}

		emit := func(k, duration, dots int, tuplet *[2]int) {
			seg := segments[k]
			first := !emitted[k]
			emitted[k] = true
			text := ""
			if len(seg.notes) > 0 && first {
				for _, n := range seg.notes {
					if n.LowConf {
						text = LowConfText
						break
					}
				}
			}
			beats = append(beats, Beat{duration, dots, tuplet, seg.isRest,
				len(seg.notes) > 0 && !first, seg.notes, text, valueLength(duration, dots, tuplet)})
		}

		for c := 0; c < len(cells); {
			chunks := cells[c]
			full := len(chunks) == 1 && chunks[0].a.Equal(zero) && chunks[0].b.Equal(bq)
			if full {
				k := chunks[0].k
				run := 1
				for c+run < len(cells) && len(cells[c+run]) == 1 && cells[c+run][0].k == k &&
					cells[c+run][0].a.Equal(zero) && cells[c+run][0].b.Equal(bq) {
					run++
				}
				for _, v := range splitCells(run, c, bar) {
					emit(k, v.duration, v.dots, nil)
				}
				c += run
				continue
			}
			s, ok := track.Grid[[2]int{bar.Index, c}]
			if !ok {
				s = DefaultGrid(den)
			}
			if s == 3 || s == 6 {
				unit := bq.Div(rat.Int(int64(s)))
				unitValue := 2 * den * s / 3
				for _, ch := range chunks {
					for _, v := range splitTuplet(int(ch.a.Div(unit).Trunc()), int(ch.b.Sub(ch.a).Div(unit).Trunc()), unitValue) {
						emit(ch.k, v, 0, &[2]int{3, 2})
					}
				}
			} else {
				unit := rat.New(1, 8)
				for _, ch := range chunks {
					for _, v := range splitBinary(int(ch.a.Div(unit).Trunc()), int(ch.b.Sub(ch.a).Div(unit).Trunc())) {
						emit(ch.k, v.duration, v.dots, nil)
					}
				}
			}
			c++
		}
		out = append(out, beats)
	}
	return out
}

func strOr0(n *score.Note) int {
	if n.String == nil {
		return 0
	}
	return *n.String
}

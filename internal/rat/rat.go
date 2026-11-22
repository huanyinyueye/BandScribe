// Package rat is an exact rational number over int64, matching Python's fractions.Fraction
// for the small denominators used by score positions. Overflow panics instead of wrapping.
package rat

import (
	"fmt"
	"math"
	"math/bits"
	"strconv"
	"strings"
)

// Q is a normalised fraction: D > 0 and gcd(|N|, D) == 1. The zero value is not valid;
// use Int(0).
type Q struct{ N, D int64 }

func gcd(a, b int64) int64 {
	if a < 0 {
		a = -a
	}
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

func mul(a, b int64) int64 {
	hi, lo := bits.Mul64(uint64(abs(a)), uint64(abs(b)))
	if hi != 0 || lo > math.MaxInt64 {
		panic("rat: overflow")
	}
	r := int64(lo)
	if (a < 0) != (b < 0) {
		r = -r
	}
	return r
}

func add(a, b int64) int64 {
	s := a + b
	if (a > 0 && b > 0 && s < 0) || (a < 0 && b < 0 && s >= 0) {
		panic("rat: overflow")
	}
	return s
}

func abs(a int64) int64 {
	if a < 0 {
		if a == math.MinInt64 {
			panic("rat: overflow")
		}
		return -a
	}
	return a
}

// New returns n/d normalised.
func New(n, d int64) Q {
	if d == 0 {
		panic("rat: zero denominator")
	}
	if d < 0 {
		n, d = -n, -d
	}
	g := gcd(n, d)
	if g == 0 {
		return Q{0, 1}
	}
	return Q{n / g, d / g}
}

func Int(n int64) Q { return Q{n, 1} }

func (a Q) Add(b Q) Q {
	g := gcd(a.D, b.D)
	return New(add(mul(a.N, b.D/g), mul(b.N, a.D/g)), mul(a.D, b.D/g))
}

func (a Q) Sub(b Q) Q { return a.Add(Q{-b.N, b.D}) }

func (a Q) Mul(b Q) Q {
	g1, g2 := gcd(a.N, b.D), gcd(b.N, a.D)
	if g1 == 0 {
		g1 = 1
	}
	if g2 == 0 {
		g2 = 1
	}
	return New(mul(a.N/g1, b.N/g2), mul(a.D/g2, b.D/g1))
}

func (a Q) Div(b Q) Q {
	if b.N == 0 {
		panic("rat: division by zero")
	}
	return a.Mul(New(b.D, b.N))
}

// Cmp returns -1, 0 or 1.
func (a Q) Cmp(b Q) int {
	l, r := mul(a.N, b.D), mul(b.N, a.D)
	switch {
	case l < r:
		return -1
	case l > r:
		return 1
	}
	return 0
}

func (a Q) Less(b Q) bool  { return a.Cmp(b) < 0 }
func (a Q) Equal(b Q) bool { return a.N == b.N && a.D == b.D }
func (a Q) Sign() int {
	switch {
	case a.N < 0:
		return -1
	case a.N > 0:
		return 1
	}
	return 0
}

// Floor is Python's math.floor / `//` on a Fraction.
func (a Q) Floor() int64 {
	q := a.N / a.D
	if a.N%a.D != 0 && a.N < 0 {
		q--
	}
	return q
}

// Trunc is Python's int() on a Fraction.
func (a Q) Trunc() int64 { return a.N / a.D }

// RoundHalfEven is Python's round() on a Fraction (ties to even).
func (a Q) RoundHalfEven() int64 {
	fl := a.Floor()
	rem := a.Sub(Int(fl)) // in [0, 1)
	switch rem.Cmp(New(1, 2)) {
	case -1:
		return fl
	case 1:
		return fl + 1
	}
	if fl%2 == 0 {
		return fl
	}
	return fl + 1
}

func (a Q) Float() float64 { return float64(a.N) / float64(a.D) }

func Min(a, b Q) Q {
	if b.Less(a) {
		return b
	}
	return a
}

func Max(a, b Q) Q {
	if a.Less(b) {
		return b
	}
	return a
}

// String is the "n/d" form used by score.json.
func (a Q) String() string { return fmt.Sprintf("%d/%d", a.N, a.D) }

// Parse reads the "n/d" form used by score.json.
func Parse(s string) (Q, error) {
	n, d, ok := strings.Cut(s, "/")
	if !ok {
		return Q{}, fmt.Errorf("rat: %q is not n/d", s)
	}
	ni, err := strconv.ParseInt(strings.TrimSpace(n), 10, 64)
	if err != nil {
		return Q{}, fmt.Errorf("rat: %q: %w", s, err)
	}
	di, err := strconv.ParseInt(strings.TrimSpace(d), 10, 64)
	if err != nil {
		return Q{}, fmt.Errorf("rat: %q: %w", s, err)
	}
	if di == 0 {
		return Q{}, fmt.Errorf("rat: %q has a zero denominator", s)
	}
	return New(ni, di), nil
}

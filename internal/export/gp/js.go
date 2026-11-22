// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy
// of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// Translated from alphaTab 1.9.0 (https://github.com/CoderLine/alphaTab), Copyright CoderLine.

package gp

import (
	"math"
	"strconv"
	"strings"
	"unicode/utf16"
)

var nan = math.NaN()

func isNaN(x float64) bool { return math.IsNaN(x) }

// jsString is ECMAScript Number::toString(x).
func jsString(x float64) string {
	switch {
	case math.IsNaN(x):
		return "NaN"
	case x == 0:
		return "0"
	case math.IsInf(x, 1):
		return "Infinity"
	case math.IsInf(x, -1):
		return "-Infinity"
	case x < 0:
		return "-" + jsString(-x)
	}
	e := strconv.FormatFloat(x, 'e', -1, 64)
	mant, exp, _ := strings.Cut(e, "e")
	digits := strings.Replace(mant, ".", "", 1)
	p, _ := strconv.Atoi(exp)
	k, n := len(digits), p+1
	switch {
	case k <= n && n <= 21:
		return digits + strings.Repeat("0", n-k)
	case 0 < n && n <= 21:
		return digits[:n] + "." + digits[n:]
	case -6 < n && n <= 0:
		return "0." + strings.Repeat("0", -n) + digits
	}
	sign := "+"
	if n-1 < 0 {
		sign = "-"
	}
	ev := strconv.Itoa(abs(n - 1))
	if k == 1 {
		return digits + "e" + sign + ev
	}
	return digits[:1] + "." + digits[1:] + "e" + sign + ev
}

func abs(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

// jsRound is Math.round: nearest integer, halves toward +Infinity.
func jsRound(x float64) float64 {
	if math.IsNaN(x) || math.IsInf(x, 0) || x == 0 {
		return x
	}
	f := math.Floor(x)
	if x-f >= 0.5 {
		return f + 1
	}
	return f
}

// jsMax is Math.max(a, b) (NaN wins).
func jsMax(a, b float64) float64 {
	if math.IsNaN(a) || math.IsNaN(b) {
		return nan
	}
	return math.Max(a, b)
}

// toInt32 is the `x | 0` conversion.
func toInt32(x float64) int64 {
	if math.IsNaN(x) || math.IsInf(x, 0) {
		return 0
	}
	m := math.Mod(math.Trunc(x), 4294967296)
	if m < 0 {
		m += 4294967296
	}
	if m >= 2147483648 {
		m -= 4294967296
	}
	return int64(m)
}

// utf16Slice is String.prototype.slice(0, n): UTF-16 code units; a split surrogate pair becomes
// U+FFFD when the string is encoded as UTF-8 (TextEncoder).
func utf16Slice(s string, n int) string {
	u := utf16.Encode([]rune(s))
	if len(u) > n {
		u = u[:n]
	}
	return string(utf16.Decode(u))
}

// key normalises a JS Map number key (SameValueZero: NaN equals NaN, -0 equals 0).
func key(x float64) float64 {
	if math.IsNaN(x) {
		return math.Inf(-1)
	}
	return x + 0
}

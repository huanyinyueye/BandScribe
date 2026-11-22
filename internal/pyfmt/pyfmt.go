// Package pyfmt reproduces the Python number formatting the original exporters relied on.
package pyfmt

import (
	"math"
	"strconv"
	"strings"
)

// Repr is Python's repr(float).
func Repr(x float64) string {
	switch {
	case math.IsNaN(x):
		return "nan"
	case math.IsInf(x, 1):
		return "inf"
	case math.IsInf(x, -1):
		return "-inf"
	case x == 0:
		if math.Signbit(x) {
			return "-0.0"
		}
		return "0.0"
	}
	e := strconv.FormatFloat(x, 'e', -1, 64) // -d.ddde±XX
	mant, expStr, _ := strings.Cut(e, "e")
	exp, _ := strconv.Atoi(expStr)
	neg := strings.HasPrefix(mant, "-")
	digits := strings.Replace(strings.TrimPrefix(mant, "-"), ".", "", 1)
	sign := ""
	if neg {
		sign = "-"
	}
	if exp < -4 || exp >= 16 {
		m := digits[:1]
		if len(digits) > 1 {
			m += "." + digits[1:]
		}
		es := strconv.Itoa(abs(exp))
		if len(es) < 2 {
			es = "0" + es
		}
		if exp < 0 {
			es = "-" + es
		} else {
			es = "+" + es
		}
		return sign + m + "e" + es
	}
	if exp < 0 {
		return sign + "0." + strings.Repeat("0", -exp-1) + digits
	}
	if len(digits) <= exp+1 {
		return sign + digits + strings.Repeat("0", exp+1-len(digits)) + ".0"
	}
	return sign + digits[:exp+1] + "." + digits[exp+1:]
}

// Round is Python's round(x, ndigits) for ndigits >= 0: correctly rounded, ties to even.
func Round(x float64, ndigits int) float64 {
	if math.IsNaN(x) || math.IsInf(x, 0) {
		return x
	}
	v, err := strconv.ParseFloat(strconv.FormatFloat(x, 'f', ndigits, 64), 64)
	if err != nil {
		return x
	}
	return v
}

// RoundInt is Python's round(x) on a float.
func RoundInt(x float64) int64 { return int64(math.RoundToEven(x)) }

func abs(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

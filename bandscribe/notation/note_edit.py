"""Validate musical positions supplied by the score editor."""

import re
from fractions import Fraction


def note_timing(position: str, duration: str, bar_length: Fraction) -> tuple[Fraction, Fraction]:
    try:
        if any(len(v) > 32 or re.fullmatch(r"\d+(?:/\d+|\.\d+)?", v) is None
               for v in (position, duration)):
            raise ValueError("invalid fraction")
        pos, dur = Fraction(position), Fraction(duration)
    except (ValueError, ZeroDivisionError, TypeError) as exc:
        raise ValueError("invalid note position or duration") from exc
    if pos < 0 or dur <= 0 or pos + dur > bar_length or pos.denominator > 960 or dur.denominator > 960:
        raise ValueError("note position or duration is outside the bar")
    return pos, dur

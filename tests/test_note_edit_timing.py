from fractions import Fraction

import pytest

from bandscribe.notation.note_edit import note_timing


def test_note_timing_rejects_invalid_or_out_of_bar_positions():
    assert note_timing("1/2", "1/4", Fraction(4)) == (Fraction(1, 2), Fraction(1, 4))
    for position, duration in [("x", "1"), ("0", "0"), ("-1", "1"), ("7/2", "1"), ("1/0", "1")]:
        with pytest.raises(ValueError):
            note_timing(position, duration, Fraction(4))

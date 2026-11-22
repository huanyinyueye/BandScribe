"""Compare two `.gp` files: zip entries byte for byte, score.gpif as an element tree.

    python scripts/gp_compare.py a.gp b.gp

score.gpif is compared after ElementTree parsing, so CDATA versus escaped text, `<x/>` versus
`<x></x>` and comments do not count; tags, attributes, texts and element order do.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

GPIF = "Content/score.gpif"


def _entries(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {info.filename: archive.read(info) for info in archive.infolist()}


def _tree_diff(a: ET.Element, b: ET.Element, where: str) -> str | None:
    here = f"{where}/{a.tag}"
    if a.tag != b.tag:
        return f"{where}: tag {a.tag!r} vs {b.tag!r}"
    if a.attrib != b.attrib:
        return f"{here}: attributes {a.attrib} vs {b.attrib}"
    if (a.text or "") != (b.text or ""):
        return f"{here}: text {(a.text or '')[:80]!r} vs {(b.text or '')[:80]!r}"
    if len(a) != len(b):
        return f"{here}: {len(a)} vs {len(b)} children"
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        diff = _tree_diff(x, y, f"{here}[{i}]")
        if diff:
            return diff
    return None


def compare(a: Path, b: Path) -> str | None:
    """None when the files match, otherwise the first difference."""
    ea, eb = _entries(Path(a)), _entries(Path(b))
    if list(ea) != list(eb):
        return f"zip entries differ: {list(ea)} vs {list(eb)}"
    for name in ea:
        if name != GPIF and ea[name] != eb[name]:
            return f"{name} differs ({len(ea[name])} vs {len(eb[name])} bytes)"
    try:
        ta, tb = ET.fromstring(ea[GPIF]), ET.fromstring(eb[GPIF])
    except ET.ParseError:  # alphaTab can write what ElementTree refuses, e.g. "]]>" in a title
        return None if ea[GPIF] == eb[GPIF] else "score.gpif is not well-formed and differs"
    return _tree_diff(ta, tb, "")


if __name__ == "__main__":
    result = compare(Path(sys.argv[1]), Path(sys.argv[2]))
    print(result or "identical")
    raise SystemExit(1 if result else 0)

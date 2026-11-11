"""Fetch only the MUSDB18-HQ test set from the Zenodo zip using parallel HTTP range requests.

Zenodo serves about 0.3-1 MB/s per connection, so the full 22.7 GB zip takes many hours.
The zip central directory gives the byte range of every member; this script downloads the
`test/` members in parallel and verifies each CRC32.

Usage: uv run python benchmarks/fetch_musdb_test.py [--workers 12]
"""

from __future__ import annotations

import argparse
import io
import struct
import time
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

URL = "https://zenodo.org/records/3338373/files/musdb18hq.zip?download=1"
DEST = Path.home() / ".cache" / "bandscribe" / "datasets" / "musdb18hq"


def fetch_range(start: int, end: int, retries: int = 8) -> bytes:
    """Bytes [start, end] inclusive."""
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}",
                                                       "User-Agent": "bandscribe-dataset-fetch"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            if len(data) == end - start + 1:
                return data
            last = RuntimeError(f"short read {len(data)} of {end - start + 1}")
        except Exception as exc:  # noqa: BLE001 - retried
            last = exc
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"range {start}-{end} failed: {last}")


class HttpTail(io.RawIOBase):
    """Read-only file object over the last ``tail`` bytes of the remote file."""

    def __init__(self, size: int, tail: int = 4 << 20) -> None:
        self.size = size
        self.base = max(0, size - tail)
        self.buf = fetch_range(self.base, size - 1)
        self.pos = size

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = self.size - self.pos
        if self.pos < self.base:
            data = fetch_range(self.pos, min(self.size, self.pos + n) - 1)
        else:
            data = self.buf[self.pos - self.base:self.pos - self.base + n]
        self.pos += len(data)
        return data


def fetch_member(info: zipfile.ZipInfo) -> str:
    out = DEST / info.filename
    if out.is_file() and out.stat().st_size == info.file_size:
        return f"cached {info.filename}"
    head = fetch_range(info.header_offset, info.header_offset + 29)
    name_len, extra_len = struct.unpack("<HH", head[26:30])
    start = info.header_offset + 30 + name_len + extra_len
    raw = fetch_range(start, start + info.compress_size - 1) if info.compress_size else b""
    if info.compress_type == zipfile.ZIP_DEFLATED:
        raw = zlib.decompress(raw, -15)
    elif info.compress_type != zipfile.ZIP_STORED:
        raise RuntimeError(f"unsupported compression {info.compress_type}")
    if zlib.crc32(raw) != info.CRC:
        raise RuntimeError(f"CRC mismatch: {info.filename}")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    tmp.write_bytes(raw)
    tmp.replace(out)
    return f"ok {info.filename}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--prefix", default="test/")
    args = ap.parse_args()
    req = urllib.request.Request(URL, method="HEAD", headers={"User-Agent": "bandscribe-dataset-fetch"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        size = int(resp.headers["Content-Length"])
    zf = zipfile.ZipFile(HttpTail(size))
    members = [i for i in zf.infolist() if i.filename.startswith(args.prefix) and not i.is_dir()]
    total = sum(i.compress_size for i in members)
    print(f"{len(members)} files, {total / 1e9:.2f} GB compressed", flush=True)
    done = 0
    started = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(fetch_member, i): i for i in members}
        for n, fut in enumerate(as_completed(futures), 1):
            fut.result()
            done += futures[fut].compress_size
            if n % 25 == 0 or n == len(members):
                rate = done / max(1e-9, time.time() - started) / 1e6
                print(f"{n}/{len(members)} files, {done / 1e9:.2f} GB, {rate:.1f} MB/s", flush=True)


if __name__ == "__main__":
    main()

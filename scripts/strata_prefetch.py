#!/usr/bin/env python3
"""Prefetch the non-expert tensors of a GGUF shard into a sparse file, for --hf-stream-vram.

The recipe still needs a pack and a tokenizer before the engine starts, and `iq_pack` reads the model's
float tensors (norms, routers, SSM/GDN projections, embeddings served from the pack).  It never reads the
routed experts or the 28.8 GB PLE table.  Downloading the whole 55 GB shard first would defeat the
engine's streaming path, so this writes a full-size sparse file with:

  * the GGUF header (metadata + tensor directory),
  * every non-expert tensor's payload (`token_embd`, `output`, norms, ...), except the PLE table.

The engine then sees `<path>.hfstream` and fetches the whole shard over HTTP Range while it loads.  The
sparse file is also what setup.py's `--gguf-dir` validation and `iq_pack` inspect before the engine starts.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request

CHUNK = 1 << 20
DEFAULT_WORKERS = 8


def total_size(url: str) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": "collabosm-strata-prefetch",
                                               "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        cr = (r.headers.get("Content-Range") or "").strip()
        m = re.match(r"bytes\s+0-0/(\d+)$", cr)
        if m:
            return int(m.group(1))
        length = r.headers.get("Content-Length")
        if length and r.status == 200:
            return int(length)
    raise RuntimeError("the server did not report the file size (HTTP %s)" % r.status)


def fetch(url: str, start: int, end: int, attempts: int = 4) -> bytes:
    """bytes [start, end) with retries; 0-based, end exclusive."""
    if end <= start:
        return b""
    last = None
    for attempt in range(attempts):
        req = urllib.request.Request(url, headers={
            "User-Agent": "collabosm-strata-prefetch",
            "Range": "bytes=%d-%d" % (start, end - 1),
        })
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
                if r.status != 206 and not (r.status == 200 and start == 0):
                    raise RuntimeError("HTTP %s for bytes %d-%d" % (r.status, start, end - 1))
                if len(data) != end - start:
                    raise RuntimeError("short range: %d of %d bytes" % (len(data), end - start))
                return data
        except (OSError, urllib.error.URLError, RuntimeError) as e:
            last = e
            time.sleep(1.0 + attempt)
    raise RuntimeError("range %d-%d failed: %s" % (start, end - 1, last))


def write_at(fd: int, data: bytes, offset: int) -> None:
    view = memoryview(data)
    done = 0
    while done < len(data):
        n = os.pwrite(fd, view[done:], offset + done)
        if n <= 0:
            raise RuntimeError("short pwrite at %d" % (offset + done))
        done += n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--tools", required=True, help="terminal/tools of the Strata source checkout")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    a = ap.parse_args()

    path = pathlib.Path(a.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, a.tools)
    import gguf_reader as G  # noqa: E402

    size = total_size(a.url)
    print("strata-prefetch: %s, %.2f GiB" % (path.name, size / 2**30), flush=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.ftruncate(fd, size)

        # The header grows geometrically until the parser accepts it.  Writing the prefix first lets the parser
        # see exactly what a reader would see, and the full size makes its payload bounds pass.
        probe = min(size, 8 << 20)
        parsed = None
        while parsed is None:
            write_at(fd, fetch(a.url, 0, probe), 0)
            try:
                g = G.GGUFFile(path)
                if g.data_start <= probe:
                    parsed = g
            except Exception as e:  # noqa: BLE001 - a short prefix is expected here
                if probe >= size or probe >= (128 << 20):
                    raise RuntimeError("cannot parse the GGUF header: %s" % e)
            if parsed is None:
                if probe >= size or probe >= (128 << 20):
                    raise RuntimeError("the GGUF header does not fit in 128 MiB")
                probe = min(size, probe * 2)
        g = parsed
        print("strata-prefetch: header %d B, data at %d, %d tensors"
              % (probe, g.data_start, len(g.tensors)), flush=True)

        ranges = []
        total = 0
        for t in g.tensors:
            if t.name.endswith("_exps.weight") or t.name == "per_layer_token_embd.weight":
                continue
            n = t.expected_bytes()
            if n <= 0:
                continue
            start = g.data_start + t.offset
            if start + n > size:
                raise RuntimeError("tensor %s runs past the end of the file" % t.name)
            ranges.append((t.name, start, n))
            total += n
        print("strata-prefetch: %d non-expert tensors, %.2f GiB to fetch"
              % (len(ranges), total / 2**30), flush=True)

        done = 0
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
            futures = {pool.submit(fetch, a.url, start, start + n): (name, start, n)
                       for name, start, n in ranges}
            for fut in concurrent.futures.as_completed(futures):
                name, start, n = futures[fut]
                data = fut.result()
                write_at(fd, data, start)
                done += n
                if done < total and (done >= 512 << 20 or time.time() - t0 > 10):
                    print("strata-prefetch: %.1f%% (%.2f GiB)"
                          % (100.0 * done / total, done / 2**30), flush=True)
                    t0 = time.time()
        os.fsync(fd)
        # The engine sees this marker and replaces the sparse file with the full HTTP Range download.
        path.with_name(path.name + ".hfstream").write_text(
            a.url + "\n" + str(size) + "\n", encoding="utf-8")
        print("strata-prefetch: done, %.2f GiB of tensor payload present; .hfstream marker written"
              % (total / 2**30), flush=True)
        return 0
    finally:
        os.close(fd)


if __name__ == "__main__":
    raise SystemExit(main())

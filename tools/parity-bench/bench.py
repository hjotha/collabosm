#!/usr/bin/env python3
"""parity-bench: reproducible A100 harness for phases 0/1 of the Strata parity plan.

Python 3 standard library only. Replaces the ad-hoc /tmp/opencode/vm_test_parity.py
and /tmp/opencode/vm_bench_parity.py.

Each measured run spawns a fresh llama-server, waits for /health, optionally sends
one discarded warmup request, sends the fixed prompt cases, samples device VRAM at
1 Hz, then stops the server with SIGTERM (so the llama_moe_cache destructor flushes
its hit/miss lines into the log) and parses those lines.

See README.md for the protocol, acceptance criteria and A100 instructions.
"""

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

SCHEMA = "parity-bench/1"
ABLATE_SCHEMA = "parity-bench-ablate/1"

HEARTBEAT_S = 15.0          # the VM aborts execs that print nothing for ~30 s
VRAM_SAMPLE_S = 1.0
HOST = "127.0.0.1"
ALIAS = "q"

DEFAULT_EXTRA_ARGS = ["--jinja", "--no-warmup"]

# Case names are part of the output contract: ablate.py and the acceptance checks
# refer to them literally.
CASE_SHORT = "short"
CASE_4K = "pp4k"
CASE_4K_DEC = "pp4k+dec"
CASE_28K = "pp28k"
CASE_28K_DEC = "pp28k+dec"
CASE_NAMES = (CASE_SHORT, CASE_4K, CASE_4K_DEC, CASE_28K, CASE_28K_DEC)

# The filler was calibrated against the Qwen3.8-Flash tokenizer: 100 blocks measured
# 4,662 prompt tokens and 620 blocks measured ~28.6k in the previous ad-hoc scripts.
# prompt_n in the JSON is the authoritative value; update these counts if the
# tokenizer or the filler changes.
FILLER = ("def process(rows):\n    out = []\n    for row in rows:\n"
          "        if row is None:\n            continue\n"
          "        out.append((row[0], sum(row[1:])))\n    return out\n\n")
CODE_BLOCKS_4K = 100
CODE_BLOCKS_28K = 620
EXPECTED_PROMPT_TOKENS = {CASE_4K: 4662, CASE_28K: 28600}

KV_CHOICES = ("f16", "q8_0")
LAZY_CHOICES = ("on", "off", "auto")
FA_CHOICES = ("on", "off", "auto")
LOAD_MODE_CHOICES = ("auto", "none", "mmap", "mlock", "mmap+mlock", "dio")

# Flags owned by the harness. Letting extra_args repeat them would silently break
# the A/B protocol, so they are rejected during validation.
MANAGED_FLAGS = {
    "-m", "--model", "--alias", "--host", "--port",
    "-c", "--ctx-size", "-ngl", "--gpu-layers", "--n-gpu-layers",
    "-b", "--batch-size", "-ub", "--ubatch-size",
    "-ctk", "--cache-type-k", "-ctv", "--cache-type-v",
    "-fa", "--flash-attn", "-lm", "--load-mode", "-lzm", "--lazy-mode",
    "--moe-cache-mib", "--vram-reserve-mib", "--n-cpu-moe", "-ncmoe",
    "--cpu-moe", "-cmoe",
}

DEFAULTS = {
    "name": None,
    "bin_dir": None,          # None -> global --bin-dir (lets one run compare builds)
    "kv": "f16",
    "moe_cache_mib": 6144,
    "batch": 2048,
    "ubatch": 512,
    "ctx": 32768,
    "ngl": 99,
    "fa": "on",
    "load_mode": "mmap",
    "lazy_mode": "on",
    "n_cpu_moe": 2,
    "vram_reserve_mib": None,
    "runs": None,
    "port": None,
    "extra_args": [],
    "env": {},
}

JSON_ALIASES = {
    "cache": "moe_cache_mib",
    "cache_mib": "moe_cache_mib",
    "b": "batch",
    "ub": "ubatch",
    "build": "bin_dir",
    "lazy": "lazy_mode",
    "ncmoe": "n_cpu_moe",
    "reserve": "vram_reserve_mib",
    "extra": "extra_args",
}

CLI_KEYS = {
    "name": ("name", str),
    "kv": ("kv", str),
    "cache": ("moe_cache_mib", int),
    "moe_cache_mib": ("moe_cache_mib", int),
    "b": ("batch", int),
    "batch": ("batch", int),
    "ub": ("ubatch", int),
    "ubatch": ("ubatch", int),
    "ctx": ("ctx", int),
    "ngl": ("ngl", int),
    "fa": ("fa", str),
    "lazy": ("lazy_mode", str),
    "lazy_mode": ("lazy_mode", str),
    "load_mode": ("load_mode", str),
    "ncmoe": ("n_cpu_moe", int),
    "n_cpu_moe": ("n_cpu_moe", int),
    "reserve": ("vram_reserve_mib", int),
    "vram_reserve_mib": ("vram_reserve_mib", int),
    "build": ("bin_dir", str),
    "bin_dir": ("bin_dir", str),
    "runs": ("runs", int),
    "port": ("port", int),
}

# No proxy for localhost, even if http_proxy is exported on the VM.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

RE_MOE_STATS = re.compile(
    r"llama_moe_cache:\s*(?P<bucket>[^:]+?):\s*"
    r"hits\s*=\s*(?P<hits>\d+),\s*"
    r"misses\s*=\s*(?P<misses>\d+),\s*"
    r"hit rate\s*=\s*(?P<rate>[0-9]+\.[0-9]+)%,\s*"
    r"uploaded\s*=\s*(?P<uploaded>[0-9]+\.[0-9]+)\s*MiB"
)
RE_MOE_LARGE_COPY = re.compile(
    r"llama_moe_cache:\s*large batches:\s*(?P<experts>\d+)\s+experts copied "
    r"from the cache,\s*(?P<mib>[0-9]+\.[0-9]+)\s*MiB"
)
RE_MOE_AUTO_SIZED = re.compile(
    r"MoE cache auto:\s*(?P<budget>[0-9]+\.[0-9]+)\s*MiB for\s*"
    r"(?P<host>[0-9]+\.[0-9]+)\s*MiB of host experts\s*"
    r"\((?P<pct>[0-9]+\.[0-9]+)% of the experts can be cached\)"
)
RE_MOE_AUTO_LOW = re.compile(
    r"MoE cache auto:\s*(?P<host>[0-9]+\.[0-9]+)\s*MiB of host experts, but only\s*"
    r"(?P<budget>[0-9]+\.[0-9]+)\s*MiB of device memory is free after the compute "
    r"buffers \(reserve\s*(?P<reserve>[0-9]+\.[0-9]+)\s*MiB\), cache disabled"
)
RE_MOE_AUTO_NONE = re.compile(
    r"MoE cache auto:\s*(?P<reason>no host-resident MoE experts, cache disabled|"
    r"no_alloc context, leaving the cache disabled)"
)
RE_MOE_SIZE = re.compile(
    r"(?P<device>\S+)\s+MoE cache size\s*=\s*(?P<cache>[0-9]+\.[0-9]+)\s*MiB for\s*"
    r"(?P<host>[0-9]+\.[0-9]+)\s*MiB of host experts"
)
RE_MOE_PLAN = re.compile(
    r"MoE cache plan:\s*(?P<device>\S+)\s*=\s*(?P<mib>[0-9]+\.[0-9]+)\s*MiB"
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_LOG = os.path.join(SCRIPT_DIR, "examples", "moe_cache_sample.log")
SAMPLE_LOG_DISABLED = os.path.join(SCRIPT_DIR, "examples", "moe_cache_disabled_sample.log")


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------

def say(msg=""):
    print(msg, flush=True)


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def median(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return float(statistics.median(vals))


def round_or_none(value, digits):
    return None if value is None else round(value, digits)


def pct_delta(value, base):
    """(value/base - 1) * 100, or None when either side is unusable."""
    if value is None or base is None or base == 0:
        return None
    return (float(value) / float(base) - 1.0) * 100.0


def fmt(value, spec="%.2f"):
    return "n/a" if value is None else spec % value


def tail_text(text, n_lines=15, max_chars=1600):
    lines = text.strip().splitlines()
    out = "\n".join(lines[-n_lines:])
    if len(out) > max_chars:
        out = out[-max_chars:]
    return out


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def write_json_atomic(path, payload):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1, sort_keys=False)
        f.write("\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# GPU / process helpers
# ---------------------------------------------------------------------------

HAS_NVSMI = shutil.which("nvidia-smi") is not None


def nvidia_smi(args, timeout=5):
    try:
        proc = subprocess.run(["nvidia-smi"] + args, capture_output=True, text=True,
                              timeout=timeout)
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def query_vram(gpu_index=None):
    """Return (used_mib, total_mib) for the selected device, or (None, None)."""
    if not HAS_NVSMI:
        return None, None
    cmd = []
    if gpu_index is not None:
        cmd += ["-i", str(gpu_index)]
    cmd += ["--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"]
    out = nvidia_smi(cmd)
    if not out:
        return None, None
    first = out.splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 2:
        return None, None
    try:
        return int(float(parts[0])), int(float(parts[1]))
    except ValueError:
        return None, None


def query_gpu_info():
    if not HAS_NVSMI:
        return {"available": False, "reason": "nvidia-smi not found"}
    out = nvidia_smi(["--query-gpu=index,name,memory.total,driver_version",
                      "--format=csv,noheader,nounits"])
    if not out:
        return {"available": False, "reason": "nvidia-smi failed"}
    devices = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            devices.append({"index": parts[0], "name": parts[1],
                            "total_mib": int(float(parts[2])), "driver": parts[3]})
    return {"available": True, "devices": devices}


class Heartbeat(threading.Thread):
    def __init__(self, interval=HEARTBEAT_S):
        super().__init__(name="heartbeat", daemon=True)
        self.interval = interval
        self.phase = "startup"
        # Not `_stop`: Python 3.12's Thread.join() calls self._stop() internally,
        # so an Event under that name shadows the method and raises TypeError.
        self._stop_event = threading.Event()
        self.t0 = time.monotonic()

    def set_phase(self, phase):
        self.phase = phase

    def run(self):
        while not self._stop_event.wait(self.interval):
            say("[heartbeat] %ds elapsed, phase: %s"
                % (int(time.monotonic() - self.t0), self.phase))

    def stop(self):
        self._stop_event.set()
        self.join(timeout=2.0)


class VramSampler(threading.Thread):
    def __init__(self, gpu_index=None, interval=VRAM_SAMPLE_S):
        super().__init__(name="vram", daemon=True)
        self.gpu_index = gpu_index
        self.interval = interval
        self.samples = []
        self.peak = None
        self.total = None
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            used, total = query_vram(self.gpu_index)
            if used is not None:
                self.samples.append(used)
                if self.peak is None or used > self.peak:
                    self.peak = used
                if self.total is None:
                    self.total = total
            self._stop_event.wait(self.interval)

    def stop(self):
        self._stop_event.set()
        self.join(timeout=5.0)


def port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((HOST, port)) == 0


def port_is_free(port):
    """True when nothing listens on the port and a fresh bind succeeds."""
    if port_in_use(port):
        return False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((HOST, port))
        except OSError:
            return False
    return True


def pick_port(preferred):
    """Return the preferred port if free, else the next free port above it."""
    for port in range(preferred, min(preferred + 50, 65536)):
        if port_is_free(port):
            if port != preferred:
                say("[port] %d busy, using %d instead" % (preferred, port))
            return port
    raise RuntimeError("no free port in [%d, %d)" % (preferred, preferred + 50))


def wait_port_free(port, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if port_is_free(port):
            return True
        time.sleep(0.25)
    return False


def send_group(child, sig):
    try:
        os.killpg(os.getpgid(child.pid), sig)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            child.send_signal(sig)
        except ProcessLookupError:
            pass


def stop_server(child, timeout):
    """SIGTERM, wait, SIGKILL. Returns exit info; SIGTERM lets MoE stats flush."""
    info = {"graceful": None, "exit_code": None, "shutdown_s": 0.0}
    if child is None:
        return info
    t0 = time.monotonic()
    if child.poll() is None:
        send_group(child, signal.SIGTERM)
        deadline = t0 + timeout
        while time.monotonic() < deadline and child.poll() is None:
            time.sleep(0.2)
    if child.poll() is None:
        send_group(child, signal.SIGKILL)
        info["graceful"] = False
        try:
            child.wait(timeout=10)
        except Exception:
            pass
    else:
        info["graceful"] = True
        try:
            child.wait(timeout=5)
        except Exception:
            pass
    info["exit_code"] = child.returncode
    info["shutdown_s"] = round(time.monotonic() - t0, 2)
    return info


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def http_health(port, timeout=2.0):
    url = "http://%s:%d/health" % (HOST, port)
    with _OPENER.open(url, timeout=timeout) as resp:
        return resp.status


def http_post_json(port, path, payload, timeout):
    url = "http://%s:%d%s" % (HOST, port, path)
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")[:600]
        except Exception:
            pass
        raise RuntimeError("HTTP %s from %s: %s" % (exc.code, path, detail))
    except urllib.error.URLError as exc:
        raise RuntimeError("connection failed to %s: %s" % (path, exc.reason))


def wait_health(port, child, timeout):
    t0 = time.monotonic()
    while True:
        elapsed = time.monotonic() - t0
        rc = child.poll()
        if rc is not None:
            return {"ok": False, "elapsed_s": round(elapsed, 1),
                    "error": "server exited with code %s during load" % rc}
        if elapsed > timeout:
            return {"ok": False, "elapsed_s": round(elapsed, 1),
                    "error": "health check timed out after %.0fs" % timeout}
        try:
            if http_health(port) == 200:
                return {"ok": True, "elapsed_s": round(elapsed, 1)}
        except Exception:
            pass
        time.sleep(1.0)


# ---------------------------------------------------------------------------
# log parsing (no GPU needed; covered by --self-test)
# ---------------------------------------------------------------------------

def parse_moe_cache_log(text):
    """Parse the MoE cache plan/stats lines from a llama-server log.

    Returns a dict with the individual matches plus a cross-line summary. All
    values come from the log; nothing is inferred when a line is absent.
    """
    parsed = {
        "stats": [],
        "large_batch_copy": [],
        "auto": [],
        "sizes": [],
        "plans": [],
        "summary": {
            "small_hits": None, "small_misses": None, "small_hit_rate_pct": None,
            "large_hits": None, "large_misses": None, "large_hit_rate_pct": None,
            "uploaded_mib_total": None,
        },
    }
    for line in text.splitlines():
        m = RE_MOE_STATS.search(line)
        if m:
            bucket = " ".join(m.group("bucket").split())
            parsed["stats"].append({
                "bucket": bucket,
                "hits": int(m.group("hits")),
                "misses": int(m.group("misses")),
                "hit_rate_pct": float(m.group("rate")),
                "uploaded_mib": float(m.group("uploaded")),
            })
            continue
        m = RE_MOE_LARGE_COPY.search(line)
        if m:
            parsed["large_batch_copy"].append({
                "experts": int(m.group("experts")),
                "mib": float(m.group("mib")),
            })
            continue
        m = RE_MOE_AUTO_SIZED.search(line)
        if m:
            parsed["auto"].append({
                "kind": "sized",
                "budget_mib": float(m.group("budget")),
                "host_mib": float(m.group("host")),
                "cached_pct": float(m.group("pct")),
            })
            continue
        m = RE_MOE_AUTO_LOW.search(line)
        if m:
            parsed["auto"].append({
                "kind": "disabled_low_budget",
                "host_mib": float(m.group("host")),
                "budget_mib": float(m.group("budget")),
                "reserve_mib": float(m.group("reserve")),
            })
            continue
        m = RE_MOE_AUTO_NONE.search(line)
        if m:
            parsed["auto"].append({"kind": "disabled", "reason": m.group("reason")})
            continue
        m = RE_MOE_SIZE.search(line)
        if m:
            parsed["sizes"].append({
                "device": m.group("device"),
                "cache_mib": float(m.group("cache")),
                "host_mib": float(m.group("host")),
            })
            continue
        m = RE_MOE_PLAN.search(line)
        if m:
            parsed["plans"].append({
                "device": m.group("device"),
                "mib": float(m.group("mib")),
            })

    def bucket_stats(name):
        return [s for s in parsed["stats"]
                if " ".join(s["bucket"].split()) == name]

    small = bucket_stats("ubatch <= 8")
    large = bucket_stats("ubatch > 8")
    summary = parsed["summary"]
    for key, entries in (("small", small), ("large", large)):
        if not entries:
            continue
        hits = sum(e["hits"] for e in entries)
        misses = sum(e["misses"] for e in entries)
        total = hits + misses
        summary[key + "_hits"] = hits
        summary[key + "_misses"] = misses
        summary[key + "_hit_rate_pct"] = round(100.0 * hits / total, 2) if total else None
    if parsed["stats"]:
        summary["uploaded_mib_total"] = round(
            sum(e["uploaded_mib"] for e in parsed["stats"]), 2)
    return parsed


def summarize_moe_runs(runs):
    """Aggregate parsed MoE stats across the runs of one config."""
    out = {
        "small_hits": 0, "small_misses": 0, "small_hit_rate_pct": None,
        "large_hits": 0, "large_misses": 0, "large_hit_rate_pct": None,
        "uploaded_mib_total": 0.0,
        "uploaded_mib_per_run": None,
        "large_batch_copy": None,
        "auto": [],
        "cache_size": [],
        "runs_with_stats": 0,
    }
    uploaded_per_run = []
    for run in runs:
        moe = run.get("moe_cache") or {}
        summary = moe.get("summary") or {}
        if (summary.get("small_hits") is not None
                or summary.get("large_hits") is not None):
            out["runs_with_stats"] += 1
        for key in ("small_hits", "small_misses", "large_hits", "large_misses"):
            if summary.get(key) is not None:
                out[key] += summary[key]
        if summary.get("uploaded_mib_total") is not None:
            out["uploaded_mib_total"] += summary["uploaded_mib_total"]
            uploaded_per_run.append(summary["uploaded_mib_total"])
        if moe.get("large_batch_copy"):
            out["large_batch_copy"] = moe["large_batch_copy"][-1]
        if moe.get("auto") and not out["auto"]:
            out["auto"] = moe["auto"]
        if moe.get("sizes") and not out["cache_size"]:
            out["cache_size"] = moe["sizes"]
    for key in ("small", "large"):
        hits = out[key + "_hits"]
        misses = out[key + "_misses"]
        total = hits + misses
        out[key + "_hit_rate_pct"] = round(100.0 * hits / total, 2) if total else None
    out["uploaded_mib_total"] = round(out["uploaded_mib_total"], 2)
    out["uploaded_mib_per_run"] = round_or_none(median(uploaded_per_run), 2)
    if not out["runs_with_stats"]:
        out["uploaded_mib_total"] = None
        out["uploaded_mib_per_run"] = None
    return out


# ---------------------------------------------------------------------------
# config handling
# ---------------------------------------------------------------------------

def build_cases(selected=None):
    lp4 = FILLER * CODE_BLOCKS_4K + "\nSummarize the code above in one sentence."
    lp28 = FILLER * CODE_BLOCKS_28K + "\nSummarize the code above in one sentence."
    cases = [
        {"name": CASE_SHORT,
         "prompt": "Write a python function that reverses a string, briefly.",
         "max_tokens": 128, "calibrated_prompt_tokens": None},
        {"name": CASE_4K, "prompt": lp4, "max_tokens": 16,
         "calibrated_prompt_tokens": EXPECTED_PROMPT_TOKENS[CASE_4K]},
        {"name": CASE_4K_DEC,
         "prompt": lp4 + " Explain the time complexity.", "max_tokens": 128,
         "calibrated_prompt_tokens": EXPECTED_PROMPT_TOKENS[CASE_4K]},
        {"name": CASE_28K, "prompt": lp28, "max_tokens": 16,
         "calibrated_prompt_tokens": EXPECTED_PROMPT_TOKENS[CASE_28K]},
        {"name": CASE_28K_DEC,
         "prompt": lp28 + " Explain the time complexity.", "max_tokens": 128,
         "calibrated_prompt_tokens": EXPECTED_PROMPT_TOKENS[CASE_28K]},
    ]
    if selected:
        wanted = set(selected)
        cases = [c for c in cases if c["name"] in wanted]
    return cases


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("1", "true", "yes", "on"):
        return True
    if isinstance(value, str) and value.lower() in ("0", "false", "no", "off"):
        return False
    raise ValueError("expected a boolean, got %r" % (value,))


def apply_alias(key):
    return JSON_ALIASES.get(key, key)


def split_config_spec(spec):
    """Split 'name=x,kv=f16,extra=--a b --c d' honoring extra= as the tail.

    extra/extra_args must be the last field: everything after it is shlex-split
    into server arguments, so commas inside those arguments stay intact.
    """
    match = re.search(r"(?:^|,)\s*(?:extra|extra_args)\s*=", spec)
    if match:
        head = spec[:match.start()]
        extra_text = spec[match.end():]
        extra_args = shlex.split(extra_text)
        for token in extra_args:
            for fragment in token.split(","):
                if "=" not in fragment:
                    continue
                key = fragment.split("=", 1)[0].strip()
                if key in CLI_KEYS:
                    raise ValueError(
                        "extra= must be the last field in --config; found %r after it"
                        % fragment)
    else:
        head = spec
        extra_args = None
    obj = {}
    if head.strip():
        for part in head.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError("expected key=value in %r" % part)
            key, value = part.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key not in CLI_KEYS:
                raise ValueError("unknown --config key %r (known: %s)"
                                 % (key, ", ".join(sorted(CLI_KEYS))))
            field, caster = CLI_KEYS[key]
            if caster is int:
                try:
                    obj[field] = int(value)
                except ValueError:
                    raise ValueError("key %r expects an integer, got %r" % (key, value))
            else:
                obj[field] = value
    if extra_args is not None:
        obj["extra_args"] = extra_args
    return obj


def normalize_config(raw, opts, index):
    if not isinstance(raw, dict):
        raise ValueError("config #%d is not a JSON object" % index)
    merged = {}
    for key, value in raw.items():
        if key in ("extra", "extra_args"):
            merged["extra_args"] = value
            continue
        merged[apply_alias(key)] = value
    unknown = set(merged) - set(DEFAULTS)
    if unknown:
        raise ValueError("config #%d has unknown keys: %s (known: %s)"
                         % (index, sorted(unknown), sorted(DEFAULTS)))

    cfg = dict(DEFAULTS)
    merged_extra = merged.pop("extra_args", None)
    cfg.update(merged)
    cfg["extra_args"] = merged_extra if merged_extra is not None else []
    cfg["name"] = cfg["name"] or "arm%d" % index
    cfg["name"] = str(cfg["name"])

    def as_int(key):
        try:
            cfg[key] = int(cfg[key])
        except (TypeError, ValueError):
            raise ValueError("config %r: %s must be an integer, got %r"
                             % (cfg["name"], key, cfg[key]))

    if not re.match(r"^[A-Za-z0-9._+-]+$", cfg["name"]):
        raise ValueError("config name %r must match [A-Za-z0-9._+-]+" % cfg["name"])
    for key in ("moe_cache_mib", "batch", "ubatch", "ctx", "ngl", "n_cpu_moe"):
        as_int(key)
    if cfg["runs"] is not None:
        as_int("runs")
        if cfg["runs"] < 1:
            raise ValueError("config %r: runs must be >= 1" % cfg["name"])
    if cfg["port"] is not None:
        as_int("port")
        if not 1 <= cfg["port"] <= 65535:
            raise ValueError("config %r: port out of range" % cfg["name"])
    if cfg["vram_reserve_mib"] is not None:
        as_int("vram_reserve_mib")
        if cfg["vram_reserve_mib"] < 0:
            raise ValueError("config %r: vram_reserve_mib must be >= 0" % cfg["name"])
    if cfg["kv"] not in KV_CHOICES:
        raise ValueError("config %r: kv must be one of %s" % (cfg["name"], KV_CHOICES))
    if cfg["lazy_mode"] not in LAZY_CHOICES:
        raise ValueError("config %r: lazy_mode must be one of %s"
                         % (cfg["name"], LAZY_CHOICES))
    if cfg["fa"] not in FA_CHOICES:
        raise ValueError("config %r: fa must be one of %s" % (cfg["name"], FA_CHOICES))
    if cfg["load_mode"] not in LOAD_MODE_CHOICES:
        raise ValueError("config %r: load_mode must be one of %s"
                         % (cfg["name"], LOAD_MODE_CHOICES))
    if cfg["lazy_mode"] == "on" and cfg["load_mode"] not in ("auto", "mmap", "mmap+mlock"):
        raise ValueError("config %r: --lazy-mode on requires mmap, load_mode is %r"
                         % (cfg["name"], cfg["load_mode"]))
    if cfg["moe_cache_mib"] < -1:
        raise ValueError("config %r: moe_cache_mib must be >= -1" % cfg["name"])
    if cfg["batch"] < 1 or cfg["ubatch"] < 1:
        raise ValueError("config %r: batch/ubatch must be >= 1" % cfg["name"])
    if cfg["ubatch"] > cfg["batch"]:
        raise ValueError("config %r: ubatch (%d) must be <= batch (%d)"
                         % (cfg["name"], cfg["ubatch"], cfg["batch"]))
    if cfg["ctx"] < 2048:
        raise ValueError("config %r: ctx must be >= 2048" % cfg["name"])
    if cfg["ngl"] < -2:
        raise ValueError("config %r: ngl must be >= -2" % cfg["name"])
    if cfg["n_cpu_moe"] < 0:
        raise ValueError("config %r: n_cpu_moe must be >= 0" % cfg["name"])

    extra = cfg["extra_args"]
    if not isinstance(extra, list) or not all(isinstance(a, str) for a in extra):
        raise ValueError("config %r: extra_args must be a list of strings" % cfg["name"])
    for token in extra:
        if token in MANAGED_FLAGS:
            raise ValueError("config %r: extra_args must not override harness flag %r"
                             % (cfg["name"], token))
    cfg["extra_args"] = list(DEFAULT_EXTRA_ARGS) + list(extra)

    env = cfg["env"]
    if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str)
                                           for k, v in env.items()):
        raise ValueError("config %r: env must be an object of string values" % cfg["name"])
    cfg["env"] = dict(env)

    cfg["bin_dir"] = os.path.abspath(cfg["bin_dir"] or opts.bin_dir)
    return cfg


def load_configs(opts):
    raw = []
    if opts.configs_file:
        with open(opts.configs_file) as f:
            data = json.load(f)
        if isinstance(data, dict) and "configs" in data:
            data = data["configs"]
        if not isinstance(data, list):
            raise ValueError("%s must contain a list of configs or {\"configs\": [...]}"
                             % opts.configs_file)
        raw.extend(data)
    for spec in opts.config_json or []:
        raw.append(json.loads(spec))
    for spec in opts.config or []:
        raw.append(split_config_spec(spec))
    if not raw:
        raw = [{"name": "baseline"}]
        say("[config] no --configs/--config given, using a single 'baseline' config")
    configs = [normalize_config(item, opts, i + 1) for i, item in enumerate(raw)]

    seen = set()
    for cfg in configs:
        if cfg["name"] in seen:
            raise ValueError("duplicate config name %r" % cfg["name"])
        seen.add(cfg["name"])
    known = {c["name"] for c in configs}
    for label, value in (("--reference", opts.reference), ("--candidate", opts.candidate)):
        if value and value not in known:
            raise ValueError("%s %r is not one of the configs: %s"
                             % (label, value, sorted(known)))
    return configs


def build_server_args(cfg, opts, port):
    server = os.path.join(cfg["bin_dir"], "llama-server")
    args = [
        server, "-m", opts.model, "--alias", opts.alias,
        "-ngl", str(cfg["ngl"]), "-c", str(cfg["ctx"]),
        "--n-cpu-moe", str(cfg["n_cpu_moe"]),
        "--moe-cache-mib", str(cfg["moe_cache_mib"]),
        "-b", str(cfg["batch"]), "-ub", str(cfg["ubatch"]),
        "-fa", cfg["fa"], "--load-mode", cfg["load_mode"],
        "--lazy-mode", cfg["lazy_mode"],
    ]
    if cfg["vram_reserve_mib"] is not None:
        args += ["--vram-reserve-mib", str(cfg["vram_reserve_mib"])]
    if cfg["kv"] == "q8_0":
        args += ["-ctk", "q8_0", "-ctv", "q8_0"]
    args += ["--host", HOST, "--port", str(port)]
    args += cfg["extra_args"]
    return args


def flags_signature(args):
    """Server args without the binary and the dynamic port (for A/B equivalence)."""
    out = []
    skip = False
    for i, token in enumerate(args[1:]):
        if skip:
            skip = False
            continue
        if token == "--port":
            skip = True
            continue
        out.append(token)
    return out


def build_env(cfg, opts):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = (
        cfg["bin_dir"] + ":/usr/local/cuda/lib64:/usr/lib64-nvidia")
    if opts.gpu_index is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(opts.gpu_index)
    env.update(cfg["env"])
    return env


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------

def run_case(port, case, opts):
    payload = {
        "model": opts.alias,
        "messages": [{"role": "user", "content": case["prompt"]}],
        "max_tokens": case["max_tokens"],
        "temperature": 0,
        "stream": False,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.monotonic()
    data = http_post_json(port, "/v1/chat/completions", payload, opts.request_timeout)
    wall = time.monotonic() - t0
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    text = message.get("content") or ""
    timings = data.get("timings") or {}
    prompt_n = timings.get("prompt_n")
    prompt_ms = timings.get("prompt_ms")
    predicted_n = timings.get("predicted_n")
    predicted_ms = timings.get("predicted_ms")
    pp_tps = None
    tg_tps = None
    if prompt_n is not None and prompt_ms:
        pp_tps = round(1000.0 * prompt_n / prompt_ms, 2)
    if predicted_n is not None and predicted_ms:
        tg_tps = round(1000.0 * predicted_n / predicted_ms, 2)
    result = {
        "prompt_n": prompt_n,
        "cache_n": timings.get("cache_n"),
        "prompt_ms": prompt_ms,
        "predicted_n": predicted_n,
        "predicted_ms": predicted_ms,
        "pp_tps": pp_tps,
        "tg_tps": tg_tps,
        "finish_reason": choice.get("finish_reason"),
        "text": text,
        "text_sha256": sha256_text(text),
        "wall_s": round(wall, 2),
    }
    if not timings:
        result["error"] = "response without timings"
    return result


def warmup(port, opts):
    payload = {
        "model": opts.alias,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
        "temperature": 0,
        "stream": False,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.monotonic()
    http_post_json(port, "/v1/chat/completions", payload, opts.request_timeout)
    return round(time.monotonic() - t0, 2)


def run_once(cfg, opts, cases, run_idx, heartbeat):
    name = cfg["name"]
    port = pick_port(cfg["port"] or opts.port_base)
    log_path = os.path.join(opts.log_dir, "%s_run%d.log" % (name, run_idx))
    args = build_server_args(cfg, opts, port)
    env = build_env(cfg, opts)
    run = {
        "run": run_idx,
        "ok": False,
        "complete": False,
        "port": port,
        "log_path": log_path,
        "args": args,
        "load_s": None,
        "warmup_s": None,
        "vram_baseline_mib": None,
        "vram_ready_mib": None,
        "vram_peak_mib": None,
        "vram_ready_delta_mib": None,
        "vram_peak_delta_mib": None,
        "cases": {},
        "errors": [],
        "warnings": [],
        "moe_cache": None,
        "server_exit": None,
        "log_tail": None,
    }
    run["vram_baseline_mib"], _ = query_vram(opts.gpu_index)

    say("[%s] run %d: starting %s" % (name, run_idx, " ".join(shlex.quote(a) for a in args)))
    child = None
    log_f = None
    sampler = None
    try:
        log_f = open(log_path, "wb", buffering=0)
        child = subprocess.Popen(args, env=env, stdout=log_f,
                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 start_new_session=True)
        if HAS_NVSMI:
            sampler = VramSampler(opts.gpu_index)
            sampler.start()

        heartbeat.set_phase("%s run %d: loading" % (name, run_idx))
        health = wait_health(port, child, opts.load_timeout)
        run["load_s"] = health["elapsed_s"]
        if not health["ok"]:
            run["errors"].append(health["error"])
            say("[%s] run %d: LOAD FAILED after %.0fs: %s"
                % (name, run_idx, health["elapsed_s"], health["error"]))
        else:
            run["ok"] = True
            run["vram_ready_mib"], _ = query_vram(opts.gpu_index)
            if run["vram_ready_mib"] is not None and run["vram_baseline_mib"] is not None:
                run["vram_ready_delta_mib"] = run["vram_ready_mib"] - run["vram_baseline_mib"]
            say("[%s] run %d: ready in %.1fs, vram %s MiB (baseline %s MiB)"
                % (name, run_idx, health["elapsed_s"],
                   fmt(run["vram_ready_mib"], "%.0f"),
                   fmt(run["vram_baseline_mib"], "%.0f")))

            for _ in range(opts.warmup_requests):
                heartbeat.set_phase("%s run %d: warmup" % (name, run_idx))
                try:
                    run["warmup_s"] = round(
                        (run["warmup_s"] or 0.0) + warmup(port, opts), 2)
                except Exception as exc:
                    run["warnings"].append("warmup: %s" % exc)
                    say("[%s] run %d: warmup failed: %s" % (name, run_idx, exc))
                    break

            for case in cases:
                cname = case["name"]
                heartbeat.set_phase("%s run %d: case %s" % (name, run_idx, cname))
                try:
                    case_result = run_case(port, case, opts)
                    run["cases"][cname] = case_result
                    if "error" in case_result:
                        run["errors"].append("%s: %s" % (cname, case_result["error"]))
                    say("[%s] run %d: %-10s pp=%s t/s tg=%s t/s prompt_n=%s cache_n=%s "
                        "pred=%s wall=%.1fs"
                        % (name, run_idx, cname,
                           fmt(case_result.get("pp_tps"), "%.1f"),
                           fmt(case_result.get("tg_tps"), "%.2f"),
                           fmt(case_result.get("prompt_n"), "%.0f"),
                           fmt(case_result.get("cache_n"), "%.0f"),
                           fmt(case_result.get("predicted_n"), "%.0f"),
                           case_result.get("wall_s") or 0.0))
                except Exception as exc:
                    run["cases"][cname] = {"error": str(exc)}
                    run["errors"].append("%s: %s" % (cname, exc))
                    say("[%s] run %d: %s FAILED: %s" % (name, run_idx, cname, exc))
                    if child.poll() is not None:
                        run["errors"].append("server exited during cases")
                        say("[%s] run %d: server exited, skipping remaining cases"
                            % (name, run_idx))
                        break

            cached = {cname: c.get("cache_n") for cname, c in run["cases"].items()
                      if (c.get("cache_n") or 0) > 0}
            if cached:
                run["warnings"].append(
                    "cache_prompt=false but cache_n > 0: %s" % cached)
            for case in cases:
                expected = case.get("calibrated_prompt_tokens")
                got = (run["cases"].get(case["name"]) or {}).get("prompt_n")
                if expected and got:
                    delta = 100.0 * (float(got) / expected - 1.0)
                    if abs(delta) >= 5.0:
                        run["warnings"].append(
                            "%s: prompt_n=%s vs calibrated %s (%+.1f%%)"
                            % (case["name"], got, expected, delta))
    except Exception as exc:
        run["errors"].append("runner: %s" % exc)
        say("[%s] run %d: RUNNER ERROR: %s" % (name, run_idx, exc))
    finally:
        if sampler is not None:
            sampler.stop()
            run["vram_peak_mib"] = sampler.peak
            if (sampler.peak is not None
                    and run["vram_baseline_mib"] is not None):
                run["vram_peak_delta_mib"] = sampler.peak - run["vram_baseline_mib"]
        heartbeat.set_phase("%s run %d: stopping server" % (name, run_idx))
        run["server_exit"] = stop_server(child, opts.shutdown_timeout)
        if run["ok"] and run["server_exit"]["graceful"] is False:
            run["warnings"].append(
                "server did not exit on SIGTERM; llama_moe_cache stats may be missing")
        if log_f is not None:
            try:
                log_f.close()
            except Exception:
                pass
        wait_port_free(port, 10.0)

        text = ""
        if os.path.isfile(log_path):
            try:
                with open(log_path, "r", errors="replace") as f:
                    text = f.read()
            except Exception as exc:
                run["errors"].append("log read: %s" % exc)
        run["moe_cache"] = parse_moe_cache_log(text)
        summary = run["moe_cache"]["summary"]
        if summary.get("small_hit_rate_pct") is not None:
            say("[%s] run %d: moe small hit=%s%% (%s/%s) large hit=%s%% uploaded=%s MiB "
                "exit=%s graceful=%s"
                % (name, run_idx,
                   fmt(summary.get("small_hit_rate_pct"), "%.2f"),
                   fmt(summary.get("small_hits"), "%.0f"),
                   fmt(summary.get("small_misses"), "%.0f"),
                   fmt(summary.get("large_hit_rate_pct"), "%.2f"),
                   fmt(summary.get("uploaded_mib_total"), "%.1f"),
                   run["server_exit"]["exit_code"], run["server_exit"]["graceful"]))
        elif not run["ok"]:
            run["log_tail"] = tail_text(text)
        if run["ok"] and run["errors"]:
            run["log_tail"] = tail_text(text)

    run["complete"] = run["ok"] and not run["errors"] and len(run["cases"]) == len(cases)
    return run


def aggregate_runs(runs, cases):
    ok_runs = [r for r in runs if r.get("ok")]
    agg = {
        "runs_total": len(runs),
        "runs_loaded": len(ok_runs),
        "runs_complete": sum(1 for r in runs if r.get("complete")),
        "load_s": None,
        "vram_ready_mib": None,
        "vram_ready_delta_mib": None,
        "vram_peak_mib": None,
        "vram_peak_delta_mib": None,
        "cases": {},
        "moe": summarize_moe_runs(runs),
    }
    loads = [r.get("load_s") for r in ok_runs]
    if loads:
        agg["load_s"] = {"median": round(median(loads), 1),
                         "min": round(min(loads), 1),
                         "max": round(max(loads), 1),
                         "runs": [round(v, 1) for v in loads]}
    ready = [r.get("vram_ready_delta_mib") for r in ok_runs]
    if any(v is not None for v in ready):
        agg["vram_ready_delta_mib"] = round_or_none(median(ready), 0)
    ready_abs = [r.get("vram_ready_mib") for r in ok_runs]
    if any(v is not None for v in ready_abs):
        agg["vram_ready_mib"] = round_or_none(median(ready_abs), 0)
    peaks = [r.get("vram_peak_mib") for r in ok_runs]
    if any(v is not None for v in peaks):
        agg["vram_peak_mib"] = round_or_none(max(peaks), 0)
    peak_deltas = [r.get("vram_peak_delta_mib") for r in ok_runs]
    if any(v is not None for v in peak_deltas):
        agg["vram_peak_delta_mib"] = round_or_none(max(peak_deltas), 0)

    for case in cases:
        cname = case["name"]
        entries = [r["cases"].get(cname) for r in ok_runs]
        entries = [e for e in entries if e]
        good = [e for e in entries if e.get("pp_tps") is not None]
        good_tg = [e for e in entries if e.get("tg_tps") is not None]
        hashes = sorted({e["text_sha256"] for e in entries if e.get("text_sha256")})
        agg["cases"][cname] = {
            "runs_ok": len(entries),
            "runs_with_timing": len(good),
            "pp_tps_runs": [e["pp_tps"] for e in good],
            "tg_tps_runs": [e["tg_tps"] for e in good_tg],
            "pp_tps_median": round_or_none(median([e["pp_tps"] for e in good]), 2),
            "tg_tps_median": round_or_none(median([e["tg_tps"] for e in good_tg]), 2),
            "prompt_n_runs": [e["prompt_n"] for e in good if e.get("prompt_n") is not None],
            "prompt_n_median": round_or_none(
                median([e["prompt_n"] for e in good if e.get("prompt_n") is not None]), 0),
            "cache_n_max": max([e["cache_n"] for e in entries
                                if e.get("cache_n") is not None], default=None),
            "predicted_n_median": round_or_none(
                median([e["predicted_n"] for e in good_tg
                        if e.get("predicted_n") is not None]), 0),
            "text_sha256": hashes,
            "errors": [e.get("error") for e in entries if e.get("error")],
        }
    return agg


def run_config(cfg, opts, cases, heartbeat):
    entry = {
        "name": cfg["name"],
        "config": cfg,
        "bin_dir": cfg["bin_dir"],
        "flags_signature": flags_signature(build_server_args(cfg, opts, 1)),
        "runs": [],
        "aggregate": None,
        "errors": [],
        "warnings": [],
    }
    runs = cfg["runs"] if cfg["runs"] is not None else opts.runs
    for run_idx in range(1, runs + 1):
        heartbeat.set_phase("%s run %d/%d" % (cfg["name"], run_idx, runs))
        try:
            run = run_once(cfg, opts, cases, run_idx, heartbeat)
        except Exception as exc:
            run = {"run": run_idx, "ok": False, "complete": False, "cases": {},
                   "errors": ["runner: %s" % exc], "warnings": [],
                   "log_path": None,
                   "vram_ready_mib": None, "vram_peak_mib": None,
                   "vram_ready_delta_mib": None, "vram_peak_delta_mib": None,
                   "moe_cache": None, "load_s": None}
            say("[%s] run %d: unhandled runner error: %s" % (cfg["name"], run_idx, exc))
        entry["runs"].append(run)
        entry["errors"].extend(run.get("errors") or [])
        entry["warnings"].extend(run.get("warnings") or [])
    entry["aggregate"] = aggregate_runs(entry["runs"], cases)
    return entry


# ---------------------------------------------------------------------------
# aggregation / comparison / report
# ---------------------------------------------------------------------------

def build_determinism(results, cases):
    out = {"per_case": {}}
    for case in cases:
        cname = case["name"]
        groups = {}
        by_cfg = {}
        for entry in results:
            hashes = []
            for run in entry.get("runs") or []:
                if not run.get("ok"):
                    continue
                case_result = (run.get("cases") or {}).get(cname) or {}
                if case_result.get("text_sha256"):
                    hashes.append(case_result["text_sha256"])
            by_cfg[entry["name"]] = {
                "hashes": sorted(set(hashes)),
                "run_to_run_identical": len(set(hashes)) <= 1,
                "runs_with_text": len(hashes),
            }
            for digest in set(hashes):
                groups.setdefault(digest[:12], []).append(entry["name"])
        clean_groups = {k: sorted(set(v)) for k, v in groups.items()}
        out["per_case"][cname] = {
            "groups": clean_groups,
            "by_config": by_cfg,
            "all_configs_identical": len(clean_groups) == 1,
        }
    return out


def find_entry(results, name):
    for entry in results:
        if entry["name"] == name:
            return entry
    return None


def agg_case_tps(entry, case_name, metric):
    agg = (entry.get("aggregate") or {}).get("cases") or {}
    node = agg.get(case_name) or {}
    return node.get(metric)


def build_comparison(results, cases, reference, candidate):
    if not reference:
        return None
    ref = find_entry(results, reference)
    if ref is None or ref.get("aggregate") is None:
        return {"reference": reference, "error": "reference config has no aggregate"}
    comp = {
        "reference": reference,
        "candidate": candidate,
        "cases": {},
        "acceptance": None,
    }
    for case in cases:
        cname = case["name"]
        ref_pp = agg_case_tps(ref, cname, "pp_tps_median")
        ref_tg = agg_case_tps(ref, cname, "tg_tps_median")
        deltas = {}
        for entry in results:
            if entry["name"] == reference:
                continue
            deltas[entry["name"]] = {
                "pp_tps_median": agg_case_tps(entry, cname, "pp_tps_median"),
                "tg_tps_median": agg_case_tps(entry, cname, "tg_tps_median"),
                "pp_delta_pct": round_or_none(
                    pct_delta(agg_case_tps(entry, cname, "pp_tps_median"), ref_pp), 2),
                "tg_delta_pct": round_or_none(
                    pct_delta(agg_case_tps(entry, cname, "tg_tps_median"), ref_tg), 2),
            }
        comp["cases"][cname] = {
            "reference_pp_tps": ref_pp,
            "reference_tg_tps": ref_tg,
            "configs": deltas,
        }
    if candidate:
        cand = find_entry(results, candidate)
        if cand is None or cand.get("aggregate") is None:
            comp["acceptance"] = {
                "reference": reference,
                "candidate": candidate,
                "checks": [],
                "pass": None,
                "note": "candidate has no aggregate yet (partial run)",
            }
        else:
            comp["acceptance"] = build_acceptance(ref, cand, cases, comp)
    return comp


def build_acceptance(ref, cand, cases, comp):
    if ref is None or cand is None:
        return {"reference": None if ref is None else ref.get("name"),
                "candidate": None if cand is None else cand.get("name"),
                "checks": [], "pass": None,
                "note": "reference or candidate missing"}
    checks = []

    same_flags = ref.get("flags_signature") == cand.get("flags_signature")
    per_case = {}
    worst = {"abs_pct": None, "case": None, "metric": None}
    for case in cases:
        cname = case["name"]
        node = comp["cases"].get(cname) or {}
        deltas = (node.get("configs") or {}).get(cand["name"]) or {}
        per_case[cname] = {"pp_delta_pct": deltas.get("pp_delta_pct"),
                           "tg_delta_pct": deltas.get("tg_delta_pct")}
        for metric in ("pp_delta_pct", "tg_delta_pct"):
            value = deltas.get(metric)
            if value is None:
                continue
            if worst["abs_pct"] is None or abs(value) > worst["abs_pct"]:
                worst = {"abs_pct": abs(value), "case": cname, "metric": metric}
    baseline_pass = None
    baseline_detail = "not applicable: candidate and reference flags differ"
    if same_flags:
        if worst["abs_pct"] is None:
            baseline_detail = "no comparable throughput data"
        else:
            baseline_pass = worst["abs_pct"] <= 5.0
            baseline_detail = "worst |delta| = %s%% on %s/%s (limit 5%%)" % (
                fmt(worst["abs_pct"], "%.2f"), worst["case"], worst["metric"])
    checks.append({
        "id": "phase0_baseline_within_5pct",
        "pass": baseline_pass,
        "detail": baseline_detail,
        "per_case": per_case,
        "same_flags": same_flags,
    })

    long_pp = per_case.get(CASE_28K, {}).get("pp_delta_pct")
    checks.append({
        "id": "phase1_prefill_long_gain_ge_25pct",
        "pass": None if long_pp is None else (long_pp >= 25.0),
        "value_pct": long_pp,
        "target_pct": 25.0,
        "detail": "candidate %s prefill delta vs %s (case %s)%s"
                  % (cand["name"], ref["name"], CASE_28K,
                     " - case not measured" if long_pp is None else ""),
    })

    long_tg = agg_case_tps(cand, CASE_28K_DEC, "tg_tps_median")
    checks.append({
        "id": "phase1_decode_long_ge_55tps",
        "pass": None if long_tg is None else (long_tg >= 55.0),
        "value_tps": long_tg,
        "target_tps": 55.0,
        "detail": "decode after the ~28.6k prompt (case %s)%s"
                  % (CASE_28K_DEC, " - case not measured" if long_tg is None else ""),
    })

    short_tg = per_case.get(CASE_SHORT, {}).get("tg_delta_pct")
    checks.append({
        "id": "phase1_short_regression_le_3pct",
        "pass": None if short_tg is None else (short_tg >= -3.0),
        "value_pct": short_tg,
        "target_pct": -3.0,
        "detail": "short decode delta vs %s (case %s)%s"
                  % (ref["name"], CASE_SHORT,
                     " - case not measured" if short_tg is None else ""),
    })

    evaluated = [c for c in checks if c["pass"] is not None]
    return {
        "reference": ref["name"],
        "candidate": cand["name"],
        "checks": checks,
        "pass": all(c["pass"] for c in evaluated) if evaluated else None,
    }


def finalize(payload, cases):
    results = payload["configs"]
    payload["determinism"] = build_determinism(results, cases)
    payload["comparison"] = build_comparison(
        results, cases, payload["protocol"].get("reference"),
        payload["protocol"].get("candidate"))
    return payload


def render_markdown(payload):
    proto = payload["protocol"]
    lines = []
    lines.append("# parity-bench")
    lines.append("")
    lines.append("- created: %s  duration: %ss" % (
        payload.get("created_utc"),
        fmt(payload.get("duration_s"), "%.0f")))
    host = payload.get("host") or {}
    gpus = host.get("gpus") or {}
    if gpus.get("available"):
        lines.append("- gpu: %s" % "; ".join(
            "%s (%s MiB, driver %s)" % (d["name"], d["total_mib"], d["driver"])
            for d in gpus["devices"]))
    else:
        lines.append("- gpu: n/a (%s)" % gpus.get("reason"))
    lines.append("- model: `%s`" % proto.get("model"))
    lines.append("- cases: %s" % ", ".join(
        "%s(max_tokens=%d)" % (c["name"], c["max_tokens"]) for c in proto["cases"]))
    lines.append("")

    reference = proto.get("reference")
    lines.append("## Throughput (median of runs)")
    lines.append("")
    headers = ["config", "case", "prompt_n", "cache_n", "pp t/s", "tg t/s", "pred_n"]
    if reference:
        headers += ["d_pp% vs ref", "d_tg% vs ref"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for entry in payload["configs"]:
        agg = entry.get("aggregate") or {}
        for case in proto["cases"]:
            cname = case["name"]
            node = (agg.get("cases") or {}).get(cname) or {}
            row = [
                entry["name"], cname,
                fmt(node.get("prompt_n_median"), "%.0f"),
                fmt(node.get("cache_n_max"), "%.0f"),
                fmt(node.get("pp_tps_median"), "%.1f"),
                fmt(node.get("tg_tps_median"), "%.2f"),
                fmt(node.get("predicted_n_median"), "%.0f"),
            ]
            if reference:
                if entry["name"] == reference:
                    row += ["-", "-"]
                else:
                    delta = ((payload.get("comparison") or {})
                             .get("cases", {}).get(cname, {})
                             .get("configs", {}).get(entry["name"], {}))
                    row += [fmt(delta.get("pp_delta_pct"), "%+.2f"),
                            fmt(delta.get("tg_delta_pct"), "%+.2f")]
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Load, VRAM and MoE cache")
    lines.append("")
    headers = ["config", "runs ok", "load s (med)", "vram ready MiB", "vram peak MiB",
               "MoE small hit%", "MoE large hit%", "uploaded MiB/run", "MoE cache plan"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for entry in payload["configs"]:
        agg = entry.get("aggregate") or {}
        moe = agg.get("moe") or {}
        load = agg.get("load_s") or {}
        plan = ""
        if moe.get("auto"):
            auto = moe["auto"][0]
            if auto.get("kind") == "sized":
                plan = "auto %.0f MiB (%.1f%% of host)" % (
                    auto.get("budget_mib") or 0.0, auto.get("cached_pct") or 0.0)
            elif auto.get("kind") == "disabled_low_budget":
                plan = "auto off (low budget)"
            else:
                plan = "auto off (%s)" % auto.get("reason", "disabled")
        elif moe.get("cache_size"):
            plan = "fixed %.0f MiB" % (moe["cache_size"][0].get("cache_mib") or 0.0)
        lines.append("| " + " | ".join([
            entry["name"],
            "%d/%d" % (agg.get("runs_loaded") or 0, agg.get("runs_total") or 0),
            fmt(load.get("median"), "%.1f"),
            fmt(agg.get("vram_ready_mib"), "%.0f"),
            fmt(agg.get("vram_peak_mib"), "%.0f"),
            fmt(moe.get("small_hit_rate_pct"), "%.2f"),
            fmt(moe.get("large_hit_rate_pct"), "%.2f"),
            fmt(moe.get("uploaded_mib_per_run"), "%.1f"),
            plan,
        ]) + " |")
    lines.append("")

    det = payload.get("determinism") or {}
    if det.get("per_case"):
        lines.append("## Determinism (sha256 of the generated text)")
        lines.append("")
        for case in proto["cases"]:
            cname = case["name"]
            node = det["per_case"].get(cname) or {}
            groups = node.get("groups") or {}
            bad_runs = [name for name, info in (node.get("by_config") or {}).items()
                        if not info.get("run_to_run_identical")]
            if not groups:
                lines.append("- %s: no text captured" % cname)
            elif len(groups) == 1:
                digest = next(iter(groups))
                lines.append("- %s: identical across all configs (%s)" % (cname, digest))
            else:
                parts = ["%s: %s" % (digest, ", ".join(names))
                         for digest, names in sorted(groups.items())]
                lines.append("- %s: %d groups -> %s" % (cname, len(groups), "; ".join(parts)))
            if bad_runs:
                lines.append("  - run-to-run mismatch: %s" % ", ".join(sorted(bad_runs)))
        lines.append("")

    comp = payload.get("comparison")
    if comp and comp.get("acceptance"):
        acc = comp["acceptance"]
        lines.append("## Acceptance (%s vs %s)" % (acc["candidate"], acc["reference"]))
        lines.append("")
        for check in acc["checks"]:
            state = "PASS" if check["pass"] else ("N/A" if check["pass"] is None else "FAIL")
            lines.append("- **%s**: %s - %s" % (check["id"], state, check["detail"]))
        lines.append("")
        lines.append("overall: **%s**" % ("PASS" if acc["pass"] else
                                           ("N/A" if acc["pass"] is None else "FAIL")))
        lines.append("")

    problems = []
    for entry in payload["configs"]:
        for err in entry.get("errors") or []:
            problems.append("%s: %s" % (entry["name"], err))
    if problems:
        lines.append("## Errors")
        lines.append("")
        for problem in problems:
            lines.append("- %s" % problem)
        lines.append("")

    warnings = payload.get("warnings") or []
    if warnings:
        lines.append("## Warnings")
        lines.append("")
        for warning in warnings:
            lines.append("- %s" % warning)
        lines.append("")

    if payload.get("log_dir"):
        lines.append("logs: `%s`" % payload["log_dir"])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# execution entry point (also used by ablate.py)
# ---------------------------------------------------------------------------

def host_info(opts):
    return {
        "hostname": socket.gethostname(),
        "gpus": query_gpu_info(),
        "python": sys.version.split()[0],
        "gpu_index": opts.gpu_index,
    }


def build_payload(configs, opts, cases):
    payload = {
        "schema": SCHEMA,
        "created_utc": utc_now(),
        "duration_s": None,
        "host": host_info(opts),
        "log_dir": opts.log_dir,
        "protocol": {
            "bin_dir": os.path.abspath(opts.bin_dir),
            "model": opts.model,
            "alias": opts.alias,
            "runs_default": opts.runs,
            "warmup_requests": opts.warmup_requests,
            "reference": opts.reference,
            "candidate": opts.candidate,
            "ld_library_path": "<bin-dir>:/usr/local/cuda/lib64:/usr/lib64-nvidia",
            "sampling": {
                "temperature": 0,
                "cache_prompt": False,
                "stream": False,
                "enable_thinking": False,
                "note": "enable_thinking is sent as chat_template_kwargs.enable_thinking",
            },
            "server_lifecycle": ("fresh llama-server per run; SIGTERM at the end so the "
                                 "llama_moe_cache destructor flushes hit/miss lines"),
            "cases": [{
                "name": c["name"],
                "max_tokens": c["max_tokens"],
                "prompt_chars": len(c["prompt"]),
                "prompt_sha256": sha256_text(c["prompt"]),
                "calibrated_prompt_tokens": c["calibrated_prompt_tokens"],
            } for c in cases],
        },
        "configs": [],
        "warnings": [],
    }
    return payload


def execute(configs, opts, cases, out_path=None):
    payload = build_payload(configs, opts, cases)
    start = time.monotonic()
    heartbeat = Heartbeat()
    heartbeat.start()
    try:
        total = len(configs)
        for idx, cfg in enumerate(configs, 1):
            heartbeat.set_phase("config %s (%d/%d)" % (cfg["name"], idx, total))
            say("")
            say("=== config %s (%d/%d) %s" % (cfg["name"], idx, total,
                                               "kv=%s cache=%s b=%s ub=%s"
                                               % (cfg["kv"], cfg["moe_cache_mib"],
                                                  cfg["batch"], cfg["ubatch"])))
            try:
                entry = run_config(cfg, opts, cases, heartbeat)
            except Exception as exc:
                entry = {"name": cfg["name"], "config": cfg, "bin_dir": cfg["bin_dir"],
                         "flags_signature": None, "runs": [], "aggregate": None,
                         "errors": ["config crashed: %s" % exc],
                         "warnings": []}
                say("[%s] config crashed: %s" % (cfg["name"], exc))
            payload["configs"].append(entry)
            for warning in entry.get("warnings") or []:
                payload["warnings"].append("%s: %s" % (entry["name"], warning))
            finalize(payload, cases)
            if out_path:
                write_json_atomic(out_path, payload)
    finally:
        heartbeat.stop()
    payload["duration_s"] = round(time.monotonic() - start, 1)
    finalize(payload, cases)
    if out_path:
        write_json_atomic(out_path, payload)
    return payload


def make_opts(args):
    """Resolve output paths and option namespace shared by bench and ablate."""
    out = os.path.abspath(args.out)
    if args.log_dir:
        log_dir = os.path.abspath(args.log_dir)
    else:
        stem = os.path.splitext(os.path.basename(out))[0]
        log_dir = os.path.join(os.path.dirname(out), "logs", stem)
    os.makedirs(log_dir, exist_ok=True)
    args.out = out
    args.log_dir = log_dir
    args.bin_dir = os.path.abspath(args.bin_dir)
    args.model = os.path.abspath(args.model)
    return args


# ---------------------------------------------------------------------------
# dry run / self test
# ---------------------------------------------------------------------------

def dry_run(configs, opts, cases):
    say("=== dry run: no server is started, no GPU is touched ===")
    problems = []
    if not os.path.isfile(opts.model):
        problems.append("model not found: %s" % opts.model)
    say("model: %s" % opts.model)
    say("out:   %s" % opts.out)
    say("logs:  %s" % opts.log_dir)
    gpus = query_gpu_info()
    if gpus.get("available"):
        say("gpu:   %s" % "; ".join("%s (%s MiB)" % (d["name"], d["total_mib"])
                                    for d in gpus["devices"]))
    else:
        say("gpu:   n/a (%s)" % gpus.get("reason"))
    say("cases: %s" % ", ".join("%s/%d out (%d chars)"
                                % (c["name"], c["max_tokens"], len(c["prompt"]))
                                for c in cases))
    say("")
    for cfg in configs:
        server = os.path.join(cfg["bin_dir"], "llama-server")
        runs = cfg["runs"] if cfg["runs"] is not None else opts.runs
        say("[%s] kv=%s moe_cache_mib=%s b=%s ub=%s lazy=%s ncmoe=%s reserve=%s runs=%d"
            % (cfg["name"], cfg["kv"], cfg["moe_cache_mib"], cfg["batch"],
               cfg["ubatch"], cfg["lazy_mode"], cfg["n_cpu_moe"],
               cfg["vram_reserve_mib"], runs))
        if not os.path.isfile(server):
            problems.append("server binary not found: %s" % server)
            say("  binary: MISSING %s" % server)
        else:
            say("  binary: %s (executable=%s)" % (server, os.access(server, os.X_OK)))
        say("  cmd: %s" % " ".join(shlex.quote(a) for a in
                                   build_server_args(cfg, opts, opts.port_base)))
        if cfg["env"]:
            say("  env: %s" % cfg["env"])
        if cfg["moe_cache_mib"] != -1 and cfg["vram_reserve_mib"] is not None:
            say("  note: --vram-reserve-mib only affects the auto cache "
                "(--moe-cache-mib -1)")
        say("")
    if opts.reference:
        say("reference: %s" % opts.reference)
    if opts.candidate:
        say("candidate: %s" % opts.candidate)
    if problems:
        say("problems:")
        for problem in problems:
            say("- %s" % problem)
        say("")
        say("dry run OK (config/JSON valid; %d non-fatal problem(s) above)" % len(problems))
        return 0
    say("dry run OK (config/JSON valid)")
    return 0


def self_test():
    failures = []

    def check(name, condition, detail=""):
        if condition:
            say("PASS %s" % name)
        else:
            say("FAIL %s%s" % (name, (": " + detail) if detail else ""))
            failures.append(name)

    # 1. log parsing against a log with the real line formats.
    with open(SAMPLE_LOG) as f:
        text = f.read()
    parsed = parse_moe_cache_log(text)
    summary = parsed["summary"]
    buckets = {s["bucket"]: s for s in parsed["stats"]}
    check("parse/sample/small-hits", summary["small_hits"] == 372, str(summary))
    check("parse/sample/small-misses", summary["small_misses"] == 169, str(summary))
    check("parse/sample/small-rate", abs((summary["small_hit_rate_pct"] or 0) - 68.76) < 0.01,
          str(summary.get("small_hit_rate_pct")))
    check("parse/sample/large-hits", summary["large_hits"] == 96)
    check("parse/sample/large-misses", summary["large_misses"] == 512)
    check("parse/sample/large-rate", abs((summary["large_hit_rate_pct"] or 0) - 15.79) < 0.01,
          str(summary.get("large_hit_rate_pct")))
    check("parse/sample/uploaded", abs((summary["uploaded_mib_total"] or 0) - 3520.5) < 0.01,
          str(summary.get("uploaded_mib_total")))
    check("parse/sample/bucket-small", "ubatch <= 8" in buckets)
    check("parse/sample/bucket-large", "ubatch > 8" in buckets)
    check("parse/sample/auto-sized",
          parsed["auto"] and parsed["auto"][0]["kind"] == "sized"
          and abs(parsed["auto"][0]["budget_mib"] - 6144.0) < 0.01
          and abs(parsed["auto"][0]["cached_pct"] - 35.3) < 0.01, str(parsed["auto"]))
    check("parse/sample/size",
          parsed["sizes"] and parsed["sizes"][0]["device"] == "CUDA0"
          and abs(parsed["sizes"][0]["cache_mib"] - 6144.0) < 0.01, str(parsed["sizes"]))
    check("parse/sample/plan",
          parsed["plans"] and abs(parsed["plans"][0]["mib"] - 6144.0) < 0.01,
          str(parsed["plans"]))
    check("parse/sample/large-batch-copy",
          parsed["large_batch_copy"] and parsed["large_batch_copy"][0]["experts"] == 96)

    with open(SAMPLE_LOG_DISABLED) as f:
        disabled = parse_moe_cache_log(f.read())
    check("parse/disabled/auto",
          disabled["auto"] and disabled["auto"][0]["kind"] == "disabled_low_budget",
          str(disabled["auto"]))
    check("parse/disabled/no-stats", disabled["stats"] == [])

    check("parse/empty", parse_moe_cache_log("")["summary"]["small_hits"] is None)

    # 2. CLI config parsing (extra must be the tail).
    spec = ("name=t1,kv=q8_0,cache=-1,b=1024,ub=256,lazy=off,ncmoe=3,reserve=2048,"
            "extra=--metrics --no-moe-host-pinned")
    obj = split_config_spec(spec)
    check("cli/all-keys", obj.get("kv") == "q8_0" and obj.get("moe_cache_mib") == -1
          and obj.get("batch") == 1024 and obj.get("ubatch") == 256
          and obj.get("lazy_mode") == "off" and obj.get("n_cpu_moe") == 3
          and obj.get("vram_reserve_mib") == 2048
          and obj.get("extra_args") == ["--metrics", "--no-moe-host-pinned"], str(obj))
    try:
        split_config_spec("kv=q8_0,extra=--metrics,kv=f16")
        check("cli/extra-must-be-last", False, "no exception")
    except ValueError:
        check("cli/extra-must-be-last", True)
    try:
        split_config_spec("bogus=1")
        check("cli/unknown-key", False, "no exception")
    except ValueError:
        check("cli/unknown-key", True)

    class Opts(object):
        bin_dir = "/tmp/bin"
        model = "/tmp/model.gguf"
        alias = "q"
        gpu_index = None
        runs = 2
        reference = None
        candidate = None

    opts = Opts()
    cfg = normalize_config({"name": "ab", "kv": "q8_0", "moe_cache_mib": -1,
                            "vram_reserve_mib": 2048,
                            "extra_args": ["--metrics"]}, opts, 1)
    args = build_server_args(cfg, opts, 8090)
    joined = " ".join(args)
    check("args/kv-q8", "-ctk q8_0 -ctv q8_0" in joined, joined)
    check("args/auto-cache", "--moe-cache-mib -1" in joined, joined)
    check("args/reserve", "--vram-reserve-mib 2048" in joined, joined)
    check("args/extras-last", args[-1] == "--metrics" and "--jinja" in args, joined)
    check("args/no-fa-default", "-fa on" in joined, joined)

    try:
        normalize_config({"name": "bad", "kv": "q4_0"}, opts, 1)
        check("validate/kv", False, "no exception")
    except ValueError:
        check("validate/kv", True)
    try:
        normalize_config({"name": "bad", "load_mode": "none"}, opts, 1)
        check("validate/lazy-requires-mmap", False, "no exception")
    except ValueError:
        check("validate/lazy-requires-mmap", True)
    try:
        normalize_config({"name": "bad", "ubatch": 4096, "batch": 2048}, opts, 1)
        check("validate/ubatch-le-batch", False, "no exception")
    except ValueError:
        check("validate/ubatch-le-batch", True)
    try:
        normalize_config({"extra_args": ["-ub", "1024"]}, opts, 1)
        check("validate/managed-flag", False, "no exception")
    except ValueError:
        check("validate/managed-flag", True)
    try:
        normalize_config({"name": "bad name"}, opts, 1)
        check("validate/name-shape", False, "no exception")
    except ValueError:
        check("validate/name-shape", True)

    example = os.path.join(SCRIPT_DIR, "examples", "configs.example.json")
    try:
        with open(example) as f:
            example_raw = json.load(f)
        example_cfgs = [normalize_config(item, opts, i + 1)
                        for i, item in enumerate(example_raw)]
        check("examples/configs",
              len(example_cfgs) == 3 and example_cfgs[0]["name"] == "base-f16"
              and example_cfgs[2]["kv"] == "q8_0"
              and example_cfgs[2]["moe_cache_mib"] == -1,
              str([c["name"] for c in example_cfgs]))
    except (OSError, ValueError) as exc:
        check("examples/configs", False, str(exc))

    # 3. aggregation math on synthetic runs.
    cases = [{"name": CASE_SHORT, "max_tokens": 128},
             {"name": CASE_28K, "max_tokens": 16},
             {"name": CASE_28K_DEC, "max_tokens": 128}]

    def make_run(idx, pp_short, tg_short, pp_long, tg_long, tg_long_dec):
        return {
            "run": idx, "ok": True, "complete": True, "load_s": 30.0 + idx,
            "vram_ready_mib": 30000 + idx, "vram_peak_mib": 31000 + idx,
            "vram_ready_delta_mib": 29000, "vram_peak_delta_mib": 29500,
            "moe_cache": {"summary": {"small_hits": 10, "small_misses": 10,
                                      "large_hits": 1, "large_misses": 3,
                                      "uploaded_mib_total": 5.0},
                          "auto": [], "sizes": [], "large_batch_copy": []},
            "cases": {
                CASE_SHORT: {"pp_tps": pp_short, "tg_tps": tg_short, "prompt_n": 20,
                             "cache_n": 0, "predicted_n": 128,
                             "text_sha256": "a" * 64},
                CASE_28K: {"pp_tps": pp_long, "tg_tps": tg_long, "prompt_n": 28600,
                           "cache_n": 0, "predicted_n": 16,
                           "text_sha256": "b" * 64},
                CASE_28K_DEC: {"pp_tps": pp_long, "tg_tps": tg_long_dec, "prompt_n": 28620,
                               "cache_n": 0, "predicted_n": 128,
                               "text_sha256": "c" * 64},
            },
            "errors": [],
        }

    runs = [make_run(1, 100.0, 40.0, 1000.0, 70.0, 60.0),
            make_run(2, 110.0, 42.0, 1100.0, 72.0, 62.0)]
    agg = aggregate_runs(runs, cases)
    check("agg/load-median", agg["load_s"]["median"] == 31.5, str(agg["load_s"]))
    check("agg/pp-median", agg["cases"][CASE_SHORT]["pp_tps_median"] == 105.0,
          str(agg["cases"][CASE_SHORT]))
    check("agg/vram-peak-max", agg["vram_peak_mib"] == 31002.0, str(agg["vram_peak_mib"]))
    check("agg/moe-sum", agg["moe"]["small_hit_rate_pct"] == 50.0
          and agg["moe"]["large_hit_rate_pct"] == 25.0, str(agg["moe"]))

    ref = {"name": "ref", "aggregate": agg, "flags_signature": ["-m", "x"],
           "runs": runs, "errors": []}
    cand_agg = aggregate_runs([make_run(1, 105.0, 40.8, 1320.0, 74.0, 56.0)], cases)
    cand = {"name": "cand", "aggregate": cand_agg,
            "flags_signature": ["-m", "x", "-ub", "1024"],
            "runs": [make_run(1, 105.0, 40.8, 1320.0, 74.0, 56.0)], "errors": []}
    same = {"name": "same", "aggregate": aggregate_runs([make_run(1, 101.0, 40.4, 1010.0, 70.7, 60.6)], cases),
            "flags_signature": ["-m", "x"],
            "runs": [make_run(1, 101.0, 40.4, 1010.0, 70.7, 60.6)], "errors": []}
    results = [ref, cand, same]
    comp = build_comparison(results, cases, "ref", "cand")
    checks = {c["id"]: c for c in comp["acceptance"]["checks"]}
    check("compare/delta", comp["cases"][CASE_SHORT]["configs"]["cand"]["pp_delta_pct"]
          == 0.0, str(comp["cases"][CASE_SHORT]))
    check("compare/phase0-non-applicable",
          checks["phase0_baseline_within_5pct"]["pass"] is None
          and checks["phase0_baseline_within_5pct"]["same_flags"] is False)
    check("compare/phase1-prefill", checks["phase1_prefill_long_gain_ge_25pct"]["pass"]
          is True, str(checks["phase1_prefill_long_gain_ge_25pct"]))
    check("compare/phase1-decode", checks["phase1_decode_long_ge_55tps"]["pass"] is True)
    check("compare/phase1-short", checks["phase1_short_regression_le_3pct"]["pass"]
          is True)
    same_comp = build_comparison(results, cases, "ref", "same")
    same_checks = {c["id"]: c for c in same_comp["acceptance"]["checks"]}
    check("compare/phase0-pass",
          same_checks["phase0_baseline_within_5pct"]["pass"] is True,
          str(same_checks["phase0_baseline_within_5pct"]))

    det = build_determinism(results[:2], cases)
    check("determinism/identical",
          det["per_case"][CASE_SHORT]["all_configs_identical"] is True,
          str(det["per_case"][CASE_SHORT]))

    # 4. markdown rendering does not crash and carries the key tables.
    payload = {
        "created_utc": utc_now(), "duration_s": 1.0,
        "host": {"gpus": {"available": False, "reason": "test"}},
        "log_dir": "/tmp/logs",
        "protocol": {"model": "m", "reference": "ref", "candidate": "cand",
                     "cases": [{"name": CASE_SHORT, "max_tokens": 128},
                               {"name": CASE_28K_DEC, "max_tokens": 128}]},
        "configs": [ref, cand],
        "determinism": det,
        "comparison": comp,
        "warnings": [],
    }
    markdown = render_markdown(payload)
    check("markdown/tables", "| config | case |" in markdown
          and "## Acceptance" in markdown, markdown[:200])

    say("")
    if failures:
        say("self-test FAILED: %d check(s)" % len(failures))
        return 1
    say("self-test OK")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser():
    p = argparse.ArgumentParser(
        description="A100 harness for phases 0/1 of the Strata parity plan.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  bench.py --bin-dir /content/new/bin --model model.gguf --out out.json \\\n"
            "      --config 'name=new-f16,kv=f16,cache=6144,b=2048,ub=512'\n"
            "  bench.py --bin-dir /content/new/bin --model model.gguf --out out.json \\\n"
            "      --configs configs.json --reference base-f16 --candidate new-f16\n"
            "  bench.py --dry-run --bin-dir /content/new/bin --model model.gguf --out out.json\n"
            "  bench.py --self-test\n"))
    p.add_argument("--bin-dir", help="directory containing llama-server and its libs")
    p.add_argument("--model", help="shard 1 of the GGUF model")
    p.add_argument("--out", help="output JSON path (logs go to <out-dir>/logs/<stem>/)")
    p.add_argument("--log-dir", help="override the server log directory")
    p.add_argument("--configs", dest="configs_file", metavar="FILE",
                   help="JSON file with a list of configs (or {\"configs\": [...]})")
    p.add_argument("--config-json", action="append", metavar="JSON",
                   help="inline JSON config object (repeatable)")
    p.add_argument("--config", action="append", metavar="SPEC",
                   help="inline config spec name=x,kv=...,cache=...,b=...,ub=...,"
                        "extra=<server flags, must be last> (repeatable)")
    p.add_argument("--runs", type=int, default=2, help="measured repetitions per config")
    p.add_argument("--cases", help="comma-separated subset of cases: %s"
                   % ",".join(CASE_NAMES))
    p.add_argument("--warmup-requests", type=int, default=1,
                   help="discarded short requests before the cases (default 1)")
    p.add_argument("--port-base", type=int, default=8090,
                   help="first port to try (the harness walks up if busy)")
    p.add_argument("--gpu-index", type=int, default=None,
                   help="physical GPU index for nvidia-smi and CUDA_VISIBLE_DEVICES")
    p.add_argument("--load-timeout", type=float, default=360.0,
                   help="seconds to wait for /health (default 360)")
    p.add_argument("--request-timeout", type=float, default=1800.0,
                   help="seconds per completion request (default 1800)")
    p.add_argument("--shutdown-timeout", type=float, default=60.0,
                   help="seconds to wait for SIGTERM before SIGKILL (default 60)")
    p.add_argument("--alias", default=ALIAS, help="model alias sent in requests")
    p.add_argument("--reference", help="config name used as the A/B reference")
    p.add_argument("--candidate", help="config name used for the acceptance checks "
                                      "(default: first config that is not the reference)")
    p.add_argument("--dry-run", action="store_true",
                   help="validate args/configs and print commands without starting a server")
    p.add_argument("--self-test", action="store_true",
                   help="run parser/math/CLI tests against the bundled sample log (no GPU)")
    return p


def main(argv=None):
    parser = build_argparser()
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    missing = [flag for flag in ("bin_dir", "model", "out") if not getattr(args, flag)]
    if missing:
        parser.error("missing required argument(s): %s"
                     % ", ".join("--" + m.replace("_", "-") for m in missing))
    if args.runs < 1:
        parser.error("--runs must be >= 1")
    if args.warmup_requests < 0:
        parser.error("--warmup-requests must be >= 0")

    selected = None
    if args.cases:
        selected = [c.strip() for c in args.cases.split(",") if c.strip()]
        unknown = [c for c in selected if c not in CASE_NAMES]
        if unknown:
            parser.error("unknown --cases value(s): %s (known: %s)"
                         % (unknown, ", ".join(CASE_NAMES)))
    cases = build_cases(selected)

    try:
        args = make_opts(args)
        configs = load_configs(args)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    if args.reference and not args.candidate:
        others = [c["name"] for c in configs if c["name"] != args.reference]
        if others:
            args.candidate = others[0]

    if args.dry_run:
        return dry_run(configs, args, cases)

    if not os.path.isfile(args.model):
        parser.error("model not found: %s" % args.model)
    for cfg in configs:
        server = os.path.join(cfg["bin_dir"], "llama-server")
        if not os.path.isfile(server):
            parser.error("llama-server not found in %s" % cfg["bin_dir"])

    say("parity-bench: %d config(s), %d run(s) each, %d case(s)"
        % (len(configs), args.runs, len(cases)))
    say("out: %s" % args.out)
    payload = execute(configs, args, cases, out_path=args.out)
    say("")
    say(render_markdown(payload))
    say("")
    say("[done] json: %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

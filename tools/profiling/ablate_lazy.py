#!/usr/bin/env python3
"""
Fase 0 do docs/PLAN-strata-parity.md: A/B de --lazy-mode on|off no prefill.

O que este script mede
----------------------
Para cada modo (on/off) e cada braco (cold/warm):

  cold = pagina do arquivo GGUF fora do page cache (drop_caches), servidor novo,
         primeiro prefill do prompt de ~28.6k tokens com cache_prompt=false.
  warm = mesma pagina ja no page cache, servidor novo, mesmo prompt.

O ponto do A/B: com --lazy-mode on o tensor per_layer_token_embd (28.8 GB no
shard 2) e lido sob demanda durante o prefill; com off ele e faulted-in no
load. Se "off" ganhar >20% no prefill cold, existe I/O de PLE material e a
Fase 3 (pipeline do PLE) tem evidencia. Ver README.md.

Este script tambem e o driver HTTP usado por profile_prefill.sh
(subcomando `drive`), para nao duplicar tokenizacao/medicao.

Nao toca em CUDA, nao compila nada e nao gerencia GPU.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Constantes verificadas no codigo em /home/hjotha/beellama.cpp
# (common/arg.cpp e include/llama.h). Nao inventar flags aqui.
# ---------------------------------------------------------------------------

PROD_ARGS_REST = [
    "-ngl", "99",
    "-c", "32768",
    "-b", "2048",
    "-ub", "1024",
    "-fa", "on",
    "--load-mode", "mmap",
    "-ctk", "q8_0",
    "-ctv", "q8_0",
]


def prod_args(ncmoe, moe_cache_mib):
    return ["-ngl", "99", "-ncmoe", str(ncmoe),
            "--moe-cache-mib", str(moe_cache_mib)] + PROD_ARGS_REST[2:]

SERVER_BASE_ARGS = [
    "--host", "127.0.0.1",
    "-np", "1",
    "--no-warmup",
    "--metrics",
    "--no-webui",
    "--alias", "fn40",
]

PROMPT_BLOCK = (
    "def process_rows(rows, cache):\n"
    "    out = []\n"
    "    for row in rows:\n"
    "        if row is None:\n"
    "            continue\n"
    "        key = row[0]\n"
    "        if key in cache:\n"
    "            out.append((key, cache[key]))\n"
    "        else:\n"
    "            total = sum(row[1:])\n"
    "            cache[key] = total\n"
    "            out.append((key, total))\n"
    "    return out\n"
    "\n"
)

PROMPT_SUFFIX = "\nSummarize the code above in one sentence.\n"

GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16", 39: "MXFP4", 40: "NVFP4", 41: "Q1_0", 42: "Q2_0",
    43: "Q6_0", 44: "Q6_1", 45: "Q3_0", 46: "Q3_1", 47: "Q2_0S",
    48: "Q2_1", 49: "F8_E4M3",
}

GGUF_KV_SCALAR = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
    4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
    10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}


# ---------------------------------------------------------------------------
# utilidades
# ---------------------------------------------------------------------------

def eprint(*a):
    print(*a, file=sys.stderr, flush=True)


def die(msg, code=2):
    eprint("ERRO: " + msg)
    sys.exit(code)


def now_ns():
    return int(time.time() * 1e9)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_text_file(path, limit=None):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        data = f.read()
    if limit is not None:
        data = data[-limit:]
    return data


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def which(name):
    return shutil.which(name)


# ---------------------------------------------------------------------------
# leitura do cabecalho GGUF (sem dependencia de gguf-py)
# ---------------------------------------------------------------------------

class _Reader:
    def __init__(self, f):
        self.f = f

    def take(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError("GGUF truncado")
        return b

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def i32(self):
        return struct.unpack("<i", self.take(4))[0]

    def string(self):
        n = self.u64()
        return self.take(n).decode("utf-8", "replace")

    def skip_value(self, vtype, depth=0):
        if depth > 8:
            raise ValueError("array GGUF aninhado demais")
        if vtype in GGUF_KV_SCALAR:
            fmt, size = GGUF_KV_SCALAR[vtype]
            self.take(size)
        elif vtype == 8:  # string
            self.string()
        elif vtype == 9:  # array
            itype = self.u32()
            n = self.u64()
            if itype in GGUF_KV_SCALAR:
                _, size = GGUF_KV_SCALAR[itype]
                self.take(size * n)
            elif itype == 8:
                for _ in range(n):
                    self.string()
            else:
                for _ in range(n):
                    self.skip_value(itype, depth + 1)
        else:
            raise ValueError("tipo GGUF desconhecido: %d" % vtype)

    def value(self, vtype):
        if vtype in GGUF_KV_SCALAR:
            fmt, size = GGUF_KV_SCALAR[vtype]
            return struct.unpack(fmt, self.take(size))[0]
        if vtype == 8:
            return self.string()
        if vtype == 9:
            itype = self.u32()
            n = self.u64()
            out = []
            for _ in range(n):
                out.append(self.value(itype))
            return out
        raise ValueError("tipo GGUF desconhecido: %d" % vtype)


def read_gguf_header(path):
    """Le o cabecalho e calcula o tamanho exato de cada tensor pelos offsets.

    Em GGUF os offsets dos tensores sao relativos ao inicio da secao de dados
    do proprio shard, entao o tamanho e a diferenca para o proximo offset (ou
    o fim do arquivo). Isso evita depender de gguf-py ou da tabela de tipos.
    """
    with open(path, "rb") as f:
        r = _Reader(f)
        if r.take(4) != b"GGUF":
            raise ValueError("nao e GGUF: %s" % path)
        version = r.u32()
        n_tensors = r.u64()
        n_kv = r.u64()
        meta = {}
        for _ in range(n_kv):
            key = r.string()
            vtype = r.u32()
            meta[key] = r.value(vtype)
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            n_dims = r.u32()
            dims = [r.u64() for _ in range(n_dims)]
            ttype = r.u32()
            offset = r.u64()
            tensors.append({
                "name": name,
                "dims": dims,
                "type": ttype,
                "type_name": GGML_TYPE_NAMES.get(ttype, "type_%d" % ttype),
                "offset": offset,
            })
        header_end = f.tell()
        file_size = os.path.getsize(path)

    alignment = int(meta.get("general.alignment", 32) or 32)
    if alignment <= 0:
        alignment = 32
    data_start = (header_end + alignment - 1) // alignment * alignment
    ordered = sorted(tensors, key=lambda t: t["offset"])
    for i, t in enumerate(ordered):
        end = ordered[i + 1]["offset"] if i + 1 < len(ordered) else (file_size - data_start)
        t["nbytes"] = max(0, end - t["offset"])
    return {"version": version, "metadata": meta, "tensors": ordered,
            "data_start": data_start, "file_size": file_size}


def resolve_model_files(model):
    """Aceita o shard 1, qualquer shard, ou o diretorio."""
    if os.path.isdir(model):
        cand = sorted(
            os.path.join(model, n) for n in os.listdir(model)
            if n.endswith(".gguf") and re.search(r"-00001-of-\d+\.gguf$", n)
        )
        if not cand:
            cand = sorted(os.path.join(model, n) for n in os.listdir(model) if n.endswith(".gguf"))
        if not cand:
            raise FileNotFoundError("nenhum .gguf em %s" % model)
        first = cand[0]
    else:
        first = model

    m = re.search(r"-(\d+)-of-(\d+)\.gguf$", os.path.basename(first))
    files = [first]
    if m:
        _idx, total = int(m.group(1)), int(m.group(2))
        prefix = first[: -len(m.group(0))]
        files = ["%s-%05d-of-%05d.gguf" % (prefix, i, total) for i in range(1, total + 1)]
    return files


def model_info(model):
    files = resolve_model_files(model)
    missing = [p for p in files if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("shard(s) ausente(s): " + ", ".join(missing))

    shards = []
    all_tensors = []
    meta = {}
    for p in files:
        h = read_gguf_header(p)
        meta.update(h["metadata"])
        for t in h["tensors"]:
            t["shard"] = p
        all_tensors.extend(h["tensors"])
        shards.append({
            "path": p,
            "bytes": os.path.getsize(p),
            "n_tensors": len(h["tensors"]),
        })

    expert_type = None
    expert_bytes = 0
    per_layer = {}
    ple = None
    count42_other = 0
    bytes42_other = 0

    for t in all_tensors:
        n = t["name"]
        if n.endswith("per_layer_token_embd.weight"):
            ple = {
                "name": n,
                "dims": t["dims"],
                "type": t["type"],
                "type_name": t["type_name"],
                "bytes": t["nbytes"],
                "shard": t["shard"],
            }
            continue

        if "_exps" in n and n.endswith(".weight"):
            if expert_type is None:
                expert_type = t["type"]
            expert_bytes += t["nbytes"]
            m = re.match(r"blk\.(\d+)\.", n)
            if m:
                il = int(m.group(1))
                per_layer[il] = per_layer.get(il, 0) + t["nbytes"]

        if t["type"] == 42 and "_exps" not in n and "_shexp" not in n:
            count42_other += 1
            bytes42_other += t["nbytes"]

    ratios = meta.get("qwen4exp.attention.compress_ratios") or []
    qsa_layers = [i for i, r in enumerate(ratios) if isinstance(r, int) and r > 0]

    return {
        "model_arg": model,
        "files": shards,
        "n_layers": meta.get("qwen4exp.block_count"),
        "n_experts": meta.get("qwen4exp.expert_count"),
        "n_experts_used": meta.get("qwen4exp.expert_used_count"),
        "n_embd": meta.get("qwen4exp.embedding_length"),
        "n_embd_per_layer": meta.get("qwen4exp.embedding_length_per_layer_input"),
        "compress_ratios": ratios,
        "qsa_layers": qsa_layers,
        "gdn_layers": len(ratios) - len(qsa_layers) if ratios else None,
        "expert_type": expert_type,
        "expert_type_name": GGML_TYPE_NAMES.get(expert_type),
        "expert_bytes": expert_bytes,
        "expert_bytes_per_layer": {str(k): v for k, v in sorted(per_layer.items())},
        "ple": ple,
        "type42_non_expert": {"count": count42_other, "bytes": bytes42_other},
    }


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def http_json(method, url, payload=None, timeout=30):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    if not body:
        return None
    return json.loads(body.decode("utf-8"))


def health(port, timeout=2):
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def port_free(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


# ---------------------------------------------------------------------------
# estado do sistema
# ---------------------------------------------------------------------------

def meminfo():
    out = {}
    with open("/proc/meminfo", "r") as f:
        for line in f:
            k, v = line.split(":", 1)
            out[k] = int(v.strip().split()[0])  # kB
    return out


def drop_page_cache(enable=True):
    if not enable:
        return False
    if os.geteuid() != 0:
        return False
    try:
        subprocess.run(["sync"], check=False)
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
        return True
    except Exception as e:
        eprint("AVISO: drop_caches falhou: %s" % e)
        return False


def read_io(pid):
    out = {}
    try:
        for line in read_text_file("/proc/%d/io" % pid).splitlines():
            k, v = line.split(":", 1)
            out[k.strip()] = int(v.strip())
    except Exception:
        pass
    return out


PROC_FIELDS = ["minflt", "cminflt", "majflt", "cmajflt", "utime", "stime",
               "cutime", "cstime", "priority", "nice", "num_threads", "itrealvalue"]


def read_proc(pid):
    """stat + soma de utime/stime de todas as threads (ticks)."""
    out = {"pid": pid}
    try:
        raw = read_text_file("/proc/%d/stat" % pid)
        # o comm pode conter espacos/parênteses
        rparen = raw.rfind(")")
        rest = raw[rparen + 2:].split()
        vals = rest
        # campo 3 (state) = vals[0]; campo 4 (ppid) = vals[1]; ...
        def field(n):
            return vals[n - 3]

        out["state"] = field(3)
        out["minflt"] = int(field(10))
        out["majflt"] = int(field(12))
        out["utime"] = int(field(14))
        out["stime"] = int(field(15))
        out["num_threads"] = int(field(20))
        out["rss_pages"] = int(field(24))
        out["rss_kib"] = out["rss_pages"] * (os.sysconf("SC_PAGE_SIZE") // 1024)
    except Exception as e:
        out["error"] = str(e)

    cpu = 0
    nthreads = 0
    try:
        for tid in os.listdir("/proc/%d/task" % pid):
            try:
                raw = read_text_file("/proc/%d/task/%s/stat" % (pid, tid))
                rest = raw[raw.rfind(")") + 2:].split()
                cpu += int(rest[11]) + int(rest[12])  # utime, stime (campos 14,15)
                nthreads += 1
            except Exception:
                continue
    except Exception:
        pass
    out["cpu_ticks_all_threads"] = cpu
    out["n_threads_scanned"] = nthreads
    return out


def nvidia_smi():
    smi = which("nvidia-smi")
    if not smi:
        return {"available": False}
    q = "name,driver_version,memory.total,memory.used,clocks.sm,clocks.mem,power.draw,temperature.gpu"
    try:
        r = subprocess.run([smi, "--query-gpu=" + q, "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20)
        apps = subprocess.run([smi, "--query-compute-apps=pid,process_name,used_memory",
                               "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=20)
        return {
            "available": True,
            "gpus": [x.strip() for x in r.stdout.strip().splitlines() if x.strip()],
            "compute_apps": [x.strip() for x in apps.stdout.strip().splitlines() if x.strip()],
        }
    except Exception as e:
        return {"available": True, "error": str(e)}


# ---------------------------------------------------------------------------
# prompt e calibracao
# ---------------------------------------------------------------------------

def build_prompt(repeats):
    return PROMPT_BLOCK * repeats + PROMPT_SUFFIX


def count_tokens(port, text, timeout=120):
    r = http_json("POST", "http://127.0.0.1:%d/tokenize" % port,
                  {"content": text, "add_special": True, "parse_special": True},
                  timeout=timeout)
    return len(r["tokens"])


def calibrate_prompt(port, target, tol_frac=0.006, max_iter=8):
    """Ajusta o numero de repeticoes para chegar em target tokens.

    A granularidade e o tamanho do bloco (~120 tokens no modelo real), entao
    uma tolerancia menor que isso pode oscilar entre n e n+1 blocos. O default
    de 0,6% cabe essa granularidade e o numero exato fica registrado.
    """
    per_block = max(1, count_tokens(port, PROMPT_BLOCK))
    repeats = max(1, int(round((target - 16) / float(per_block))))
    text = build_prompt(repeats)
    n = count_tokens(port, text)
    eprint("calibracao: %d repeticoes -> %d tokens (alvo %d)" % (repeats, n, target))

    best = (abs(n - target), repeats, n, text)
    seen = set()
    for _ in range(max_iter):
        tol = max(4, int(target * tol_frac))
        if abs(n - target) <= tol:
            break
        if repeats in seen:
            break
        seen.add(repeats)
        diff = target - n
        step = int(round(diff / float(per_block))) if per_block else 0
        if step == 0:
            step = 1 if diff > 0 else -1
        repeats = max(1, repeats + step)
        text = build_prompt(repeats)
        n = count_tokens(port, text)
        eprint("calibracao: %d repeticoes -> %d tokens (alvo %d)" % (repeats, n, target))
        if abs(n - target) < best[0]:
            best = (abs(n - target), repeats, n, text)

    tol = max(4, int(target * tol_frac))
    if abs(n - target) > tol:
        if best[0] <= max(tol * 2, int(target * 0.02)):
            eprint("AVISO: melhor ajuste possivel = %d tokens (alvo %d, +- %d); usando" % (
                best[2], target, best[0]))
            repeats, n, text = best[1], best[2], best[3]
        else:
            raise RuntimeError("nao consegui calibrar o prompt: %d tokens (alvo %d +- %d)" % (
                n, target, tol))

    return {
        "text": text,
        "repeats": repeats,
        "tokens": n,
        "target": target,
        "sha256": sha256_text(text),
        "chars": len(text),
    }


# ---------------------------------------------------------------------------
# servidor
# ---------------------------------------------------------------------------

class Server(object):
    def __init__(self, args, mode, log_path, extra_env=None, nsys_prefix=None):
        self.args = args
        self.mode = mode
        self.log_path = log_path
        self.extra_env = extra_env or {}
        self.nsys_prefix = nsys_prefix or []
        self.proc = None
        self.t0 = 0.0

    def cmd(self):
        cmd = []
        cmd += self.nsys_prefix
        cmd += [self.args.server, "-m", self.args.model]
        cmd += prod_args(self.args.ncmoe, self.args.moe_cache_mib)
        cmd += ["--lazy-mode", self.mode]
        cmd += SERVER_BASE_ARGS
        cmd += ["--port", str(self.args.port)]
        if self.args.extra_arg:
            for a in self.args.extra_arg:
                cmd += a.split()
        return cmd

    def env(self):
        env = dict(os.environ)
        env["LLAMA_LOG_COLORS"] = "0"
        env.update(self.extra_env)
        return env

    def start(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.logf = open(self.log_path, "w", encoding="utf-8")
        cmd = self.cmd()
        self.logf.write("# CMD: " + " ".join(cmd) + "\n")
        for k, v in sorted(self.extra_env.items()):
            self.logf.write("# ENV: %s=%s\n" % (k, v))
        self.logf.write("#\n")
        self.logf.flush()
        self.t0 = time.time()
        self.proc = subprocess.Popen(
            cmd, stdout=self.logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            env=self.env(), start_new_session=True, cwd=os.path.dirname(self.args.server) or None,
        )
        return self.proc

    def wait_health(self, timeout):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                tail = read_text_file(self.log_path, limit=6000)
                raise RuntimeError("servidor saiu com rc=%s\n%s" % (self.proc.returncode, tail))
            if health(self.args.port):
                return time.time() - self.t0
            time.sleep(2.0)
        raise RuntimeError("timeout de %ds esperando /health" % timeout)

    def stop(self, timeout=120):
        if self.proc is None or self.proc.poll() is not None:
            if self.proc is not None:
                self.logf.flush()
                self.logf.close()
            return 0
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except Exception:
            self.proc.terminate()
        try:
            rc = self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except Exception:
                self.proc.kill()
            rc = self.proc.wait(timeout=30)
        self.logf.flush()
        self.logf.close()
        return rc


# ---------------------------------------------------------------------------
# uma requisicao de prefill
# ---------------------------------------------------------------------------

def run_prefill(port, prompt_text, pid, timeout, n_predict=1):
    before = {
        "epoch_ns": now_ns(),
        "io": read_io(pid) if pid else {},
        "proc": read_proc(pid) if pid else {},
        "smi": nvidia_smi(),
    }
    body = {
        "prompt": prompt_text,
        "n_predict": n_predict,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "min_p": 0.0,
        "seed": 0,
        "cache_prompt": False,
        "stream": False,
        "timings_per_token": False,
    }
    t_wall0 = now_ns()
    err = None
    resp = None
    try:
        resp = http_json("POST", "http://127.0.0.1:%d/completion" % port, body, timeout=timeout)
    except Exception as e:
        err = "%s: %s" % (type(e).__name__, e)
    finally:
        t_wall1 = now_ns()
    after = {
        "epoch_ns": now_ns(),
        "io": read_io(pid) if pid else {},
        "proc": read_proc(pid) if pid else {},
        "smi": nvidia_smi(),
    }

    timings = (resp or {}).get("timings", {}) or {}
    io_delta = {}
    for k in ("rchar", "wchar", "read_bytes", "write_bytes"):
        if k in before["io"] and k in after["io"]:
            io_delta[k] = after["io"][k] - before["io"][k]
    proc_delta = {
        "cpu_ticks": (after["proc"].get("cpu_ticks_all_threads", 0)
                      - before["proc"].get("cpu_ticks_all_threads", 0)),
        "majflt": after["proc"].get("majflt", 0) - before["proc"].get("majflt", 0),
        "minflt": after["proc"].get("minflt", 0) - before["proc"].get("minflt", 0),
    }
    return {
        "ok": resp is not None,
        "error": err,
        "epoch_start_ns": t_wall0,
        "epoch_end_ns": t_wall1,
        "wall_ms": (t_wall1 - t_wall0) / 1e6,
        "timings": timings,
        "n_prompt_tokens": timings.get("prompt_n"),
        "n_cached_tokens": timings.get("cache_n"),
        "prompt_ms": timings.get("prompt_ms"),
        "prompt_per_second": timings.get("prompt_per_second"),
        "predicted_n": timings.get("predicted_n"),
        "io_delta": io_delta,
        "proc_delta": proc_delta,
        "rss_kib_before": before["proc"].get("rss_kib"),
        "rss_kib_after": after["proc"].get("rss_kib"),
        "before": before,
        "after": after,
        "response_head": {k: resp.get(k) for k in ("content", "stop", "model")} if resp else {},
    }


# ---------------------------------------------------------------------------
# subcomando: model-info
# ---------------------------------------------------------------------------

def cmd_model_info(args):
    info = model_info(args.model)
    print(json.dumps(info, indent=2, sort_keys=True))
    return 0


# ---------------------------------------------------------------------------
# subcomando: drive (usado por profile_prefill.sh)
# ---------------------------------------------------------------------------

def cmd_drive(args):
    if args.prompt_file and os.path.exists(args.prompt_file):
        text = read_text_file(args.prompt_file)
        if args.calibrate:
            n = count_tokens(args.port, text)
        else:
            n = None
        meta = {"path": args.prompt_file, "sha256": sha256_text(text),
                "chars": len(text), "tokens": n}
    else:
        if not args.target_tokens:
            die("drive sem --prompt-file exige --target-tokens")
        cal = calibrate_prompt(args.port, args.target_tokens)
        text = cal.pop("text")
        meta = dict(cal)
        if args.prompt_file:
            os.makedirs(os.path.dirname(args.prompt_file), exist_ok=True)
            with open(args.prompt_file, "w", encoding="utf-8") as f:
                f.write(text)
    out = run_prefill(args.port, text, args.server_pid, args.request_timeout)
    out["prompt"] = meta
    if args.server_pid:
        out["server_pid"] = args.server_pid
    write_json(args.out_json, out)

    if not out["ok"]:
        die("requisicao falhou: %s" % out.get("error"))
    if out["n_cached_tokens"] not in (0, None):
        die("o servidor reutilizou %s tokens do cache; o prefill nao foi completo" % out["n_cached_tokens"])
    print("prefill: %s tokens, %.1f ms, %.2f t/s, wall %.1f ms, read_bytes %s" % (
        out["n_prompt_tokens"], out["prompt_ms"] or -1,
        out["prompt_per_second"] or -1, out["wall_ms"],
        out["io_delta"].get("read_bytes", "?")))
    return 0


# ---------------------------------------------------------------------------
# orcamento de VRAM
# ---------------------------------------------------------------------------

def vram_budget(info, ncmoe, cache_mib):
    """(pesos_gpu, cache, necessidade_total, host_experts) em bytes.

    A necessidade usa 4 GiB de margem para KV q8_0 + buffers de compute +
    contexto CUDA. E uma estimativa conservadora; o objetivo e pegar o caso
    obvio (cache fixo grande demais), nao substituir o sched_reserve.
    """
    per = [v for _k, v in sorted(info.get("expert_bytes_per_layer", {}).items(),
                                 key=lambda kv: int(kv[0]))]
    if not per:
        return 0, 0, 0, 0
    expert = sum(per)
    per_layer = per[0] if per else 0
    host = min(ncmoe * per_layer, expert) if per_layer else 0
    total = sum(f["bytes"] for f in info["files"])
    ple = (info.get("ple") or {}).get("bytes", 0)
    gpu_weights = max(0, total - ple - host)
    cache = host if cache_mib < 0 else cache_mib * 1024 * 1024
    margin = 4096 * 1024 * 1024
    return gpu_weights, cache, gpu_weights + cache + margin, host

# ---------------------------------------------------------------------------
# subcomando: ablate
# ---------------------------------------------------------------------------

def fmt_table(rows, cols):
    widths = {c: len(c) for c in cols}
    for r in rows:
        for c in cols:
            widths[c] = max(widths[c], len(str(r.get(c, ""))))
    out = []
    out.append("| " + " | ".join(c.ljust(widths[c]) for c in cols) + " |")
    out.append("| " + " | ".join("-" * widths[c] for c in cols) + " |")
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols) + " |")
    return "\n".join(out)


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    if n % 2:
        return xs[n // 2]
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def cmd_ablate(args):
    info = None
    try:
        info = model_info(args.model)
    except Exception as e:
        eprint("AVISO: nao consegui ler o cabecalho GGUF: %s" % e)
        if not args.dry_run:
            die("modelo invalido: %s" % e)

    ple_bytes = (info or {}).get("ple", {}).get("bytes") or 0
    ple_gib = ple_bytes / (1024.0 ** 3)
    need_gib = max(args.min_free_gib, int(ple_gib + 8) if ple_bytes else 0)

    plan = {
        "server": args.server,
        "model": args.model,
        "modes": args.modes,
        "reps": args.reps,
        "target_tokens": args.target_tokens,
        "drop_caches": args.drop_caches,
        "min_free_gib": need_gib,
        "ple_gib": round(ple_gib, 2),
        "outdir": args.outdir,
        "server_args": prod_args(args.ncmoe, args.moe_cache_mib) + SERVER_BASE_ARGS,
        "ncmoe": args.ncmoe,
        "moe_cache_mib": args.moe_cache_mib,
    }

    if args.dry_run:
        print("=== DRY-RUN ablate_lazy.py ===")
        print(json.dumps(plan, indent=2))
        mi = meminfo()
        avail = mi.get("MemAvailable", 0) / 1024.0 / 1024.0
        print("MemAvailable agora: %.1f GiB (necessario para lazy off: %.1f GiB)" % (avail, need_gib))
        if info:
            gw, cache, need, host = vram_budget(info, args.ncmoe, args.moe_cache_mib)
            print("VRAM: pesos %.0f MiB + cache %.0f MiB + 4 GiB margem = %.0f MiB" % (
                gw / 1048576.0, cache / 1048576.0, need / 1048576.0))
            print("PLE: %.2f GiB (%s)" % (ple_gib, info.get("ple", {}).get("name")))
            print("expert type: %s (id %s)" % (info.get("expert_type_name"), info.get("expert_type")))
            print("expert bytes: %.2f GiB" % (info.get("expert_bytes", 0) / (1024.0 ** 3)))
        print("prompt: ~%d tokens, %d repeticoes do bloco" % (
            args.target_tokens, max(1, args.target_tokens // 120)))
        print("comando por modo:")
        srv = Server(args, "on", os.path.join(args.outdir, "logs", "dry.log"))
        print("  " + " ".join(srv.cmd()))
        print("OK: dry-run (nada foi executado)")
        return 0

    # ---- preflight ----
    args.server = os.path.abspath(args.server)
    args.model = os.path.abspath(args.model)
    if not os.path.isfile(args.server) or not os.access(args.server, os.X_OK):
        die("llama-server nao encontrado/executavel: %s" % args.server)
    if not os.path.exists(args.model):
        die("modelo nao encontrado: %s" % args.model)
    if info is None:
        die("cabecalho GGUF invalido")
    if not port_free(args.port):
        die("porta %d ja esta em uso" % args.port)
    os.makedirs(args.outdir, exist_ok=True)

    smi = nvidia_smi()
    if smi.get("available") and smi.get("compute_apps") and not args.allow_busy_gpu:
        die("GPU ocupada (a A100 pode estar em benchmark):\n  "
            + "\n  ".join(smi["compute_apps"]) + "\nuse --allow-busy-gpu para ignorar")

    if smi.get("available") and smi.get("gpus"):
        fields = [x.strip() for x in smi["gpus"][0].split(",")]
        total_mib = None
        for f in fields:
            if f.endswith("MiB"):
                total_mib = int(f.split()[0])
                break
        if total_mib:
            gw, cache, need, host = vram_budget(info, args.ncmoe, args.moe_cache_mib)
            eprint("VRAM: pesos %.0f + cache %.0f + margem 4096 = %.0f MiB (total %d MiB)" % (
                gw / 1048576.0, cache / 1048576.0, need / 1048576.0, total_mib))
            if need > 0 and need > total_mib * 1024 * 1024:
                die("configuracao de VRAM nao cabe: faltam %.0f MiB. "
                    "Use --moe-cache-mib -1 (auto) ou --ncmoe 8." % (
                        (need - total_mib * 1024 * 1024) / 1048576.0))

    mi = meminfo()
    avail_gib = mi.get("MemAvailable", 0) / 1024.0 / 1024.0
    if "off" in args.modes and mi.get("MemAvailable", 0) < need_gib * 1024 * 1024:
        die("MemAvailable %.1f GiB < %.1f GiB necessarios (PLE %.1f GiB + margem). "
            "Nao rode o braco --lazy-mode off nessa maquina agora; use --modes on." % (
                avail_gib, need_gib, ple_gib))

    do_drop = args.drop_caches == "yes" or (args.drop_caches == "auto" and os.geteuid() == 0)
    if args.drop_caches != "no" and not do_drop:
        eprint("AVISO: sem root, drop_caches nao sera feito; o braco cold pode estar quente.")

    eprint("modelo: %d shards, %d layers, expert type %s (%s)" % (
        len(info["files"]), info.get("n_layers") or -1,
        info.get("expert_type_name"), info.get("expert_type")))
    eprint("PLE: %.2f GiB -> residente no modo off" % ple_gib)

    runs = []
    prompt_meta = None
    prompt_text = None
    prompt_path = os.path.join(args.outdir, "prompt.txt")

    for rep in range(args.reps):
        order = list(args.modes)
        if rep % 2 == 1:
            order = list(reversed(order))
        for mode in order:
            # o braco "cold" precisa das paginas do GGUF/PLE fora do page cache
            # em toda corrida, nao so na primeira: sem isso ele vira "warm".
            cold_pages = drop_page_cache(do_drop)
            eprint("\n== rep %d modo %s (drop_caches=%s) ==" % (rep + 1, mode, cold_pages))

            log_path = os.path.join(args.outdir, "logs", "r%d-%s.log" % (rep + 1, mode))
            srv = Server(args, mode, log_path)
            try:
                srv.start()
                load_s = srv.wait_health(args.load_timeout)
                eprint("carregado em %.1f s" % load_s)
                time.sleep(args.settle)

                if prompt_text is None:
                    if os.path.exists(prompt_path) and not args.rebuild_prompt:
                        prompt_text = read_text_file(prompt_path)
                        n = count_tokens(args.port, prompt_text)
                        prompt_meta = {
                            "path": prompt_path, "sha256": sha256_text(prompt_text),
                            "chars": len(prompt_text), "tokens": n,
                            "target": args.target_tokens, "repeats": None,
                        }
                    else:
                        cal = calibrate_prompt(args.port, args.target_tokens)
                        prompt_text = cal.pop("text")
                        prompt_meta = dict(cal)
                        prompt_meta["path"] = prompt_path
                        with open(prompt_path, "w", encoding="utf-8") as f:
                            f.write(prompt_text)
                        write_json(os.path.join(args.outdir, "prompt_meta.json"), prompt_meta)

                for arm in ("cold", "warm"):
                    if arm == "warm" and not args.warm_arm:
                        continue
                    if arm == "warm":
                        time.sleep(args.settle)
                    r = run_prefill(args.port, prompt_text, srv.proc.pid, args.request_timeout)
                    r.update({
                        "rep": rep + 1, "mode": mode, "arm": arm,
                        "load_s": load_s, "drop_caches": cold_pages,
                        "log": log_path,
                    })
                    if not r["ok"]:
                        die("prefill falhou (rep %d, %s, %s); veja %s" % (rep + 1, mode, arm, log_path))
                    if r["n_cached_tokens"] not in (0, None):
                        die("cache_prompt=false mas cache_n=%s (rep %d, %s, %s)" % (
                            r["n_cached_tokens"], rep + 1, mode, arm))
                    runs.append(r)
                    eprint("  %-4s %-3s: %8.1f ms  %8.2f t/s  read=%.1f MiB  rss=%.1f GiB" % (
                        arm, mode, r["prompt_ms"] or -1, r["prompt_per_second"] or -1,
                        r["io_delta"].get("read_bytes", 0) / 1048576.0,
                        (r["rss_kib_after"] or 0) / 1048576.0))
            finally:
                srv.stop()

    # ---- resumo ----
    def med(mode, arm, field):
        return median([r.get(field) for r in runs if r["mode"] == mode and r["arm"] == arm])

    c_on = med("on", "cold", "prompt_ms")
    c_off = med("off", "cold", "prompt_ms")
    w_on = med("on", "warm", "prompt_ms")
    w_off = med("off", "warm", "prompt_ms")

    d_cold = None if not (c_on and c_off) else (c_on - c_off) / c_on
    d_warm = None if not (w_on and w_off) else (w_on - w_off) / w_on
    warm_gain_on = None if not (c_on and w_on) else (c_on - w_on) / c_on

    if d_cold is not None and d_cold > 0.20:
        verdict = "EVIDENCIA_FASE3"
        reason = ("--lazy-mode off foi %.1f%% mais rapido no prefill cold (>20%%): "
                  "o I/O do PLE pesa e a Fase 3 tem evidencia." % (100 * d_cold))
    elif d_cold is not None and abs(d_cold) < 0.05 and (d_warm is None or abs(d_warm) < 0.05):
        verdict = "SEM_EVIDENCIA"
        reason = ("off vs on dentro de +-5% no cold e no warm: o I/O do PLE nao "
                  "justifica a Fase 3 neste modelo/maquina.")
    else:
        verdict = "INCONCLUSIVO"
        reason = ("off vs on = %.1f%% no cold e %.1f%% no warm: entre 5%% e 20%%, "
                  "repetir com --reps 2 e/ou prompt maior antes de decidir." % (
                      100 * (d_cold or 0), 100 * (d_warm or 0)))

    cols = ["rep", "mode", "arm", "load_s", "prompt_n", "cache_n", "prompt_ms",
            "prompt_per_second", "read_MiB", "majflt", "rss_GiB"]
    table = []
    for r in runs:
        table.append({
            "rep": r["rep"], "mode": r["mode"], "arm": r["arm"],
            "load_s": "%.1f" % (r["load_s"] or 0),
            "prompt_n": r["n_prompt_tokens"],
            "cache_n": r["n_cached_tokens"],
            "prompt_ms": "%.1f" % (r["prompt_ms"] or 0),
            "prompt_per_second": "%.2f" % (r["prompt_per_second"] or 0),
            "read_MiB": "%.0f" % (r["io_delta"].get("read_bytes", 0) / 1048576.0),
            "majflt": r["proc_delta"].get("majflt"),
            "rss_GiB": "%.1f" % ((r["rss_kib_after"] or 0) / 1048576.0),
        })

    result = {
        "plan": plan,
        "model": info,
        "prompt": prompt_meta,
        "runs": runs,
        "medians_ms": {"on_cold": c_on, "off_cold": c_off, "on_warm": w_on, "off_warm": w_off},
        "deltas": {"cold_off_vs_on": d_cold, "warm_off_vs_on": d_warm, "warm_gain_on": warm_gain_on},
        "verdict": verdict,
        "reason": reason,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_json(os.path.join(args.outdir, "run.json"), result)

    csv_cols = ["rep", "mode", "arm", "load_s", "prompt_n", "cache_n", "prompt_ms",
                "prompt_per_second", "wall_ms", "read_bytes", "write_bytes", "majflt",
                "rss_kib_after"]
    with open(os.path.join(args.outdir, "summary.csv"), "w", encoding="utf-8") as f:
        f.write(",".join(csv_cols) + "\n")
        for r in runs:
            f.write(",".join(str(x) for x in [
                r["rep"], r["mode"], r["arm"], "%.2f" % (r["load_s"] or 0),
                r["n_prompt_tokens"], r["n_cached_tokens"],
                "%.2f" % (r["prompt_ms"] or 0), "%.3f" % (r["prompt_per_second"] or 0),
                "%.2f" % r["wall_ms"], r["io_delta"].get("read_bytes", 0),
                r["io_delta"].get("write_bytes", 0), r["proc_delta"].get("majflt", 0),
                r["rss_kib_after"],
            ]) + "\n")

    md = []
    md.append("# A/B --lazy-mode on vs off (Fase 0)\n")
    md.append("Gerado: %s\n" % result["generated_at"])
    md.append("- modelo: `%s`\n" % args.model)
    md.append("- shards: %d, layers: %s, expert type: %s\n" % (
        len(info["files"]), info.get("n_layers") or "?", info.get("expert_type_name")))
    md.append("- PLE: `%s` %.2f GiB\n" % (
        (info.get("ple") or {}).get("name"), ple_gib))
    md.append("- prompt: %s tokens (alvo %s), sha256 `%s`\n" % (
        (prompt_meta or {}).get("tokens"), args.target_tokens,
        ((prompt_meta or {}).get("sha256") or "")[:16]))
    md.append("- flags: `%s`\n" % " ".join(prod_args(args.ncmoe, args.moe_cache_mib)))
    md.append("- cache_prompt=false em todas as requisicoes; cache_n=0 confirmado\n")
    md.append("\n## Corridas\n\n")
    md.append(fmt_table(table, cols))
    md.append("\n\n## Medianas de prefill (ms)\n\n")
    md.append("| | cold | warm |\n|---|---:|---:|\n")
    md.append("| `--lazy-mode on` | %s | %s |\n" % (
        "%.1f" % c_on if c_on else "-", "%.1f" % w_on if w_on else "-"))
    md.append("| `--lazy-mode off` | %s | %s |\n" % (
        "%.1f" % c_off if c_off else "-", "%.1f" % w_off if w_off else "-"))
    md.append("\n- off vs on (cold): %s\n" % (
        "%.1f%%" % (100 * d_cold) if d_cold is not None else "-"))
    md.append("- off vs on (warm): %s\n" % (
        "%.1f%%" % (100 * d_warm) if d_warm is not None else "-"))
    md.append("- ganho de paginas quentes dentro do modo on: %s\n" % (
        "%.1f%%" % (100 * warm_gain_on) if warm_gain_on is not None else "-"))
    md.append("\n## Veredito: **%s**\n\n%s\n" % (verdict, reason))
    md.append("\n> `off` empurra o I/O do PLE para o load do modelo (veja load_s). "
              "O veredito e sobre o prefill; o custo de startup e reportado separadamente.\n")
    with open(os.path.join(args.outdir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("".join(md))

    print("".join(md))
    print("\nresultados em: %s" % args.outdir)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_common(p):
    p.add_argument("--server", default="/content/llama-parity/beellama/bin/llama-server")
    p.add_argument("--model", default="/content/fn40/real/Q2_0")
    p.add_argument("--port", type=int, default=8098)
    p.add_argument("--outdir", default="/tmp/opencode/profiling/lazy-ab")
    p.add_argument("--target-tokens", type=int, default=28600)
    p.add_argument("--request-timeout", type=int, default=3600)
    p.add_argument("--allow-busy-gpu", action="store_true")
    p.add_argument("--ncmoe", type=int, default=2, help="layers com experts no host")
    p.add_argument("--moe-cache-mib", type=int, default=6144,
                   help="cache de experts na GPU em MiB; -1 = auto")
    p.add_argument("--extra-arg", action="append", default=[], help="argumento extra (string) para o llama-server")


def main():
    ap = argparse.ArgumentParser(description="A/B --lazy-mode para a Fase 0 do plano Strata")
    sub = ap.add_subparsers(dest="cmd")

    p_info = sub.add_parser("model-info", help="imprime o cabecalho GGUF em JSON")
    p_info.add_argument("--model", default="/content/fn40/real/Q2_0")
    p_info.set_defaults(func=cmd_model_info)

    p_drive = sub.add_parser("drive", help="uma requisicao de prefill contra um servidor ja no ar")
    p_drive.add_argument("--port", type=int, required=True)
    p_drive.add_argument("--prompt-file", default=None)
    p_drive.add_argument("--target-tokens", type=int, default=28600)
    p_drive.add_argument("--calibrate", action="store_true", help="reconta tokens do --prompt-file")
    p_drive.add_argument("--server-pid", type=int, default=0)
    p_drive.add_argument("--out-json", required=True)
    p_drive.add_argument("--request-timeout", type=int, default=3600)
    p_drive.set_defaults(func=cmd_drive)

    p_ab = sub.add_parser("ablate", help="A/B completo on vs off, cold e warm")
    add_common(p_ab)
    p_ab.add_argument("--modes", default="on,off")
    p_ab.add_argument("--reps", type=int, default=2)
    p_ab.add_argument("--warm-arm", action="store_true", default=True)
    p_ab.add_argument("--no-warm-arm", dest="warm_arm", action="store_false")
    p_ab.add_argument("--drop-caches", choices=["auto", "yes", "no"], default="auto")
    p_ab.add_argument("--rebuild-prompt", action="store_true")
    p_ab.add_argument("--load-timeout", type=int, default=3600)
    p_ab.add_argument("--settle", type=float, default=3.0)
    p_ab.add_argument("--min-free-gib", type=int, default=0,
                      help="0 = automatico (PLE + 8 GiB)")
    p_ab.add_argument("--dry-run", action="store_true")
    p_ab.set_defaults(func=cmd_ablate)

    args = ap.parse_args()
    if args.cmd is None:
        ap.print_help()
        return 2
    if args.cmd == "ablate":
        args.modes = [m.strip() for m in args.modes.split(",") if m.strip()]
        for m in args.modes:
            if m not in ("on", "off"):
                die("modo invalido: %s" % m)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Classifica um trace nsys de prefill do Qwen3.8-Flash-Next em:

  GDN            -> gated_delta_net / ssm_conv / ssm_scan / l2_norm
  QSA/atencao    -> flash_attn / lightning_indexer / top_k_mask / set_rows / cumsum
  MoE/FFN        -> mul_mat_q com ggml_type 42 (Q2_0 dos experts), roteador e reducao
  copias         -> atividade CUDA de memcpy (HtoD/DtoH)
  host idle      -> wall - CPU do processo - GPU busy (ou SCHED_EVENTS, se existir)
  outro GPU      -> todo o resto (projecoes densas, norms, rope, glu, ...)

Uso:
  summarize_nsys.py --sqlite trace.sqlite --outdir OUT \
      --drive-json OUT/client.json --server-log OUT/server.log \
      --expert-type 42

O --expert-type vem de `ablate_lazy.py model-info` (nao hardcodar: o tipo dos
experts depende da quantizacao do GGUF). Sem ele, os kernels mul_mat_q nao sao
separados entre experts e projecoes densas, e o balde MoE fica incompleto.

Nao depende de gguf-py nem de nsys python; so sqlite3 stdlib.
"""

import argparse
import csv
import json
import os
import re
import sqlite3
import sys
from collections import defaultdict

COPY_H2D = {"HtoD", "HtoP", "PtoH"}
COPY_D2H = {"DtoH", "DtoP", "PtoD"}


def eprint(*a):
    print(*a, file=sys.stderr, flush=True)


def die(msg):
    eprint("ERRO: " + msg)
    sys.exit(2)


def table_exists(cur, name):
    cur.execute("SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name=?", (name,))
    return cur.fetchone() is not None


def table_columns(cur, name):
    cur.execute("PRAGMA table_info(%s)" % name)
    return [r[1] for r in cur.fetchall()]


def table_rowcount(cur, name):
    try:
        cur.execute("SELECT count(*) FROM %s" % name)
        return cur.fetchone()[0]
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# atividade CUDA
# ---------------------------------------------------------------------------

def kernel_name_expr(cols):
    """Devolve (expr_sql, precisa_join). Aceita StringIds (int) ou TEXT."""
    for cand in ("demangledName", "shortName", "mangledName", "name"):
        if cand in cols:
            return "COALESCE(s.value, CAST(k.%s AS TEXT))" % cand, True
    return None, False


def query_kernels(cur, start, end):
    cols = table_columns(cur, "CUPTI_ACTIVITY_KIND_KERNEL")
    expr, need_join = kernel_name_expr(cols)
    if expr is None:
        raise RuntimeError("CUPTI_ACTIVITY_KIND_KERNEL sem coluna de nome; schema: %s" % cols)
    join = ""
    if need_join and table_exists(cur, "StringIds"):
        join = "LEFT JOIN StringIds s ON s.id = k.%s" % (
            "demangledName" if "demangledName" in cols else
            ("shortName" if "shortName" in cols else
             ("mangledName" if "mangledName" in cols else "name"))
        )
    sql = ("SELECT k.start, k.end, %s AS name FROM CUPTI_ACTIVITY_KIND_KERNEL k %s "
           "WHERE k.end >= ? AND k.start <= ?" % (expr, join if need_join else ""))
    cur.execute(sql, (start, end))
    out = []
    for st, en, name in cur.fetchall():
        out.append((int(st) // 1000, int(en) // 1000, name or ""))
    return out


def query_memcpys(cur, start, end):
    if not table_exists(cur, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        return []
    cols = table_columns(cur, "CUPTI_ACTIVITY_KIND_MEMCPY")
    if "copyKind" not in cols:
        return []
    label = "CAST(m.copyKind AS TEXT)"
    join = ""
    if table_exists(cur, "ENUM_CUDA_MEMCPY_OPER"):
        enum_cols = table_columns(cur, "ENUM_CUDA_MEMCPY_OPER")
        lcol = "label" if "label" in enum_cols else ("name" if "name" in enum_cols else None)
        if lcol:
            label = "COALESCE(e.%s, CAST(m.copyKind AS TEXT))" % lcol
            join = "LEFT JOIN ENUM_CUDA_MEMCPY_OPER e ON e.id = m.copyKind"
    bytes_expr = "m.bytes" if "bytes" in cols else "0"
    sql = ("SELECT m.start, m.end, %s, %s FROM CUPTI_ACTIVITY_KIND_MEMCPY m %s "
           "WHERE m.end >= ? AND m.start <= ?" % (label, bytes_expr, join))
    cur.execute(sql, (start, end))
    return [(int(st) // 1000, int(en) // 1000, str(kind), int(nb or 0))
            for st, en, kind, nb in cur.fetchall()]


def query_osrt(cur, start, end):
    """Syscalls/OS runtime (precisa de --trace=osrt). Retorna (start,end,name)."""
    if not table_exists(cur, "OSRT_API"):
        return []
    cols = table_columns(cur, "OSRT_API")
    if "nameId" not in cols:
        return []
    join = "LEFT JOIN StringIds s ON s.id = o.nameId" if table_exists(cur, "StringIds") else ""
    expr = "COALESCE(s.value, CAST(o.nameId AS TEXT))" if join else "CAST(o.nameId AS TEXT)"
    sql = ("SELECT o.start, o.end, %s FROM OSRT_API o %s "
           "WHERE o.end >= ? AND o.start <= ?" % (expr, join))
    cur.execute(sql, (start, end))
    return [(int(st) // 1000, int(en) // 1000, nm or "") for st, en, nm in cur.fetchall()]


# ---------------------------------------------------------------------------
# classificacao
# ---------------------------------------------------------------------------

GDN_RE = re.compile(r"gated_delta_net|ssm_conv|ssm_scan|ssm_ssd|l2_norm_f32", re.I)
QSA_RE = re.compile(r"flash_attn|fattn|lightning_indexer|top_k_mask|k_set_rows|"
                    r"cumsum_kernel|diag_mask|topk_qsa", re.I)
MOE_AUX_RE = re.compile(r"topk_moe|moe_weighted_reduction|mm_ids_helper|add_id_kernel|"
                        r"mul_mat_vec_q_moe|mmq_build_moe_block_map", re.I)
MMQ_RE = re.compile(r"mul_mat_q", re.I)


def expert_mmq(name, expert_type):
    if not MMQ_RE.search(name):
        return False
    # o nome desmembrado traz os template args: "mul_mat_q<(ggml_type)42, 128, ...>"
    return re.search(r"\(ggml_type\)\s*%d\b" % expert_type, name) is not None


def classify_kernel(name, expert_type):
    if GDN_RE.search(name):
        return "gdn"
    if QSA_RE.search(name):
        return "qsa_attn"
    if MOE_AUX_RE.search(name):
        return "moe"
    if expert_mmq(name, expert_type):
        return "moe"
    if MMQ_RE.search(name):
        return "other_gpu"   # projecoes densas (ou experts se o tipo nao bateu)
    return "other_gpu"


def classify_copy(kind):
    k = (kind or "").replace(" ", "")
    if k in COPY_H2D:
        return "copy_h2d"
    if k in COPY_D2H:
        return "copy_d2h"
    return "copy_other"


def union_us(intervals):
    """Uniao de intervalos (us). Retorna (total_us, n_merged)."""
    if not intervals:
        return 0, 0
    iv = sorted(intervals)
    total = 0
    cur_s, cur_e = iv[0]
    n = 1
    for s, e in iv[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += max(0, cur_e - cur_s)
            cur_s, cur_e = s, e
            n += 1
    total += max(0, cur_e - cur_s)
    return total, n


def subtract_union(a, b):
    """a - b para intervalos [s,e). Usado para 'outro GPU'."""
    if not b:
        return union_us(a)[0]
    res = []
    for s, e in a:
        pieces = [(s, e)]
        for bs, be in b:
            nxt = []
            for ps, pe in pieces:
                if be <= ps or bs >= pe:
                    nxt.append((ps, pe))
                    continue
                if bs > ps:
                    nxt.append((ps, min(bs, pe)))
                if be < pe:
                    nxt.append((max(be, ps), pe))
            pieces = nxt
            if not pieces:
                break
        res.extend(pieces)
    return union_us(res)[0]


# ---------------------------------------------------------------------------
# logs do llama-server
# ---------------------------------------------------------------------------

RE_GRAPHPREP = re.compile(
    r"\[GRAPHPREP\] start_us=(-?\d+) nq=(\d+) reused=(\d+) apply_ms=([\d.]+) "
    r"check_ms=([\d.]+) reset_ms=([\d.]+) build_ms=([\d.]+) alloc_ms=([\d.]+) "
    r"inputs_ms=([\d.]+)")
RE_F5STAGE = re.compile(r"\[F5STAGE\] n_tokens=(\d+) graph_compute_host_ms=([\d.]+)")
RE_SPLITSPAN = re.compile(
    r"\[SPLITSPAN\] gid=(-?\d+) backend=(\S+) nodes=(\d+) t_iter_us=(-?\d+) "
    r"t_postsync_us=(-?\d+) t_postcopy_us=(-?\d+)")
RE_SPLIT = re.compile(r"\[SPLIT\] backend=(\S+) nodes=(\d+) wall_ms=([\d.]+)")
RE_SPLITWAIT = re.compile(
    r"\[SPLITWAIT\] gid=(-?\d+) backend=(\S+) wait_us=(-?\d+) memcpy_us=(-?\d+) ncopies=(\d+)")
RE_MOE_STATS = re.compile(
    r"llama_moe_cache: (.+?): hits = (\d+), misses = (\d+), hit rate = ([\d.]+)%, "
    r"uploaded = ([\d.]+) MiB")
RE_PROMPT_EVAL = re.compile(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens")


def parse_logs(path):
    out = {
        "graphprep": {"count": 0, "nq": {}, "reused": 0, "apply_ms": 0.0,
                      "check_ms": 0.0, "reset_ms": 0.0, "build_ms": 0.0,
                      "alloc_ms": 0.0, "inputs_ms": 0.0},
        "f5stage": {"count": 0, "host_ms": 0.0},
        "splitspan": {"count": 0, "per_backend": defaultdict(lambda: {
            "wait_us": 0, "copy_us": 0, "submit_us": 0, "nodes": 0, "splits": 0}),
            "gids": set()},
        "split_wall": {"count": 0, "per_backend": defaultdict(float)},
        "splitwait": {"count": 0, "memcpy_us": 0, "ncopies": 0},
        "moe_stats": [],
        "prompt_eval_ms": None,
        "prompt_eval_n": None,
    }
    if not path or not os.path.exists(path):
        return out

    spans = defaultdict(list)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = RE_GRAPHPREP.search(line)
            if m:
                g = out["graphprep"]
                g["count"] += 1
                g["nq"][m.group(2)] = g["nq"].get(m.group(2), 0) + 1
                g["reused"] += int(m.group(3))
                g["apply_ms"] += float(m.group(4))
                g["check_ms"] += float(m.group(5))
                g["reset_ms"] += float(m.group(6))
                g["build_ms"] += float(m.group(7))
                g["alloc_ms"] += float(m.group(8))
                g["inputs_ms"] += float(m.group(9))
                continue
            m = RE_F5STAGE.search(line)
            if m:
                out["f5stage"]["count"] += 1
                out["f5stage"]["host_ms"] += float(m.group(2))
                continue
            m = RE_SPLITSPAN.search(line)
            if m:
                gid, backend = int(m.group(1)), m.group(2)
                t_iter, t_post, t_copy = int(m.group(4)), int(m.group(5)), int(m.group(6))
                spans[gid].append((t_iter, t_post, t_copy, backend, int(m.group(3))))
                continue
            m = RE_SPLIT.search(line)
            if m:
                out["split_wall"]["count"] += 1
                out["split_wall"]["per_backend"][m.group(1)] += float(m.group(3))
                continue
            m = RE_SPLITWAIT.search(line)
            if m:
                out["splitwait"]["count"] += 1
                out["splitwait"]["memcpy_us"] += max(0, int(m.group(4)))
                out["splitwait"]["ncopies"] += int(m.group(5))
                continue
            m = RE_MOE_STATS.search(line)
            if m:
                out["moe_stats"].append({
                    "name": m.group(1), "hits": int(m.group(2)), "misses": int(m.group(3)),
                    "hit_rate": float(m.group(4)), "uploaded_mib": float(m.group(5)),
                })
                continue
            m = RE_PROMPT_EVAL.search(line)
            if m:
                out["prompt_eval_ms"] = float(m.group(1))
                out["prompt_eval_n"] = int(m.group(2))

    for gid, items in spans.items():
        items.sort()
        out["splitspan"]["gids"].add(gid)
        for i, (t_iter, t_post, t_copy, backend, nodes) in enumerate(items):
            b = out["splitspan"]["per_backend"][backend]
            b["wait_us"] += max(0, t_post - t_iter)
            b["copy_us"] += max(0, t_copy - t_post)
            b["nodes"] += nodes
            b["splits"] += 1
            if i + 1 < len(items):
                b["submit_us"] += max(0, items[i + 1][0] - t_copy)
            out["splitspan"]["count"] += 1
    return out


# ---------------------------------------------------------------------------
# CPU / SCHED_EVENTS
# ---------------------------------------------------------------------------

def sched_host_busy_us(cur, pid, start_us, end_us):
    """Uniao de intervalos 'thread rodando' do processo pid, em us."""
    if not table_exists(cur, "SCHED_EVENTS"):
        return None, 0
    cols = table_columns(cur, "SCHED_EVENTS")
    if not {"start", "isSchedIn", "globalTid"}.issubset(set(cols)):
        return None, 0
    start, end = start_us * 1000, end_us * 1000
    cur.execute("SELECT start, isSchedIn, globalTid FROM SCHED_EVENTS "
                "WHERE start <= ? ORDER BY start", (end,))
    events = defaultdict(list)
    for ts, is_in, gtid in cur.fetchall():
        p = int((gtid / 0x1000000) % 0x1000000)
        if p != pid:
            continue
        events[gtid].append((int(ts) // 1000, int(is_in)))

    busy = []
    for gtid, evs in events.items():
        open_ts = None
        for ts, is_in in evs:
            if is_in:
                open_ts = ts
            elif open_ts is not None:
                s, e = max(open_ts, start_us), min(ts, end_us)
                if e > s:
                    busy.append((s, e))
                open_ts = None
        if open_ts is not None:
            s, e = max(open_ts, start_us), end_us
            if e > s:
                busy.append((s, e))
    return union_us(busy)[0], len(busy)


# ---------------------------------------------------------------------------
# relatorio
# ---------------------------------------------------------------------------

CLASS_LABEL = {
    "gdn": "GDN (linear attention)",
    "qsa_attn": "QSA / atencao (FA + indexer)",
    "moe": "MoE / expert FFN (Q2_0)",
    "copy_h2d": "copias H2D",
    "copy_d2h": "copias D2H",
    "host_idle": "host idle",
    "other_gpu": "outro GPU (projecoes/norms/rope)",
    "copy_other": "outras copias/memset",
}


def pct(x, total):
    return 100.0 * x / total if total else 0.0


def log_only_report(args, drive):
    """Fallback sem nsys: so o que da para medir pelos logs e pelo /proc."""
    logs = parse_logs(args.server_log)
    wall_ms = args.wall_ms
    if wall_ms is None:
        wall_ms = drive.get("wall_ms") or ((drive.get("timings") or {}).get("prompt_ms"))
    read_mib = (drive.get("io_delta") or {}).get("read_bytes", 0) / 1048576.0
    majflt = (drive.get("proc_delta") or {}).get("majflt", 0)
    cpu_ticks = (drive.get("proc_delta") or {}).get("cpu_ticks", 0)
    try:
        hz = os.sysconf("SC_CLK_TCK")
    except Exception:
        hz = 100
    cpu_ms = cpu_ticks * 1000.0 / hz
    md = []
    md.append("# Fase 0 - resumo log-only (sem nsys)\n\n")
    md.append("- server log: `%s`\n" % args.server_log)
    md.append("- wall: %.1f ms; read_bytes %.1f MiB; majflt %s; CPU do processo %.1f ms\n" % (
        wall_ms or 0, read_mib, majflt, cpu_ms))
    if logs["graphprep"]["count"]:
        g = logs["graphprep"]
        md.append("- GRAPHPREP: %d grafos, total %.1f ms (apply %.1f, build %.1f, alloc %.1f)\n" % (
            g["count"], g["apply_ms"] + g["check_ms"] + g["reset_ms"] + g["build_ms"] +
            g["alloc_ms"] + g["inputs_ms"], g["apply_ms"], g["build_ms"], g["alloc_ms"]))
    if logs["f5stage"]["count"]:
        md.append("- F5STAGE: %.1f ms em %d ubatches (%.1f ms/ubatch)\n" % (
            logs["f5stage"]["host_ms"], logs["f5stage"]["count"],
            logs["f5stage"]["host_ms"] / max(1, logs["f5stage"]["count"])))
    if logs["splitspan"]["count"]:
        md.append("\n| backend | splits | wait ms | copy host ms | submit/compute host ms |\n")
        md.append("|---|---:|---:|---:|---:|\n")
        for be, d in logs["splitspan"]["per_backend"].items():
            md.append("| %s | %d | %.1f | %.1f | %.1f |\n" % (
                be, d.get("splits", 0), d["wait_us"] / 1000.0, d["copy_us"] / 1000.0,
                d["submit_us"] / 1000.0))
    if logs["split_wall"]["count"]:
        md.append("\n| backend | wall ms (com sync por split) |\n|---|---:|\n")
        for be, ms in logs["split_wall"]["per_backend"].items():
            md.append("| %s | %.1f |\n" % (be, ms))
    for s in logs["moe_stats"]:
        md.append("- MoE cache `%s`: hits %d, misses %d, %.2f%%, uploaded %.1f MiB\n" % (
            s["name"], s["hits"], s["misses"], s["hit_rate"], s["uploaded_mib"]))
    md.append("\n> Sem timeline de GPU: nao ha como separar GDN/QSA/MoE nem medir o idle\n")
    md.append("> com precisao. Use o A/B do PLE para a Fase 3 e repita com nsys para a Fase 4.\n")
    text = "".join(md)
    with open(os.path.join(args.outdir, "log_summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"log_only": True, "wall_ms": wall_ms, "read_mib": read_mib,
                   "majflt": majflt, "cpu_ms": cpu_ms,
                   "logs": {"graphprep": logs["graphprep"], "f5stage": logs["f5stage"],
                            "split_wall": logs["split_wall"]["per_backend"],
                            "moe_stats": logs["moe_stats"]}},
                  f, indent=2, sort_keys=True, default=str)
    print(text)
    return 0


def main():
    ap = argparse.ArgumentParser(description="resumo por classe de um trace nsys de prefill")
    ap.add_argument("--sqlite", default=None)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--drive-json", default=None, help="client.json do ablate_lazy.py drive")
    ap.add_argument("--server-log", default=None)
    ap.add_argument("--expert-type", type=int, default=None)
    ap.add_argument("--wall-ms", type=float, default=None)
    ap.add_argument("--start-ns", type=int, default=None, help="epoch ns do inicio do request")
    ap.add_argument("--end-ns", type=int, default=None, help="epoch ns do fim do request")
    ap.add_argument("--pad-ms", type=float, default=5.0)
    ap.add_argument("--window", choices=["auto", "markers"], default="auto")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--log-only", action="store_true",
                    help="sem sqlite: resume so os logs e o /proc (fallback sem nsys)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    drive = {}
    if args.drive_json and os.path.exists(args.drive_json):
        with open(args.drive_json, "r", encoding="utf-8") as f:
            drive = json.load(f)

    if args.log_only or not args.sqlite:
        return log_only_report(args, drive)

    if not os.path.exists(args.sqlite):
        die("sqlite nao encontrado: %s" % args.sqlite)

    wall_ms = args.wall_ms
    if wall_ms is None:
        wall_ms = drive.get("wall_ms")
    start_epoch = args.start_ns if args.start_ns is not None else drive.get("epoch_start_ns")
    end_epoch = args.end_ns if args.end_ns is not None else drive.get("epoch_end_ns")
    pid = drive.get("server_pid") or (drive.get("before", {}).get("proc", {}) or {}).get("pid")

    con = sqlite3.connect(args.sqlite)
    cur = con.cursor()

    if not table_exists(cur, "CUPTI_ACTIVITY_KIND_KERNEL") or \
            table_rowcount(cur, "CUPTI_ACTIVITY_KIND_KERNEL") == 0:
        die("trace sem kernels CUDA. O nsys rodou sem CUDA? Veja o log e o README "
            "(secao 'se nsys nao funcionar').")

    # --- janela ---
    session_epoch = None
    if table_exists(cur, "TARGET_INFO_SESSION_START_TIME"):
        cols = table_columns(cur, "TARGET_INFO_SESSION_START_TIME")
        col = "utcEpochNs" if "utcEpochNs" in cols else cols[0]
        cur.execute("SELECT %s FROM TARGET_INFO_SESSION_START_TIME LIMIT 1" % col)
        row = cur.fetchone()
        if row:
            session_epoch = int(row[0])

    cur.execute("SELECT min(start), max(end) FROM CUPTI_ACTIVITY_KIND_KERNEL")
    kmin, kmax = cur.fetchone()
    kmin, kmax = int(kmin), int(kmax)

    window_source = "markers"
    if start_epoch and end_epoch and session_epoch:
        pad = int(args.pad_ms * 1e6)
        start_ns = max(0, start_epoch - session_epoch - pad)
        end_ns = end_epoch - session_epoch + pad
    else:
        window_source = "auto-sem-markers"
        if wall_ms:
            start_ns, end_ns = max(0, kmax - int(wall_ms * 1.3e6)), kmax
        else:
            start_ns, end_ns = kmin, kmax

    start_us, end_us = start_ns // 1000, end_ns // 1000
    kernels = query_kernels(cur, start_us * 1000, end_us * 1000)
    if args.window == "auto" and len(kernels) == 0:
        eprint("AVISO: janela dos marcadores nao tem kernels; usando a ultima rajada do trace.")
        if wall_ms:
            start_ns, end_ns = max(0, kmax - int(wall_ms * 1.3e6)), kmax
            start_us, end_us = start_ns // 1000, end_ns // 1000
            window_source = "auto-rajada-final"
            kernels = query_kernels(cur, start_us * 1000, end_us * 1000)

    memcpys = query_memcpys(cur, start_us * 1000, end_us * 1000)
    osrt = query_osrt(cur, start_us * 1000, end_us * 1000)
    wall_us = max(1, end_us - start_us)

    # --- classes GPU ---
    expert_type = args.expert_type
    if expert_type is None:
        # pode vir do drive-json? nao vem; tenta inferir pelo nome mais comum
        counts = defaultdict(int)
        for _s, _e, nm in kernels:
            m = re.search(r"mul_mat_q<\(ggml_type\)(\d+)", nm)
            if m:
                counts[int(m.group(1))] += 1
        if counts:
            expert_type = max(counts.items(), key=lambda kv: kv[1])[0]
            eprint("AVISO: --expert-type ausente; inferido %d pelo kernel mais comum" % expert_type)

    by_class = defaultdict(lambda: {"us": 0, "n": 0, "intervals": []})
    for st, en, name in kernels:
        cls = classify_kernel(name, expert_type if expert_type is not None else -1)
        d = by_class[cls]
        d["us"] += max(0, en - st)
        d["n"] += 1
        d["intervals"].append((st, en))

    copy_us = {"copy_h2d": 0, "copy_d2h": 0, "copy_other": 0}
    copy_bytes = {"copy_h2d": 0, "copy_d2h": 0, "copy_other": 0}
    copy_n = {"copy_h2d": 0, "copy_d2h": 0, "copy_other": 0}
    copy_intervals = []
    copies_by_kind = defaultdict(lambda: {"us": 0, "bytes": 0, "n": 0})
    for st, en, kind, nb in memcpys:
        cls = classify_copy(kind)
        dur = max(0, en - st)
        copy_us[cls] += dur
        copy_bytes[cls] += nb
        copy_n[cls] += 1
        copy_intervals.append((st, en))
        k = copies_by_kind[kind]
        k["us"] += dur
        k["bytes"] += nb
        k["n"] += 1

    osrt_agg = defaultdict(lambda: {"us": 0, "n": 0})
    for st, en, nm in osrt:
        d = osrt_agg[nm]
        d["us"] += max(0, en - st)
        d["n"] += 1

    gpu_intervals = [(s, e) for s, e, _ in kernels] + copy_intervals
    gpu_busy_us, _ = union_us(gpu_intervals)

    maskset = {"gdn": [], "qsa_attn": [], "moe": []}
    for cls in maskset:
        maskset[cls] = by_class[cls]["intervals"]
    # "outro GPU" e o resto dos KERNELS; copias ficam no seu proprio balde.
    class_intervals = [iv for v in maskset.values() for iv in v]
    other_gpu_us = subtract_union([(s, e) for s, e, _ in kernels], class_intervals)

    # --- host ---
    host_busy_us = None
    host_src = ""
    sched_us, sched_n = (None, 0)
    if pid:
        sched_us, sched_n = sched_host_busy_us(cur, int(pid), start_us, end_us)
    if sched_us and sched_us > 0:
        host_busy_us = sched_us
        host_src = "SCHED_EVENTS (threads rodando)"
    elif drive.get("proc_delta", {}).get("cpu_ticks") is not None:
        try:
            hz = os.sysconf("SC_CLK_TCK")
        except Exception:
            hz = 100
        host_busy_us = int(drive["proc_delta"]["cpu_ticks"] * 1e6 / hz)
        host_src = "/proc/<pid>/task/*/stat (CPU de todas as threads)"
    else:
        host_busy_us = 0
        host_src = "indisponivel"

    host_busy_us = max(0, min(host_busy_us, wall_us))
    host_idle_us = max(0, wall_us - gpu_busy_us - host_busy_us)

    logs = parse_logs(args.server_log)

    # --- linhas do resumo ---
    rows = []

    def add(cls, us, n=0, source="nsys", note=""):
        rows.append({
            "class": cls,
            "label": CLASS_LABEL.get(cls, cls),
            "ms": us / 1000.0,
            "pct": pct(us, wall_us),
            "n": n,
            "source": source,
            "note": note,
        })

    add("gdn", by_class["gdn"]["us"], by_class["gdn"]["n"])
    add("qsa_attn", by_class["qsa_attn"]["us"], by_class["qsa_attn"]["n"])
    add("moe", by_class["moe"]["us"], by_class["moe"]["n"],
        note="mul_mat_q com ggml_type=%s" % expert_type)
    add("copy_h2d", copy_us["copy_h2d"], copy_n["copy_h2d"],
        note="%.1f MiB" % (copy_bytes["copy_h2d"] / 1048576.0))
    add("copy_d2h", copy_us["copy_d2h"], copy_n["copy_d2h"],
        note="%.1f MiB" % (copy_bytes["copy_d2h"] / 1048576.0))
    add("host_idle", host_idle_us, 0, source=host_src,
        note="wall - GPU busy - CPU do processo")
    add("other_gpu", other_gpu_us, 0,
        note="inclui mul_mat_q denso, norms, rope, glu, memset")
    if copy_us["copy_other"] or copy_n["copy_other"]:
        add("copy_other", copy_us["copy_other"], copy_n["copy_other"])

    five = sum(r["ms"] for r in rows
               if r["class"] in ("gdn", "qsa_attn", "moe", "copy_h2d", "copy_d2h", "host_idle"))
    five_pct = pct(five, wall_ms if wall_ms else wall_us / 1000.0)

    gpu_busy_ms = gpu_busy_us / 1000.0
    host_busy_ms = host_busy_us / 1000.0

    summary = {
        "sqlite": args.sqlite,
        "window": {"source": window_source, "start_ns": start_ns, "end_ns": end_ns,
                   "wall_ms": wall_us / 1000.0, "pad_ms": args.pad_ms},
        "client_wall_ms": wall_ms,
        "session_epoch_ns": session_epoch,
        "expert_type": expert_type,
        "gpu_busy_ms": gpu_busy_ms,
        "gpu_idle_ms": max(0.0, (wall_us - gpu_busy_us) / 1000.0),
        "host_busy_ms": host_busy_ms,
        "host_busy_source": host_src,
        "host_idle_ms": host_idle_us / 1000.0,
        "five_class_ms": five,
        "five_class_pct": five_pct,
        "rows": rows,
        "kernels": {"n": len(kernels), "total_ms": sum(max(0, e - s) for s, e, _ in kernels) / 1000.0},
        "osrt": {k: {"ms": v["us"] / 1000.0, "n": v["n"]}
                 for k, v in sorted(osrt_agg.items(), key=lambda kv: kv[1]["us"], reverse=True)[:20]},
        "copies": {
            "h2d": {"ms": copy_us["copy_h2d"] / 1000.0, "mib": copy_bytes["copy_h2d"] / 1048576.0,
                    "n": copy_n["copy_h2d"]},
            "d2h": {"ms": copy_us["copy_d2h"] / 1000.0, "mib": copy_bytes["copy_d2h"] / 1048576.0,
                    "n": copy_n["copy_d2h"]},
            "by_kind": {k: {"ms": v["us"] / 1000.0, "mib": v["bytes"] / 1048576.0, "n": v["n"]}
                        for k, v in copies_by_kind.items()},
        },
        "logs": {
            "graphprep": logs["graphprep"],
            "f5stage": logs["f5stage"],
            "splitspan": {k: (dict(v) if isinstance(v, dict) else v)
                          for k, v in logs["splitspan"].items() if k != "gids"},
            "n_graphs_split": len(logs["splitspan"]["gids"]),
            "split_wall": {"count": logs["split_wall"]["count"],
                           "per_backend": dict(logs["split_wall"]["per_backend"])},
            "splitwait": logs["splitwait"],
            "moe_stats": logs["moe_stats"],
            "prompt_eval_ms": logs["prompt_eval_ms"],
            "prompt_eval_n": logs["prompt_eval_n"],
        },
        "acceptance": {
            "five_class_pct": five_pct,
            "pass": five_pct >= 80.0,
            "baseline_within_5pct": None,
            "note": "baseline ±5% e calculado por profile_prefill.sh comparando clean vs trace",
        },
    }

    with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    with open(os.path.join(args.outdir, "class_summary.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "label", "ms", "pct_of_wall", "instances", "source", "note"])
        for r in rows:
            w.writerow([r["class"], r["label"], "%.3f" % r["ms"], "%.2f" % r["pct"],
                        r["n"], r["source"], r["note"]])

    top = sorted(kernels, key=lambda x: x[1] - x[0], reverse=True)[:args.top]
    with open(os.path.join(args.outdir, "kernels_top.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kernel", "class", "instances", "total_ms", "avg_us"])
        agg = defaultdict(lambda: [0, 0])
        for st, en, nm in kernels:
            a = agg[nm]
            a[0] += max(0, en - st)
            a[1] += 1
        for nm, (us, n) in sorted(agg.items(), key=lambda kv: kv[1][0], reverse=True)[:args.top]:
            w.writerow([nm, classify_kernel(nm, expert_type if expert_type is not None else -1),
                        n, "%.3f" % (us / 1000.0), "%.1f" % (us / n if n else 0)])

    with open(os.path.join(args.outdir, "memcpy_by_kind.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "class", "ms", "MiB", "count"])
        for k, v in sorted(copies_by_kind.items(), key=lambda kv: kv[1]["us"], reverse=True):
            w.writerow([k, classify_copy(k), "%.3f" % (v["us"] / 1000.0),
                        "%.1f" % (v["bytes"] / 1048576.0), v["n"]])

    with open(os.path.join(args.outdir, "osrt_top.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["osrt", "count", "total_ms"])
        for nm, v in sorted(osrt_agg.items(), key=lambda kv: kv[1]["us"], reverse=True)[:args.top]:
            w.writerow([nm, v["n"], "%.3f" % (v["us"] / 1000.0)])

    md = []
    md.append("# Fase 0 - atribuicao do prefill\n\n")
    md.append("- trace: `%s`\n" % args.sqlite)
    md.append("- janela: %s, %.1f ms (pad %.1f ms)\n" % (
        window_source, wall_us / 1000.0, args.pad_ms))
    if wall_ms:
        md.append("- wall do cliente (prompt_ms do servidor): %.1f ms\n" % wall_ms)
    md.append("- expert type: `%s`\n" % expert_type)
    md.append("- kernels: %d (%.1f ms somados), GPU busy (uniao): %.1f ms\n" % (
        len(kernels), summary["kernels"]["total_ms"], gpu_busy_ms))
    md.append("- host CPU (todas as threads): %.1f ms via %s\n" % (host_busy_ms, host_src))
    md.append("\n| classe | ms | % do wall | instancias | fonte | nota |\n")
    md.append("|---|---:|---:|---:|---|---|\n")
    for r in rows:
        md.append("| %s | %.1f | %.1f%% | %d | %s | %s |\n" % (
            r["label"], r["ms"], r["pct"], r["n"], r["source"], r["note"]))
    md.append("\n**Cobertura das 5 classes (GDN + QSA + MoE + H2D + D2H + host idle): "
              "%.1f%%** -> %s (criterio: >= 80%%)\n" % (five_pct, "PASS" if five_pct >= 80 else "FAIL"))
    md.append("\n## Contabilidade do host\n\n")
    md.append("- GPU busy (uniao): %.1f ms\n" % gpu_busy_ms)
    md.append("- host CPU: %.1f ms (%s)\n" % (host_busy_ms, host_src))
    md.append("- host idle: %.1f ms\n" % (host_idle_us / 1000.0))
    if logs["graphprep"]["count"]:
        g = logs["graphprep"]
        md.append("- GRAPHPREP: %d grafos, apply %.1f + check %.1f + reset %.1f + build %.1f + alloc %.1f + inputs %.1f ms\n" % (
            g["count"], g["apply_ms"], g["check_ms"], g["reset_ms"], g["build_ms"],
            g["alloc_ms"], g["inputs_ms"]))
    if logs["f5stage"]["count"]:
        md.append("- F5STAGE (submissao do graph_compute): %.1f ms em %d ubatches\n" % (
            logs["f5stage"]["host_ms"], logs["f5stage"]["count"]))
    if logs["splitspan"]["count"]:
        md.append("\n## Splits (GGML_SCHED_SPLIT_SPAN)\n\n")
        md.append("| backend | splits | wait (ms) | copy host (ms) | submit/compute host (ms) |\n")
        md.append("|---|---:|---:|---:|---:|\n")
        for be, d in logs["splitspan"]["per_backend"].items():
            md.append("| %s | %d | %.1f | %.1f | %.1f |\n" % (
                be, d.get("splits", 0), d["wait_us"] / 1000.0, d["copy_us"] / 1000.0,
                d["submit_us"] / 1000.0))
    if logs["moe_stats"]:
        md.append("\n## MoE cache\n\n")
        for s in logs["moe_stats"]:
            md.append("- `%s`: hits %d, misses %d, hit rate %.2f%%, uploaded %.1f MiB\n" % (
                s["name"], s["hits"], s["misses"], s["hit_rate"], s["uploaded_mib"]))
    if osrt_agg:
        md.append("\n## Syscalls / OS runtime (top 10, --trace=osrt)\n\n")
        md.append("| syscall | chamadas | ms |\n|---|---:|---:|\n")
        for nm, v in sorted(osrt_agg.items(), key=lambda kv: kv[1]["us"], reverse=True)[:10]:
            md.append("| `%s` | %d | %.1f |\n" % (nm, v["n"], v["us"] / 1000.0))
        md.append("\n> `madvise` é o `llama_prefetch_rows` do PLE; `pread64`/`read` são\n")
        md.append("> leituras explícitas. Page faults de mmap não aparecem como syscall,\n")
        md.append("> mas aparecem como `majflt` no client.json e como tempo no split CPU.\n")

    md.append("\n> `outro GPU` inclui as projecoes densas (mul_mat_q com tipo != %s), "
              "norms, rope e GLU. `host idle` = wall - GPU busy - CPU do processo; "
              "CPU que roda em paralelo com a GPU aparece como host CPU, nao como idle.\n" % expert_type)
    md.append("\n> Trace: GGML_SCHED_SPLIT_TIMING=1 adiciona um synchronize por split; "
              "use o total do run `clean` como denominador de aceitacao.\n")

    with open(os.path.join(args.outdir, "class_summary.md"), "w", encoding="utf-8") as f:
        f.write("".join(md))

    print("".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())

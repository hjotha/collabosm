#!/usr/bin/env python3
"""Teste sintetico do summarize_nsys.py (sem GPU, sem nsys)."""
import os
import sqlite3
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DB = os.path.join(HERE, "synthetic.sqlite")
LOG = os.path.join(HERE, "synthetic-server.log")
OUT = os.path.join(HERE, "synthetic-out")
SESSION_EPOCH = 1_800_000_000_000_000_000
PID = 4242


def gtid(pid, tid):
    return pid * 0x1000000 + tid


def main():
    if os.path.exists(DB):
        os.unlink(DB)
    os.makedirs(OUT, exist_ok=True)
    con = sqlite3.connect(DB)
    c = con.cursor()
    c.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    names = [
        "void gated_delta_net_cuda<128, false, false, false, false, false>(...)",
        "void flash_attn_tile<128, 128, 16, 1, false>(...)",
        "void mul_mat_q<(ggml_type)42, 128, false, (ggml_prec)30, false>(...)",
        "void mul_mat_q<(ggml_type)30, 128, false, (ggml_prec)30, false>(...)",
        "void topk_moe_cuda<128, false>(...)",
        "void k_set_rows<__half, int, __half>(...)",
        "void ssm_conv_f32<false, 128ul, 4ul>(...)",
    ]
    for i, n in enumerate(names, 1):
        c.execute("INSERT INTO StringIds VALUES (?,?)", (i, n))

    c.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
              "(start INTEGER, end INTEGER, demangledName INTEGER, shortName INTEGER)")
    t = 10_000_000
    rows = [
        (1, 10_000_000), (2, 8_000_000), (3, 20_000_000),
        (4, 3_000_000), (5, 1_000_000), (6, 2_000_000), (7, 4_000_000),
    ]
    for name_id, dur in rows:
        c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?)",
                  (t, t + dur, name_id, name_id))
        t += dur

    c.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY "
              "(start INTEGER, end INTEGER, copyKind INTEGER, bytes INTEGER)")
    c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?,?,?,?)",
              (t, t + 5_000_000, 1, 64 * 1024 * 1024))
    t += 5_000_000
    c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?,?,?,?)",
              (t, t + 1_000_000, 2, 1024))
    t += 1_000_000
    c.execute("CREATE TABLE ENUM_CUDA_MEMCPY_OPER (id INTEGER, label TEXT)")
    c.execute("INSERT INTO ENUM_CUDA_MEMCPY_OPER VALUES (1,'HtoD'),(2,'DtoH')")

    c.execute("INSERT INTO StringIds VALUES (?,?)", (100, "madvise"))
    c.execute("INSERT INTO StringIds VALUES (?,?)", (101, "pread64"))
    c.execute("CREATE TABLE OSRT_API (start INTEGER, end INTEGER, nameId INTEGER)")
    c.execute("INSERT INTO OSRT_API VALUES (?,?,?)", (20_000_000, 21_000_000, 100))
    c.execute("INSERT INTO OSRT_API VALUES (?,?,?)", (30_000_000, 32_000_000, 101))

    c.execute("CREATE TABLE TARGET_INFO_SESSION_START_TIME "
              "(utcEpochNs INTEGER, utcTime TEXT, localTime TEXT, systemClockNs INTEGER)")
    c.execute("INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES (?,?,?,?)",
              (SESSION_EPOCH, "x", "y", 12345))

    c.execute("CREATE TABLE SCHED_EVENTS (start INTEGER, cpu INTEGER, isSchedIn INTEGER, globalTid INTEGER)")
    # uma thread rodando de 64 ms a 68 ms, depois que todo o trabalho de GPU acabou
    c.execute("INSERT INTO SCHED_EVENTS VALUES (?,?,?,?)", (64_000_000, 0, 1, gtid(PID, 1)))
    c.execute("INSERT INTO SCHED_EVENTS VALUES (?,?,?,?)", (68_000_000, 0, 0, gtid(PID, 1)))
    con.commit()
    con.close()

    with open(LOG, "w") as f:
        f.write("[GRAPHPREP] start_us=100 nq=1024 reused=1 apply_ms=1.000 check_ms=0.100 "
                "reset_ms=0.200 build_ms=0.300 alloc_ms=0.400 inputs_ms=0.500\n")
        f.write("[GRAPHPREP] start_us=200 nq=1024 reused=1 apply_ms=1.000 check_ms=0.100 "
                "reset_ms=0.200 build_ms=0.300 alloc_ms=0.400 inputs_ms=0.500\n")
        f.write("[F5STAGE] n_tokens=1024 graph_compute_host_ms=2.000\n")
        f.write("[SPLITSPAN] gid=1 backend=CPU nodes=3 t_iter_us=0 t_postsync_us=0 t_postcopy_us=1000\n")
        f.write("[SPLITSPAN] gid=1 backend=CUDA0 nodes=900 t_iter_us=1500 t_postsync_us=1600 t_postcopy_us=2000\n")
        f.write("[SPLIT] backend=CPU nodes=3 wall_ms=1.000\n")
        f.write("[SPLIT] backend=CUDA0 nodes=900 wall_ms=45.000\n")
        f.write("llama_moe_cache: CUDA0: hits = 100, misses = 24, hit rate = 80.65%, uploaded = 1234.50 MiB\n")
        f.write("prompt eval time =   48000.00 ms / 28596 tokens (   1.68 ms per token, 595.75 tokens per second)\n")

    drive = os.path.join(OUT, "drive.json")
    with open(drive, "w") as f:
        import json
        json.dump({"server_pid": PID, "wall_ms": 64.0}, f)

    cmd = [
        sys.executable, os.path.join(ROOT, "summarize_nsys.py"),
        "--sqlite", DB, "--outdir", OUT, "--server-log", LOG,
        "--drive-json", drive,
        "--expert-type", "42",
        "--start-ns", str(SESSION_EPOCH + 10_000_000),
        "--end-ns", str(SESSION_EPOCH + 74_000_000),
        "--pad-ms", "0",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        return r.returncode

    import csv
    with open(os.path.join(OUT, "class_summary.csv")) as f:
        rows = list(csv.DictReader(f))
    by = {x["class"]: x for x in rows}
    checks = [
        (abs(float(by["gdn"]["ms"]) - 14.0) < 0.01, "GDN 14 ms (gated 10 + ssm 4)"),
        (abs(float(by["qsa_attn"]["ms"]) - 10.0) < 0.01, "QSA 10 ms (FA 8 + set_rows 2)"),
        (abs(float(by["moe"]["ms"]) - 21.0) < 0.01, "MoE 21 ms (mmq42 20 + topk 1)"),
        (abs(float(by["copy_h2d"]["ms"]) - 5.0) < 0.01, "H2D 5 ms"),
        (abs(float(by["copy_d2h"]["ms"]) - 1.0) < 0.01, "D2H 1 ms"),
        (abs(float(by["other_gpu"]["ms"]) - 3.0) < 0.01, "outro 3 ms (mmq30)"),
        (abs(float(by["host_idle"]["ms"]) - 6.0) < 0.1, "host idle 6 ms (tree running 64-68 ms)"),
    ]
    osrt_csv = os.path.join(OUT, "osrt_top.csv")
    checks.append((os.path.exists(osrt_csv), "osrt_top.csv gerado"))
    if os.path.exists(osrt_csv):
        with open(osrt_csv) as f:
            txt = f.read()
        checks.append(("madvise" in txt and "pread64" in txt, "madvise/pread64 no OSRT"))
    ok = True
    for cond, label in checks:
        print(("OK   " if cond else "FAIL ") + label)
        ok = ok and cond
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

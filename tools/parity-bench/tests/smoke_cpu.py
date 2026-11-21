#!/usr/bin/env python3
"""CPU-only integration test of bench.py/ablate.py using tests/fake_server.py.

Run from anywhere:  python3 tests/smoke_cpu.py
Needs no GPU, no model and never starts the real llama-server.
"""

import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BENCH = os.path.join(ROOT, "bench.py")
ABLATE = os.path.join(ROOT, "ablate.py")
FAKE = os.path.join(HERE, "fake_server.py")

failures = []


def check(name, condition, detail=""):
    if condition:
        print("PASS %s" % name, flush=True)
    else:
        print("FAIL %s%s" % (name, (": " + detail) if detail else ""), flush=True)
        failures.append(name)


def run(cmd, env=None, timeout=300):
    merged = os.environ.copy()
    if env:
        merged.update(env)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          env=merged)
    return proc


def main():
    work = tempfile.mkdtemp(prefix="parity-bench-smoke-", dir="/tmp")
    try:
        bin_dir = os.path.join(work, "bin")
        os.makedirs(bin_dir)
        wrapper = os.path.join(bin_dir, "llama-server")
        with open(wrapper, "w") as f:
            f.write("#!/bin/sh\nexec %s %s \"$@\"\n"
                    % (sys.executable, FAKE))
        os.chmod(wrapper, os.stat(wrapper).st_mode | stat.S_IEXEC)
        model = os.path.join(work, "model.gguf")
        with open(model, "wb") as f:
            f.write(b"fake gguf\n")

        out = os.path.join(work, "bench.json")
        req_log = os.path.join(work, "requests.json")
        cfg = os.path.join(work, "configs.json")
        with open(cfg, "w") as f:
            json.dump([
                {"name": "base-f16", "kv": "f16", "moe_cache_mib": 6144,
                 "batch": 2048, "ubatch": 512},
                {"name": "new-f16", "kv": "f16", "moe_cache_mib": 6144,
                 "batch": 2048, "ubatch": 512},
                {"name": "new-q8", "kv": "q8_0", "moe_cache_mib": -1,
                 "vram_reserve_mib": 2048, "batch": 2048, "ubatch": 1024,
                 "extra_args": ["--metrics"]},
            ], f)

        proc = run([sys.executable, BENCH, "--bin-dir", bin_dir, "--model", model,
                    "--out", out, "--configs", cfg, "--runs", "2",
                    "--warmup-requests", "1", "--cases", "short,pp4k",
                    "--reference", "base-f16", "--candidate", "new-f16",
                    "--port-base", "18090", "--load-timeout", "30",
                    "--shutdown-timeout", "15"],
                   env={"FAKE_LOAD_S": "1.0", "FAKE_REQUESTS_PATH": req_log})
        check("bench/exit", proc.returncode == 0, proc.stdout[-1500:] + proc.stderr[-1500:])
        check("bench/stdout-markdown", "## Throughput" in proc.stdout
              and "## Load, VRAM and MoE cache" in proc.stdout, proc.stdout[-800:])
        check("bench/acceptance-section", "## Acceptance" in proc.stdout,
              proc.stdout[-800:])
        payload = json.load(open(out))
        check("bench/schema", payload["schema"] == "parity-bench/1")
        names = [c["name"] for c in payload["configs"]]
        check("bench/config-order", names == ["base-f16", "new-f16", "new-q8"], str(names))
        base = payload["configs"][0]
        check("bench/runs", len(base["runs"]) == 2, str(len(base["runs"])))
        run1 = base["runs"][0]
        check("bench/load-ok", run1["ok"] and run1["complete"], str(run1["errors"]))
        check("bench/load-time", 0.5 <= (run1["load_s"] or 0) <= 10,
              str(run1["load_s"]))
        check("bench/cases", set(run1["cases"]) == {"short", "pp4k"},
              str(run1["cases"].keys()))
        short = run1["cases"]["short"]
        check("bench/timings", short["prompt_n"] > 0 and short["pp_tps"] == 100.0
              and short["tg_tps"] == 50.0, str(short))
        check("bench/cache-n", short["cache_n"] == 0)
        check("bench/moe-parse", run1["moe_cache"]["summary"]["small_hit_rate_pct"] == 68.76
              and run1["moe_cache"]["summary"]["large_hits"] == 96,
              str(run1["moe_cache"]["summary"]))
        check("bench/moe-aggregate",
              base["aggregate"]["moe"]["small_hits"] == 744,
              str(base["aggregate"]["moe"]))
        check("bench/graceful", run1["server_exit"]["graceful"] is True
              and run1["server_exit"]["exit_code"] == 0, str(run1["server_exit"]))
        check("bench/determinism",
              payload["determinism"]["per_case"]["short"]["all_configs_identical"] is True,
              str(payload["determinism"]["per_case"]["short"]))
        check("bench/comparison",
              payload["comparison"]["cases"]["short"]["configs"]["new-f16"]["pp_delta_pct"]
              == 0.0, str(payload["comparison"]["cases"]["short"]))
        check("bench/phase0-same-flags-pass",
              payload["comparison"]["acceptance"]["checks"][0]["pass"] is True,
              str(payload["comparison"]["acceptance"]["checks"][0]))
        check("bench/warmup", run1["warmup_s"] is not None, str(run1["warmup_s"]))
        requests = [json.loads(line) for line in open(req_log) if line.strip()]
        # 1 warmup + 2 cases per run, 2 runs, 3 configs
        check("bench/request-count", len(requests) == 18, str(len(requests)))
        check("bench/protocol-kwargs",
              all(r["cache_prompt"] is False
                  and r["chat_template_kwargs"] == {"enable_thinking": False}
                  and r["temperature"] == 0 for r in requests),
              str(requests[:1]))

        # failed load must be recorded, not fatal
        fail_out = os.path.join(work, "fail.json")
        proc = run([sys.executable, BENCH, "--bin-dir", bin_dir, "--model", model,
                    "--out", fail_out, "--runs", "1", "--cases", "short",
                    "--port-base", "18110", "--load-timeout", "6",
                    "--shutdown-timeout", "10", "--config-json",
                    '{"name":"broken","env":{"FAKE_FAIL":"1"}}'])
        check("fail/exit", proc.returncode == 0, proc.stdout[-1000:] + proc.stderr[-1000:])
        failed = json.load(open(fail_out))["configs"][0]
        check("fail/recorded", failed["runs"][0]["ok"] is False
              and failed["runs"][0]["errors"], str(failed["runs"][0].get("errors")))
        check("fail/log-tail", bool(failed["runs"][0].get("log_tail")),
              str(failed["runs"][0].get("log_tail")))

        # crash mid-cases: remaining cases are recorded as errors, harness survives
        crash_out = os.path.join(work, "crash.json")
        proc = run([sys.executable, BENCH, "--bin-dir", bin_dir, "--model", model,
                    "--out", crash_out, "--runs", "1", "--warmup-requests", "0",
                    "--cases", "short,pp4k", "--port-base", "18130",
                    "--load-timeout", "20", "--shutdown-timeout", "5",
                    "--config-json",
                    '{"name":"crashy","env":{"FAKE_DIE_AFTER_REQUESTS":"1"}}'])
        check("crash/exit", proc.returncode == 0, proc.stderr[-800:])
        crashed = json.load(open(crash_out))["configs"][0]
        check("crash/partial", "short" in crashed["runs"][0]["cases"]
              and crashed["runs"][0]["ok"] is True
              and crashed["runs"][0]["complete"] is False,
              str(crashed["runs"][0]["cases"].keys()))
        check("crash/errors", len(crashed["errors"]) >= 1, str(crashed["errors"]))

        # ablate: real run against the fake server, 1 run, 1 case
        ab_out = os.path.join(work, "ablate.json")
        proc = run([sys.executable, ABLATE, "--bin-dir", bin_dir, "--model", model,
                    "--out", ab_out, "--runs", "1", "--cases", "short",
                    "--moe-cache-mib", "-1", "--port-base", "18150",
                    "--load-timeout", "30", "--shutdown-timeout", "15"],
                   env={"FAKE_LOAD_S": "0.3"})
        check("ablate/exit", proc.returncode == 0, proc.stdout[-1200:] + proc.stderr[-1200:])
        ab = json.load(open(ab_out))
        check("ablate/schema", ab["schema"] == "parity-bench-ablate/1")
        check("ablate/variants", len(ab["ablation"]["variants"]) == 11,
              str(len(ab["ablation"]["variants"])))
        check("ablate/aliases", ab["ablation"]["alias_of"].get("kv-f16") == "base"
              and ab["ablation"]["alias_of"].get("lazy-on") == "base"
              and ab["ablation"]["alias_of"].get("ub-512") == "base",
              str(ab["ablation"]["alias_of"]))
        check("ablate/alias-runs", any(c["name"] == "kv-f16" and c.get("alias_of") == "base"
                                       for c in ab["configs"]),
              str([(c["name"], c.get("alias_of")) for c in ab["configs"]]))
        check("ablate/table", "| variant | axis | change |" in proc.stdout)
        check("ablate/raw-json", os.path.isfile(ab_out.replace(".json", ".bench.json")))

        print("", flush=True)
        if failures:
            print("smoke FAILED: %d check(s)" % len(failures), flush=True)
            print("work dir kept: %s" % work, flush=True)
            return 1
        print("smoke OK", flush=True)
        return 0
    finally:
        if not failures:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""ablate.py: phase 0 ablations on top of one base config.

Generates a deterministic set of variants (kv, lazy-mode, ubatch, n-cpu-moe,
vram-reserve), runs them with bench.py's engine and writes a comparative JSON
plus a Markdown delta table. Variants that resolve to exactly the same server
flags as the base are reported as alias_of=base and are not re-run.

No GPU work happens in --dry-run / --self-test.
"""

import argparse
import copy
import json
import os
import shlex
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402

SCHEMA = bench.ABLATE_SCHEMA

AXES = ("kv", "lazy-mode", "ubatch", "n_cpu_moe", "vram-reserve")

KV_VARIANTS = ("f16", "q8_0")
LAZY_VARIANTS = ("on", "off")
UBATCH_VARIANTS = (512, 1024)
NCMOE_VARIANTS = (1, 2, 3)
RESERVE_VARIANTS = (1536, 2048)


def base_config(args):
    extra = shlex.split(args.extra) if args.extra else []
    return {
        "name": "base",
        "kv": args.kv,
        "moe_cache_mib": args.moe_cache_mib,
        "batch": args.batch,
        "ubatch": args.ubatch,
        "ctx": args.ctx,
        "ngl": args.ngl,
        "n_cpu_moe": args.n_cpu_moe,
        "lazy_mode": args.lazy_mode,
        "vram_reserve_mib": args.vram_reserve_mib,
        "extra_args": extra,
    }


def build_variants(args):
    """Return the ordered list of (axis, change, raw config) variants.

    Names are deterministic: base, kv-f16, kv-q8_0, lazy-on, lazy-off,
    ub-512, ub-1024, ncmoe-1, ncmoe-2, ncmoe-3, reserve-1536, reserve-2048.
    """
    base = base_config(args)
    variants = [{"axis": None, "change": {}, "config": dict(base)}]

    def add(axis, change, name):
        cfg = dict(base)
        cfg.update(change)
        cfg["name"] = name
        variants.append({"axis": axis, "change": dict(change), "config": cfg})

    for kv in KV_VARIANTS:
        add("kv", {"kv": kv}, "kv-%s" % kv)
    for lazy in LAZY_VARIANTS:
        add("lazy-mode", {"lazy_mode": lazy}, "lazy-%s" % lazy)
    for ubatch in UBATCH_VARIANTS:
        add("ubatch", {"ubatch": ubatch}, "ub-%d" % ubatch)
    for ncmoe in NCMOE_VARIANTS:
        add("n_cpu_moe", {"n_cpu_moe": ncmoe}, "ncmoe-%d" % ncmoe)
    for reserve in RESERVE_VARIANTS:
        add("vram-reserve", {"vram_reserve_mib": reserve}, "reserve-%d" % reserve)
    return variants


def plan_variants(variants, opts):
    """Normalize configs, detect exact duplicates and return an execution plan."""
    normalized = []
    for item in variants:
        cfg = bench.normalize_config(item["config"], opts, len(normalized) + 1)
        signature = tuple(bench.flags_signature(bench.build_server_args(cfg, opts, 1)))
        normalized.append({"axis": item["axis"], "change": item["change"],
                           "config": cfg, "signature": signature})

    to_run = []
    by_signature = {}
    aliases = {}
    for item in normalized:
        name = item["config"]["name"]
        if item["signature"] in by_signature:
            aliases[name] = by_signature[item["signature"]]
            item["alias_of"] = by_signature[item["signature"]]
            continue
        by_signature[item["signature"]] = name
        item["alias_of"] = None
        to_run.append(item["config"])
    order = [item["config"]["name"] for item in normalized]
    return normalized, to_run, aliases, order


def variant_row(payload, item):
    name = item["config"]["name"]
    entry = None
    for candidate in payload["configs"]:
        if candidate["name"] == name:
            entry = candidate
            break
    if entry is None:
        return None
    agg = entry.get("aggregate") or {}
    base_agg = None
    for candidate in payload["configs"]:
        if candidate["name"] == "base":
            base_agg = candidate.get("aggregate") or {}
            break
    base_cases = (base_agg or {}).get("cases") or {}
    cases = {}
    for case in payload["protocol"]["cases"]:
        cname = case["name"]
        node = (agg.get("cases") or {}).get(cname) or {}
        base_node = base_cases.get(cname) or {}
        cases[cname] = {
            "pp_tps_median": node.get("pp_tps_median"),
            "tg_tps_median": node.get("tg_tps_median"),
            "pp_delta_pct": bench.round_or_none(
                bench.pct_delta(node.get("pp_tps_median"), base_node.get("pp_tps_median")), 2),
            "tg_delta_pct": bench.round_or_none(
                bench.pct_delta(node.get("tg_tps_median"), base_node.get("tg_tps_median")), 2),
        }
    load = agg.get("load_s") or {}
    base_load = ((base_agg or {}).get("load_s") or {}).get("median")
    load_delta = bench.pct_delta(load.get("median"), base_load)
    moe = agg.get("moe") or {}
    peak_delta = None
    if (agg.get("vram_peak_mib") is not None
            and (base_agg or {}).get("vram_peak_mib") is not None):
        peak_delta = agg["vram_peak_mib"] - base_agg["vram_peak_mib"]
    return {
        "name": name,
        "axis": item["axis"],
        "changes": item["change"],
        "alias_of": item.get("alias_of"),
        "cases": cases,
        "load_s_median": load.get("median"),
        "load_s_delta_pct": bench.round_or_none(load_delta, 2),
        "vram_ready_mib": agg.get("vram_ready_mib"),
        "vram_peak_mib": agg.get("vram_peak_mib"),
        "vram_peak_delta_mib": bench.round_or_none(peak_delta, 0),
        "moe": {
            "small_hit_rate_pct": moe.get("small_hit_rate_pct"),
            "large_hit_rate_pct": moe.get("large_hit_rate_pct"),
            "uploaded_mib_total": moe.get("uploaded_mib_total"),
            "auto": moe.get("auto"),
        },
        "errors": entry.get("errors") or [],
    }


def build_ablation(payload, items, aliases, args):
    warnings = []
    if args.moe_cache_mib != -1:
        warnings.append(
            "--moe-cache-mib is %s (fixed): --vram-reserve-mib only affects the auto "
            "cache, so reserve-* variants are expected to match base. Use "
            "--moe-cache-mib -1 to exercise them." % args.moe_cache_mib)
    for item in items:
        if item.get("alias_of"):
            warnings.append("%s resolves to the same server flags as %s; reused its runs"
                            % (item["config"]["name"], item["alias_of"]))

    rows = []
    for item in items:
        row = variant_row(payload, item)
        if row is not None:
            rows.append(row)

    base_row = next((r for r in rows if r["name"] == "base"), None)
    variants = []
    for row in rows:
        if row["name"] == "base":
            continue
        long_pp = (row["cases"].get(bench.CASE_28K) or {}).get("pp_delta_pct")
        long_tg = (row["cases"].get(bench.CASE_28K_DEC) or {}).get("tg_tps_median")
        short_tg = (row["cases"].get(bench.CASE_SHORT) or {}).get("tg_delta_pct")
        row["targets_vs_base"] = {
            "long_prefill_delta_pct": long_pp,
            "long_prefill_ge_25pct": (long_pp is not None and long_pp >= 25.0),
            "long_decode_tps": long_tg,
            "long_decode_ge_55tps": (long_tg is not None and long_tg >= 55.0),
            "short_decode_delta_pct": short_tg,
            "short_regression_le_3pct": (short_tg is not None and short_tg >= -3.0),
        }
        variants.append(row)

    return {
        "schema": SCHEMA,
        "base": "base" if base_row else None,
        "axes": list(AXES),
        "base_config": payload["configs"][0].get("config") if payload["configs"] else None,
        "lenient_targets_note": (
            "targets_vs_base is informational: the phase-1 thresholds (+25% long "
            "prefill, >=55 t/s decode after 28.6k, -3% short decode) apply to the "
            "new build vs the baseline build with identical flags, not to kv/lazy "
            "ablations. phase 0's +/-5% check applies to same-flags A/B runs."),
        "variants": variants,
        "warnings": warnings,
        "alias_of": dict(sorted(aliases.items())),
    }


def render_ablation(payload):
    ablation = payload.get("ablation") or {}
    proto = payload.get("protocol") or {}
    lines = []
    lines.append("## Ablation deltas vs base")
    lines.append("")
    lines.append("| variant | axis | change | pp4k d% | pp28k d% | short tg d% | "
                 "28k decode t/s | load d% | peak VRAM d MiB | MoE small hit% |")
    lines.append("|" + "|".join(["---"] * 10) + "|")

    def cell(mapping, key, spec):
        return bench.fmt((mapping or {}).get(key), spec)

    for row in ablation.get("variants") or []:
        cases = row.get("cases") or {}
        change = ", ".join("%s=%s" % (k, v) for k, v in sorted(row["changes"].items()))
        name = row["name"] + (" (alias of base)" if row.get("alias_of") else "")
        lines.append("| " + " | ".join([
            name,
            row.get("axis") or "-",
            change or "-",
            cell(cases.get(bench.CASE_4K), "pp_delta_pct", "%+.2f"),
            cell(cases.get(bench.CASE_28K), "pp_delta_pct", "%+.2f"),
            cell(cases.get(bench.CASE_SHORT), "tg_delta_pct", "%+.2f"),
            cell(cases.get(bench.CASE_28K_DEC), "tg_tps_median", "%.2f"),
            bench.fmt(row.get("load_s_delta_pct"), "%+.2f"),
            bench.fmt(row.get("vram_peak_delta_mib"), "%+.0f"),
            cell(row.get("moe"), "small_hit_rate_pct", "%.2f"),
        ]) + " |")
    lines.append("")

    for row in ablation.get("variants") or []:
        targets = row.get("targets_vs_base") or {}
        if not targets:
            continue
        lines.append("- %s: long prefill %s%% (target >= +25%%, %s), decode@28k %s t/s "
                     "(target >= 55, %s), short decode %s%% (target >= -3%%, %s)"
                     % (row["name"],
                        bench.fmt(targets.get("long_prefill_delta_pct"), "%+.2f"),
                        "PASS" if targets.get("long_prefill_ge_25pct") else "FAIL",
                        bench.fmt(targets.get("long_decode_tps"), "%.2f"),
                        "PASS" if targets.get("long_decode_ge_55tps") else "FAIL",
                        bench.fmt(targets.get("short_decode_delta_pct"), "%+.2f"),
                        "PASS" if targets.get("short_regression_le_3pct") else "FAIL"))
    lines.append("")
    for warning in ablation.get("warnings") or []:
        lines.append("- warning: %s" % warning)
    if ablation.get("warnings"):
        lines.append("")
    lines.append("_%s_" % ablation.get("lenient_targets_note"))
    return "\n".join(lines)


def make_bench_opts(args):
    ns = argparse.Namespace(
        bin_dir=args.bin_dir,
        model=args.model,
        out=args.raw_out,
        log_dir=args.log_dir,
        alias=bench.ALIAS,
        gpu_index=args.gpu_index,
        runs=args.runs,
        reference="base",
        candidate=None,
        warmup_requests=args.warmup_requests,
        port_base=args.port_base,
        load_timeout=args.load_timeout,
        request_timeout=args.request_timeout,
        shutdown_timeout=args.shutdown_timeout,
    )
    return bench.make_opts(ns)


def dry_run(items, args, cases):
    say = bench.say
    say("=== ablation dry run: %d variant(s), no server is started ==="
        % len(items))
    opts = make_bench_opts(args)
    problems = []
    if not os.path.isfile(opts.model):
        problems.append("model not found: %s" % opts.model)
    say("model: %s" % opts.model)
    say("out:   %s" % os.path.abspath(args.out))
    say("raw:   %s" % opts.out)
    say("logs:  %s" % opts.log_dir)
    say("cases: %s" % ", ".join("%s/%d" % (c["name"], c["max_tokens"]) for c in cases))
    say("")
    seen = {}
    for item in items:
        cfg = item["config"]
        server = os.path.join(cfg["bin_dir"], "llama-server")
        signature = tuple(bench.flags_signature(bench.build_server_args(cfg, opts, 1)))
        alias = seen.get(signature)
        if alias:
            state = "alias of %s" % alias
        else:
            seen[signature] = cfg["name"]
            state = "run"
        if not os.path.isfile(server):
            problems.append("server binary not found: %s" % server)
        say("[%-12s] axis=%-12s %-12s %s"
            % (cfg["name"], item["axis"] or "-", state,
               "kv=%s cache=%s b=%s ub=%s lazy=%s ncmoe=%s reserve=%s"
               % (cfg["kv"], cfg["moe_cache_mib"], cfg["batch"], cfg["ubatch"],
                  cfg["lazy_mode"], cfg["n_cpu_moe"], cfg["vram_reserve_mib"])))
        say("  cmd: %s" % " ".join(shlex.quote(a) for a in
                                   bench.build_server_args(cfg, opts, args.port_base)))
    say("")
    if args.moe_cache_mib != -1:
        say("note: --vram-reserve-mib only matters with --moe-cache-mib -1")
    if problems:
        say("problems:")
        for problem in problems:
            say("- %s" % problem)
    say("dry run OK (%d unique command line(s), %d alias(es))"
        % (len(seen), len(items) - len(seen)))
    return 0


def self_test():
    failures = []

    def check(name, condition, detail=""):
        if condition:
            bench.say("PASS %s" % name)
        else:
            bench.say("FAIL %s%s" % (name, (": " + detail) if detail else ""))
            failures.append(name)

    args = argparse.Namespace(
        kv="f16", moe_cache_mib=-1, batch=2048, ubatch=512, ctx=32768, ngl=99,
        n_cpu_moe=2, lazy_mode="on", vram_reserve_mib=None, extra="")
    variants = build_variants(args)
    names = [v["config"]["name"] for v in variants]
    expected = ["base", "kv-f16", "kv-q8_0", "lazy-on", "lazy-off", "ub-512",
                "ub-1024", "ncmoe-1", "ncmoe-2", "ncmoe-3", "reserve-1536",
                "reserve-2048"]
    check("names/deterministic", names == expected, str(names))
    check("names/count", len(names) == len(set(names)))

    opts = argparse.Namespace(bin_dir="/tmp/bin", model="/tmp/m.guff", alias="q",
                              gpu_index=None, runs=2, reference=None, candidate=None)
    items, to_run, aliases, order = plan_variants(variants, opts)
    check("plan/base-first", to_run[0]["name"] == "base")
    check("plan/aliases", aliases.get("kv-f16") == "base"
          and aliases.get("lazy-on") == "base"
          and aliases.get("ub-512") == "base"
          and aliases.get("ncmoe-2") == "base", str(aliases))
    check("plan/unique", "kv-q8_0" not in aliases and "lazy-off" not in aliases
          and "ub-1024" not in aliases and "ncmoe-1" not in aliases,
          str(aliases))
    check("plan/order", order == expected, str(order))

    # base already q8_0: kv-q8_0 must alias, kv-f16 must not.
    args2 = copy.copy(args)
    args2.kv = "q8_0"
    items2, to_run2, aliases2, _ = plan_variants(build_variants(args2), opts)
    check("plan/alias-flips", aliases2.get("kv-q8_0") == "base"
          and "kv-f16" not in aliases2, str(aliases2))

    # ubatch must stay <= batch, so a base with ub=2048/b=2048 is invalid for 1024? no:
    # 1024 <= 2048 is valid; instead check that an invalid base is rejected.
    args3 = copy.copy(args)
    args3.ubatch = 4096
    try:
        plan_variants(build_variants(args3), opts)
        check("plan/validates-base", False, "no exception")
    except ValueError:
        check("plan/validates-base", True)

    # rendering works with a minimal synthetic payload
    base_agg = {"load_s": {"median": 30.0}, "vram_peak_mib": 31000,
                "cases": {}, "moe": {}}
    configs = []
    for item in items:
        name = item["config"]["name"]
        if name == "kv-q8_0":
            configs.append({
                "name": name,
                "aggregate": {"load_s": {"median": 31.5}, "vram_peak_mib": 30000,
                              "cases": {bench.CASE_28K: {"pp_tps_median": 120.0,
                                                         "tg_tps_median": 80.0},
                                        bench.CASE_28K_DEC: {"pp_tps_median": 120.0,
                                                             "tg_tps_median": 55.5},
                                        bench.CASE_SHORT: {"pp_tps_median": 200.0,
                                                           "tg_tps_median": 50.0},
                                        bench.CASE_4K: {"pp_tps_median": 500.0,
                                                        "tg_tps_median": 60.0}},
                              "moe": {"small_hit_rate_pct": 40.0,
                                      "large_hit_rate_pct": 10.0,
                                      "uploaded_mib_total": 100.0}},
                "errors": []})
        else:
            configs.append({"name": name, "aggregate": base_agg, "errors": []})
    payload = {
        "protocol": {"cases": [{"name": n, "max_tokens": 16} for n in bench.CASE_NAMES]},
        "configs": configs,
        "comparison": {"cases": {bench.CASE_SHORT: {"configs": {"kv-q8_0": {"pp_delta_pct": 1.0, "tg_delta_pct": -2.0, "pp_tps_median": 200.0, "tg_tps_median": 50.0}}},
                                 bench.CASE_4K: {"configs": {"kv-q8_0": {"pp_delta_pct": 2.0, "tg_delta_pct": 1.0, "pp_tps_median": 500.0, "tg_tps_median": 60.0}}},
                                 bench.CASE_28K: {"configs": {"kv-q8_0": {"pp_delta_pct": 24.0, "tg_delta_pct": 1.0, "pp_tps_median": 120.0, "tg_tps_median": 80.0}}},
                                 bench.CASE_28K_DEC: {"configs": {"kv-q8_0": {"pp_delta_pct": 24.0, "tg_delta_pct": 1.0, "pp_tps_median": 120.0, "tg_tps_median": 55.5}}}},
                       },
    }
    payload["ablation"] = build_ablation(payload, items, aliases, args)
    markdown = render_ablation(payload)
    check("render/table", "| variant | axis | change |" in markdown, markdown[:200])
    check("render/targets", "target >= +25%" in markdown)
    built = {row["name"]: row for row in payload["ablation"]["variants"]}
    check("ablation/alias-row", built["kv-f16"]["alias_of"] == "base")
    check("ablation/target-eval",
          built["kv-q8_0"]["targets_vs_base"]["long_prefill_ge_25pct"] is False
          and built["kv-q8_0"]["targets_vs_base"]["long_decode_ge_55tps"] is True,
          str(built["kv-q8_0"]["targets_vs_base"]))

    bench.say("")
    if failures:
        bench.say("ablate self-test FAILED: %d check(s)" % len(failures))
        return 1
    bench.say("ablate self-test OK")
    return 0


def build_argparser():
    p = argparse.ArgumentParser(
        description="Phase 0 ablations for the Strata parity plan (kv, lazy, ubatch, "
                    "n-cpu-moe, vram-reserve).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  ablate.py --bin-dir /content/new/bin --model model.gguf \\\n"
            "      --out ablate.json --moe-cache-mib -1 --vram-reserve-mib 1536\n"))
    p.add_argument("--bin-dir", help="directory containing llama-server")
    p.add_argument("--model", help="shard 1 of the GGUF model")
    p.add_argument("--out", help="comparative output JSON")
    p.add_argument("--raw-out", help="raw bench JSON (default: <out>.bench.json)")
    p.add_argument("--log-dir", help="override the server log directory")

    p.add_argument("--kv", default="f16", choices=bench.KV_CHOICES,
                   help="base KV cache type (default f16)")
    p.add_argument("--moe-cache-mib", type=int, default=6144,
                   help="base --moe-cache-mib (default 6144; -1 = auto)")
    p.add_argument("--batch", type=int, default=2048, help="base -b (default 2048)")
    p.add_argument("--ubatch", type=int, default=512, help="base -ub (default 512)")
    p.add_argument("--ctx", type=int, default=32768, help="base -c (default 32768)")
    p.add_argument("--ngl", type=int, default=99, help="base -ngl (default 99)")
    p.add_argument("--n-cpu-moe", type=int, default=2,
                   help="base --n-cpu-moe (default 2)")
    p.add_argument("--lazy-mode", default="on", choices=bench.LAZY_CHOICES,
                   help="base --lazy-mode (default on)")
    p.add_argument("--vram-reserve-mib", type=int, default=None,
                   help="base --vram-reserve-mib (default: server default 1536)")
    p.add_argument("--extra", default="",
                   help="extra server flags for every variant (shell-quoted string)")

    p.add_argument("--runs", type=int, default=2, help="repetitions per variant")
    p.add_argument("--cases", help="comma-separated subset of cases")
    p.add_argument("--warmup-requests", type=int, default=1)
    p.add_argument("--port-base", type=int, default=8090)
    p.add_argument("--gpu-index", type=int, default=None)
    p.add_argument("--load-timeout", type=float, default=360.0)
    p.add_argument("--request-timeout", type=float, default=1800.0)
    p.add_argument("--shutdown-timeout", type=float, default=60.0)
    p.add_argument("--dry-run", action="store_true",
                   help="print the generated variants and commands, run nothing")
    p.add_argument("--self-test", action="store_true",
                   help="test variant generation/validation/rendering (no GPU)")
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

    if not args.raw_out:
        stem, ext = os.path.splitext(os.path.abspath(args.out))
        args.raw_out = stem + ".bench" + (ext or ".json")

    selected = None
    if args.cases:
        selected = [c.strip() for c in args.cases.split(",") if c.strip()]
        unknown = [c for c in selected if c not in bench.CASE_NAMES]
        if unknown:
            parser.error("unknown --cases value(s): %s (known: %s)"
                         % (unknown, ", ".join(bench.CASE_NAMES)))
    cases = bench.build_cases(selected)

    variants = build_variants(args)
    try:
        items, to_run, aliases, order = plan_variants(variants, make_bench_opts(args))
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    if args.dry_run:
        return dry_run(items, args, cases)

    opts = make_bench_opts(args)
    if not os.path.isfile(opts.model):
        parser.error("model not found: %s" % opts.model)
    for cfg in to_run:
        server = os.path.join(cfg["bin_dir"], "llama-server")
        if not os.path.isfile(server):
            parser.error("llama-server not found in %s" % cfg["bin_dir"])

    bench.say("ablate: %d variants (%d unique command lines, %d aliases), "
              "%d run(s) each, %d case(s)"
              % (len(items), len(to_run), len(aliases), args.runs, len(cases)))
    payload = bench.execute(to_run, opts, cases, out_path=opts.out)

    # Rebuild the config list in variant order, materializing aliases from the
    # entry they resolve to (same flags => same results by construction).
    by_name = {entry["name"]: entry for entry in payload["configs"]}
    ordered = []
    for item in items:
        name = item["config"]["name"]
        if name in by_name:
            entry = by_name[name]
            entry["alias_of"] = None
            ordered.append(entry)
        else:
            source = by_name[item["alias_of"]]
            entry = copy.deepcopy(source)
            entry["name"] = name
            entry["alias_of"] = item["alias_of"]
            ordered.append(entry)
    payload["configs"] = ordered
    payload["protocol"]["reference"] = "base"
    payload["protocol"]["candidate"] = None
    bench.finalize(payload, cases)

    payload["schema"] = SCHEMA
    payload["bench_schema"] = bench.SCHEMA
    payload["ablation"] = build_ablation(payload, items, aliases, args)
    bench.write_json_atomic(args.out, payload)

    bench.say("")
    bench.say(bench.render_markdown(payload))
    bench.say("")
    bench.say(render_ablation(payload))
    bench.say("")
    bench.say("[done] json: %s" % os.path.abspath(args.out))
    bench.say("[done] raw:  %s" % opts.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""collabosm - restore (or create) the Colab A100-80GB High-RAM session.

Why this script exists
----------------------
1. The local session record is disposable. `~/.config/colab-cli/sessions.json` is
   pruned whenever the runtime proxy token lapses, and the CLI then reports
   "No active sessions found" while the VM is still running and still billing.
   A session can be re-attached only when its recorded endpoint matches server truth.
   Observed twice on 2026-09-24, once mid-run with /content fully intact.

2. `colab new --gpu A100` never sends `shape`, so it is a lottery between
   80 GB High-RAM (167 GB RAM, 6.77 CU/h as Colab reports it) and 40 GB standard (83 GB RAM,
   5.37 CU/h by Colab's published figure).
   The 40 GB box cannot load this model: the 4.05 bpw pack is ~100 GiB on disk with
   ~63.6 GiB of weights that must be VRAM-resident. Eleven consecutive unpatched
   attempts gave 40 GB. This requests HIGH_RAM explicitly (google-colab-cli#47:
   Shape.HIGH_RAM exists and machineShape is parsed, but is never sent).

3. Verify VRAM before downloading. An unsuitable newly-created runtime is stopped;
   a pre-existing one is left running so this script cannot stop another workload.

Deliberately NOT done here: no keep-alive daemon is started. On a metered plan the
right default is "use it, then stop it"; a keep-alive turns a 2 h session into 24 h.
Pass --keepalive only when you are actively working and want the VM held open.

Exit codes: 0 ok | 2 bad args | 3 too many assignments | 4 assign failed
            5 cannot read assignments | 6 unsuitable box | 7 timeout
            12 unrecognized compatible assignment; explicit --force-new required
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY = os.path.expanduser("~/.config/colab-cli/sessions.json")
# High-RAM shapes verified on 2026-11-20: CPU, T4 and A100 (A100-SXM4-80GB,
# 79.3 GiB VRAM / 167.1 GiB RAM). G4 rejects shape=hm; L4/TPU have a single
# shape; H100 is untested (no entitlement) but the CLI exposes the choice.
HIGH_MEM_ACCELERATORS = frozenset({"CPU", "T4", "A100", "H100"})


def high_mem_supported(accelerator: str) -> bool:
    """Whether `shape=hm` is sent for this accelerator (2026-11-20 tests)."""
    return (accelerator or "").upper() in HIGH_MEM_ACCELERATORS


if HERE not in sys.path:
    sys.path.insert(0, HERE)
from colab_record import extra_fields, keep_alive  # noqa: E402 - either CLI version


def _import_colab_cli():
    """Import the Colab CLI from the running interpreter, else from the uv tool venv."""
    try:
        from colab_cli.common import state
        from colab_cli.state import SessionState
        return state, SessionState
    except Exception:
        pass
    # Added only when the current interpreter cannot import the CLI. The glob is
    # pinned to this interpreter's pythonX.Y: prepending a 3.13 site-packages to
    # a 3.12 venv shadows its own pydantic_core and fails with "unknown location".
    tag = "python%d.%d" % sys.version_info[:2]
    for cand in glob.glob(os.path.expanduser(
            "~/.local/share/uv/tools/google-colab-cli/lib/%s/site-packages" % tag)):
        if cand not in sys.path:
            sys.path.insert(0, cand)
    try:
        from colab_cli.common import state
        from colab_cli.state import SessionState
        return state, SessionState
    except Exception as exc:
        raise SystemExit(
            "cannot import google-colab-cli (%r).\n"
            "Install it with: uv tool install google-colab-cli\n"
            "Run this script with that tool venv python, e.g.\n"
            "  ~/.local/share/uv/tools/google-colab-cli/bin/python scripts/restore.py" % exc)


def local_registry():
    try:
        with open(REGISTRY) as fh:
            return json.load(fh)
    except Exception:
        return {}


def claimed_endpoints(except_name=None):
    """Endpoints already owned by another local session name.

    Two names pointing at one VM is how you end up stopping the wrong machine.
    """
    out = set()
    for name, rec in local_registry().items():
        if name == except_name:
            continue
        if isinstance(rec, dict) and rec.get("endpoint"):
            out.add(rec["endpoint"])
    return out


def attached_shape(assignment):
    return str(getattr(getattr(assignment, "machine_shape", None), "name", "?"))


def describe(assignment):
    accel = getattr(getattr(assignment, "accelerator", None), "value", "?")
    return "%s accel=%s shape=%s" % (assignment.endpoint, accel, attached_shape(assignment))


def register(state, SessionState, name, endpoint, token, url, accelerator, shape_note="",
             assignment=None):
    # with the token's expiry where the CLI keeps one (0.7): scripts/colab_record.py
    state.store.add(SessionState(name=name, token=token, url=url, endpoint=endpoint,
                                 variant=_variant(accelerator).value, accelerator=accelerator,
                                 **(extra_fields(SessionState, assignment) if assignment else {})))
    try:
        state.history.log_event(name, "session_registered",
                                {"endpoint": endpoint, "accelerator": accelerator,
                                 "shape": shape_note})
    except Exception:
        pass


def maybe_keepalive(state, name, endpoint, want):
    if not want:
        return
    try:
        from colab_cli.commands import session as cli_session
        spawn_keep_alive = getattr(cli_session, "spawn_keep_alive", None)
        if spawn_keep_alive is None:
            # 0.7 has no keep-alive daemon: kernel activity holds a box, and the
            # frontend's heartbeat (scripts/heartbeat.py) is that activity
            print("[restore] --keepalive: this CLI has no keep-alive daemon; the frontend's "
                  "heartbeat holds a box while it is in use")
            return
        pid = spawn_keep_alive(endpoint, name, auth_provider=state.auth_provider,
                               config_path=state.config_path)
        rec = state.store.get(name)
        if rec is not None:
            rec.keep_alive_pid = pid
            state.store.add(rec)
        print("[restore] keep-alive daemon started (pid %s) -- remember to stop it" % pid)
    except Exception as exc:
        print("[restore] could not start keep-alive: %r" % exc)


def _variant(accelerator):
    from colab_cli.commands.session import Variant
    return Variant.TPU if accelerator in {"V5E1", "V6E1"} else Variant.GPU


def create_high_ram(state, SessionState, name, accelerator, shape_code):
    """Assign a new runtime, actually sending `shape=hm` (the upstream gap).

    `st` sends no shape at all: Colab's default is the standard 40 GB shape (every
    unpatched draw in colab.log came back machineShape 0), while `shape=st` itself
    has never been sent and is not known to be accepted.

    `hm` is sent for the accelerators whose High-RAM shape was verified on
    2026-11-20 (CPU, T4, A100). G4 rejects `shape=hm`; L4/TPU have a single shape;
    H100 is untested but the CLI exposes it.
    """
    from urllib.parse import urljoin

    import requests
    from colab_cli import client as C
    from colab_cli.client import TUN_ENDPOINT, TooManyAssignmentsError, uuid_to_web_safe_base64

    orig = C.Client._build_assign_url

    def _build(self, notebook_hash, variant=None, accelerator=None, shape=None):
        url = urljoin(self.colab_domain, "%s/assign" % TUN_ENDPOINT)
        params = {"nbh": uuid_to_web_safe_base64(notebook_hash)}
        if variant:
            params["variant"] = variant.value
        if accelerator:
            params["accelerator"] = accelerator.value
        if shape_code == "hm" and high_mem_supported(getattr(accelerator, "value", "")):
            params["shape"] = shape_code      # the only added field
        return requests.Request("GET", url, params=params).prepare().url

    C.Client._build_assign_url = _build
    try:
        res = state.client.assign(uuid.uuid4(), variant=_variant(accelerator.value), accelerator=accelerator)
    except TooManyAssignmentsError:
        raise SystemExit(3)
    except Exception as exc:
        print("[restore] assign failed: %s: %s" % (type(exc).__name__, exc))
        raise SystemExit(4)
    finally:
        C.Client._build_assign_url = orig

    # Emit ownership before registration or verification can fail.
    print("COLLABOSM_CREATED_ENDPOINT %s" % res.endpoint, flush=True)
    register(state, SessionState, name, res.endpoint, res.runtime_proxy_info.token,
             res.runtime_proxy_info.url, accelerator.value, shape_code, assignment=res)
    granted = "?"
    try:
        for a in state.client.list_assignments():
            if a.endpoint == res.endpoint:
                granted = attached_shape(a)
    except Exception:
        pass
    print("[restore] created %s endpoint=%s granted shape=%s" % (name, res.endpoint, granted))
    return res.endpoint, granted


def colab_argv(exe=None):
    """The Colab CLI: `exe` ($COLAB, --colab) when given, else the CLI itself through
    colab_cmd.py under this interpreter -- the same package, and it also runs on
    Windows, where the `colab` command dies on its Unix-only console."""
    return [exe] if exe else [sys.executable, os.path.join(HERE, "colab_cmd.py")]


def probe(colab, session, attempts=8):
    """Run scripts/probe_gpu.py on the VM and parse its one-line JSON answer."""
    probe_path = os.path.join(HERE, "probe_gpu.py")
    for i in range(1, attempts + 1):
        try:
            out = subprocess.run(colab + ["exec", "-s", session, "--timeout", "30",
                                          "-f", probe_path],
                                 capture_output=True, text=True, encoding="utf-8",
                                 errors="replace", timeout=40,
                                 stdin=subprocess.DEVNULL).stdout
        except Exception:
            out = ""
        for line in out.splitlines():
            if line.startswith("PROBE "):
                try:
                    return json.loads(line[len("PROBE "):])
                except Exception:
                    pass
        if i < attempts:
            print("[restore] probe %d/%d did not answer yet; waiting 5 s" % (i, attempts))
            time.sleep(5)
    return None


def stop_session(colab, session):
    subprocess.run(colab + ["stop", "-s", session], capture_output=True, text=True,
                   encoding="utf-8", errors="replace", timeout=300, stdin=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser(description="restore or create the collabosm A100-80GB box")
    ap.add_argument("-n", "--name", default="collabosm")
    ap.add_argument("--accelerator", default="A100")
    ap.add_argument("--fallback-accelerators", default=os.environ.get("FALLBACK_ACCELERATORS", ""),
                    help="comma-separated fallback accelerators if primary assign fails (e.g. G4,H100)")
    ap.add_argument("--assign-retries", type=int, default=int(os.environ.get("ASSIGN_RETRIES") or 3),
                    help="attempts per accelerator before moving to fallback (default: $ASSIGN_RETRIES or 3)")
    ap.add_argument("--assign-retry-delay", type=float, default=float(os.environ.get("ASSIGN_RETRY_DELAY") or 3.0),
                    help="seconds to wait between assign retries (default: $ASSIGN_RETRY_DELAY or 3)")
    ap.add_argument("--shape", choices=["hm", "st"], default="hm",
                    help="hm = HIGH_RAM (80 GB VRAM / 167 GB RAM), st = standard (40 GB)")
    ap.add_argument("--min-vram-gib", type=float, default=70.0,
                    help="refuse a box below this and stop it only if newly created")
    ap.add_argument("--max-vram-gib", type=float,
                    help="refuse a newly-created box above this VRAM size")
    ap.add_argument("--colab", default=os.environ.get("COLAB"),
                    help="a Colab CLI executable (default: the CLI through colab_cmd.py)")
    ap.add_argument("--keepalive", action="store_true",
                    help="hold the VM open (bad on a metered plan)")
    ap.add_argument("--force-new", action="store_true",
                    help="create despite an unrecognized compatible assignment")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    state, SessionState = _import_colab_cli()
    colab = colab_argv(args.colab)

    try:
        live = list(state.client.list_assignments())
    except Exception as exc:
        print("[restore] cannot read assignments: %r" % exc)
        return 5

    mine = state.store.get(args.name)
    caught = claimed_endpoints(except_name=args.name)
    # Our record names a box the listing does not show: list_assignments has been seen
    # to leave out a live box for a few seconds. Creating a VM on that one snapshot
    # would leave the first one billing with no record, so ask twice more first.
    for _ in range(2):
        if not mine or any(a.endpoint == mine.endpoint for a in live):
            break
        print("[restore] %s is not listed; asking again in 5 s before treating it as gone"
              % mine.endpoint)
        time.sleep(5)
        try:
            live = list(state.client.list_assignments())
        except Exception as exc:
            print("[restore] cannot read assignments: %r" % exc)
            return 5

    # ---- 1. refresh our own registration, if the VM is still ours
    if mine and any(a.endpoint == mine.endpoint for a in live):
        a = next(a for a in live if a.endpoint == mine.endpoint)
        if args.accelerator in {"V5E1", "V6E1"} and a.accelerator != args.accelerator:
            print("[restore] existing session does not match the requested TPU")
            return 12
        keep_alive(state.client, mine.endpoint)          # 0.6 only; 0.7 has no ping
        register(state, SessionState, args.name, a.endpoint, a.runtime_proxy_info.token,
                 a.runtime_proxy_info.url, args.accelerator, attached_shape(a), assignment=a)
        print("[restore] re-issued token for existing registration: %s" % describe(a))
        maybe_keepalive(state, args.name, a.endpoint, args.keepalive)
        endpoint, granted, created = a.endpoint, attached_shape(a), False
    else:
        # An unregistered assignment can belong to another tool or machine.
        for a in live:
            mark = " (claimed by another local session)" if a.endpoint in caught else ""
            print("[restore] live: %s%s" % (describe(a), mark))
        def _supports_hm(acc_name: str) -> bool:
            return high_mem_supported(acc_name)

        def _parse_cand(raw: str, default_shape: str) -> tuple[str, str]:
            if ":" in raw:
                acc_part, shape_part = raw.split(":", 1)
                acc_name = acc_part.strip().upper()
                sh = shape_part.strip().lower()
                sh = sh if sh in ("st", "hm") else default_shape
                return acc_name, (sh if _supports_hm(acc_name) else "st")
            acc_name = raw.strip().upper()
            return acc_name, (default_shape if _supports_hm(acc_name) else "st")

        candidates = [(args.accelerator.upper(), args.shape if _supports_hm(args.accelerator.upper()) else "st")]
        for fb in (args.fallback_accelerators or "").split(","):
            fb = fb.strip()
            if not fb:
                continue
            pair = _parse_cand(fb, args.shape)
            if pair[0] and pair not in candidates:
                candidates.append(pair)
        candidate_set = {c[0] for c in candidates}
        unrecognized = [a for a in live if a.endpoint not in caught
                        and (getattr(getattr(a, "accelerator", None), "value", "") or "").upper()
                        in candidate_set]
        if unrecognized and not args.force_new:
            print("[restore] !! unrecognized compatible VM exists; refusing to adopt it or create"
                  " another without --force-new")
            return 12
        from colab_cli.commands.session import Accelerator
        endpoint = granted = None
        last_rc = 4
        attempts = max(1, args.assign_retries)
        for idx, (cand, cand_shape) in enumerate(candidates):
            acc = getattr(Accelerator, cand, None)
            if acc is None:
                print("[restore] unknown accelerator %s; skipping" % cand)
                last_rc = 2
                continue
            if idx > 0:
                print("[restore] trying fallback accelerator %s (shape %s)" % (cand, cand_shape))
            for attempt in range(1, attempts + 1):
                try:
                    endpoint, granted = create_high_ram(state, SessionState, args.name, acc, cand_shape)
                    break
                except SystemExit as exc:
                    last_rc = exc.code if isinstance(exc.code, int) else 4
                    if last_rc != 4:
                        return last_rc
                    if attempt < attempts:
                        print("[restore] %s assign attempt %d/%d failed; retrying in %.1fs..."
                              % (cand, attempt, attempts, args.assign_retry_delay))
                        if args.assign_retry_delay > 0:
                            time.sleep(args.assign_retry_delay)
            if endpoint is not None:
                break
        else:
            return last_rc
        created = True

    result = {"name": args.name, "endpoint": endpoint, "granted_shape": granted,
              "created": created}

    # ---- 3. verify the box before it costs anything
    if not args.no_verify:
        info = probe(colab, args.name)
        if info is None:
            print("[restore] !! the VM did not answer the probe; the kernel may still be starting")
            result["probe"] = None
            if args.max_vram_gib is not None:
                print("[restore] !! refusing a size-capped recipe without a VRAM measurement")
                if created:
                    stop_session(colab, args.name)
                    result["stopped"] = True
                if args.json:
                    print(json.dumps(result))
                return 11
        else:
            result["probe"] = info
            vram = info.get("vram_GiB")
            print("[restore] box: %s | vram %.1f GiB | ram %s GiB | cc %s | python %s | disk %s free"
                  % (info.get("device"), vram or -1, info.get("ram_GiB"), info.get("cc"),
                     info.get("python"), info.get("disk_free")))
            if vram is not None and vram < args.min_vram_gib:
                print("[restore] !! %.1f GiB VRAM is below the %.1f GiB this recipe needs."
                      % (vram, args.min_vram_gib))
                print("[restore]    Its weights and cache must be VRAM-resident, so this box")
                print("[restore]    cannot load it at all (Flash-Next: ~63.6 GiB of weights).")
                if created:
                    print("[restore]    stopping %s now, before any download; exit 6" % args.name)
                    stop_session(colab, args.name)
                    result["stopped"] = True
                else:
                    print("[restore]    leaving the pre-existing VM running")
                print("[restore]    re-run this script to draw a new box; the lottery is real.")
                if args.json:
                    print(json.dumps(result))
                return 6
            if args.max_vram_gib is not None and (vram is None or vram > args.max_vram_gib):
                print("[restore] !! refusing size-capped recipe: measured VRAM %s exceeds/does not meet cap %.1f GiB"
                      % ("unknown" if vram is None else "%.1f GiB" % vram, args.max_vram_gib))
                if created:
                    print("[restore]    stopping the newly-created %s before upload; exit 11" % args.name)
                    stop_session(colab, args.name)
                    result["stopped"] = True
                if args.json:
                    print(json.dumps(result))
                return 11

    if args.json:
        print(json.dumps(result))
    print("[restore] ready: session '%s' -> %s" % (args.name, endpoint))
    return 0


if __name__ == "__main__":
    sys.exit(main())

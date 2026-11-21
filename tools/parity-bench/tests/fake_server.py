#!/usr/bin/env python3
"""Fake llama-server used only by tests/smoke_cpu.py.

Implements the tiny surface bench.py needs (/health and
/v1/chat/completions), emits the real llama_moe_cache log lines on SIGTERM and
can simulate a failed load (FAKE_FAIL=1) or a crash mid-cases
(FAKE_DIE_AFTER_REQUESTS=N). No GPU, no model.
"""

import hashlib
import json
import os
import signal
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = 8090
for i, arg in enumerate(sys.argv):
    if arg == "--port" and i + 1 < len(sys.argv):
        PORT = int(sys.argv[i + 1])

FAIL_LOAD = os.environ.get("FAKE_FAIL") == "1"
DIE_AFTER = int(os.environ.get("FAKE_DIE_AFTER_REQUESTS", "0"))
LOAD_S = float(os.environ.get("FAKE_LOAD_S", "1.5"))
REQUESTS_PATH = os.environ.get("FAKE_REQUESTS_PATH")
CAPTURED = []
HOST = "127.0.0.1"


def log(line):
    print(line, flush=True)


def on_sigterm(signum, frame):
    log("srv  operator(): cleaning up before exit...")
    log("llama_moe_cache: ubatch <= 8: hits = 372, misses = 169, "
        "hit rate = 68.76%, uploaded = 120.50 MiB")
    log("llama_moe_cache: ubatch  > 8: hits = 96, misses = 512, "
        "hit rate = 15.79%, uploaded = 3400.00 MiB")
    log("llama_moe_cache: large batches: 96 experts copied from the cache, 3400.00 MiB")
    os._exit(0)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            if FAIL_LOAD:
                self._send(503, {"error": "loading"})
            else:
                self._send(200, {"status": "ok"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        prompt = body["messages"][0]["content"]
        max_tokens = int(body.get("max_tokens") or 16)
        if body.get("cache_prompt") is not False:
            log("fake: unexpected cache_prompt=%r" % body.get("cache_prompt"))
        if (body.get("chat_template_kwargs") or {}).get("enable_thinking") is not False:
            log("fake: unexpected chat_template_kwargs=%r"
                % body.get("chat_template_kwargs"))
        CAPTURED.append({
            "prompt_chars": len(prompt),
            "max_tokens": max_tokens,
            "temperature": body.get("temperature"),
            "cache_prompt": body.get("cache_prompt"),
            "chat_template_kwargs": body.get("chat_template_kwargs"),
        })
        if REQUESTS_PATH:
            with open(REQUESTS_PATH, "a") as f:
                f.write(json.dumps(CAPTURED[-1]) + "\n")
        if DIE_AFTER and len(CAPTURED) >= DIE_AFTER:
            log("fake: exiting mid-cases")
            os._exit(3)
        prompt_n = max(1, len(prompt) // 3)
        predicted_n = max(1, max_tokens)
        prompt_ms = prompt_n * 10.0
        predicted_ms = predicted_n * 20.0
        text = "reply-" + hashlib.sha256(prompt.encode()).hexdigest()[:12]
        self._send(200, {
            "choices": [{"message": {"content": text}, "finish_reason": "length"}],
            "timings": {
                "prompt_n": prompt_n,
                "cache_n": 0,
                "prompt_ms": prompt_ms,
                "predicted_n": predicted_n,
                "predicted_ms": predicted_ms,
            },
        })


def main():
    signal.signal(signal.SIGTERM, on_sigterm)
    signal.signal(signal.SIGINT, on_sigterm)
    log("llama_context: n_ctx = 32768, n_batch = 2048, n_ubatch = 512")
    log("llama_context: MoE cache auto: 6144.00 MiB for 17408.25 MiB of host experts "
        "(35.3% of the experts can be cached)")
    log("llama_moe_cache: CUDA0 MoE cache size =  6144.00 MiB for 17408.25 MiB "
        "of host experts")
    log("llama_context: MoE cache plan: CUDA0 = 6144.00 MiB")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    log("srv  listening on http://%s:%d" % (HOST, PORT))
    time.sleep(LOAD_S)
    if FAIL_LOAD:
        log("fake: load failure requested")
        time.sleep(1.0)
        return 2
    log("srv  load_model: model loaded")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())

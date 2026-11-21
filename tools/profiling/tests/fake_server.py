#!/usr/bin/env python3
"""llama-server falso para testar o fluxo de ponta a ponta sem GPU."""
import json
import os
import signal
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = 8098
T0 = time.time()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            if time.time() - T0 < 0.5:
                self._json(503, {"status": "loading"})
            else:
                self._json(200, {"status": "ok"})
        else:
            self._json(404, {})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/tokenize":
            text = body.get("content", "")
            self._json(200, {"tokens": [1] * max(1, len(text) // 4)})
        elif self.path == "/completion":
            text = body.get("prompt", "")
            ntok = max(1, len(text) // 4)
            time.sleep(0.3)
            self._json(200, {
                "content": "ok", "stop": True, "model": "fake",
                "timings": {
                    "prompt_n": ntok, "prompt_ms": 300.0,
                    "prompt_per_second": ntok / 0.3, "cache_n": 0,
                    "predicted_n": 1, "predicted_ms": 1.0,
                },
            })
        else:
            self._json(404, {})


def main():
    global PORT
    args = sys.argv[1:]
    if "--help" in args:
        print("usage: llama-server [options]")
        for f in ["-m", "-ngl", "-ncmoe", "--moe-cache-mib", "-c", "-b", "-ub",
                  "-fa", "--load-mode", "--lazy-mode", "-ctk", "-ctv", "--host",
                  "-np", "--no-warmup", "--metrics", "--no-webui", "--alias", "--port"]:
            print("  %s VALUE" % f)
        return 0
    if "--version" in args:
        print("fake llama-server 0")
        return 0
    for i, a in enumerate(args):
        if a == "--port":
            PORT = int(args[i + 1])
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    print("fake llama-server on %d" % PORT, flush=True)
    print("llama_moe_cache: CUDA0: hits = 100, misses = 24, hit rate = 80.65%, uploaded = 1234.50 MiB", flush=True)
    sys.stdout.flush()

    def on_term(signum, frame):
        print("llama_moe_cache: CUDA0: hits = 999, misses = 1, hit rate = 99.90%, uploaded = 42.00 MiB", flush=True)
        os._exit(0)

    signal.signal(signal.SIGTERM, on_term)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

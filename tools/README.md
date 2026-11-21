# tools

Development-only helpers. Nothing here is part of the app runtime: `frontend/collabosm.py`
never imports it, `scripts/provision.py` does not upload it to a Colab VM, and no recipe runs it.

## parity-bench

`parity-bench/` is the reproducible benchmark harness for the A100-40G parity work. It starts its
own `llama-server`, measures the fixed cases, VRAM, load time and the MoE-cache log lines, and writes
JSON plus Markdown tables. Python 3 standard library only; it does not build anything, download
models or touch the GPU by itself.

Read [parity-bench/README.md](parity-bench/README.md) for the protocol, config schema, acceptance
criteria and the CPU-only tests. From the repository root:

```bash
cd tools/parity-bench
python3 bench.py --self-test
python3 ablate.py --self-test
python3 tests/smoke_cpu.py
```

The real runs happen on a dedicated GPU VM (the README's `/content/...` paths), not on the machine
that serves the collabosm app.

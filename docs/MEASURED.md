# Measurements

All figures below were produced on **one Colab A100-80GB High-RAM** (79.3 GiB VRAM, 167 GB RAM, 12
vCPU) against `turboderp/Qwen3.8-Flash-Next-exl3` @ revision `4.05bpw_h6_ng6`
(sha `55a732e0c4c3d4614bc42b68493bb930d9b02c0a`, 107,463,600,896 B) with ExLlamaV3 1.5.1.
Runs reported here returned rc=0 with no errors unless stated.

## Prefill and decode, MTP `ndt=4`, Q4 KV, `-ngr`

Prompts were built to a target token count from the tokenizer and sent with a unique nonce, so
`cached_tokens = 0` and every number is a genuine cold prefill. "Second of pair" is the figure to
trust; both are shown.

| test | prompt tokens | prefill t/s (r0 / r1) | decode t/s (r0 / r1) | MTP accept |
|---|---:|---:|---:|---:|
| 4K pair | 4,143 | 2,518.8 / **2,498.4** | 74.7 / 75.0 | 41.7% |
| 30K pair | 30,256 | 2,805.2 / **2,805.8** | 74.8 / 75.0 | 41.7% |
| 114K pair | 114,097 | 2,707.9 / **2,793.3** | 74.3 / 74.5 | 41.7% |
| decode @85 ctx | 85 | — | **78.6** | 44.6% |
| decode @30K ctx | 30,254 | 2,809.4 | **97.4** | 63.9% |
| decode @114K ctx | 114,095 | 2,792.6 | **90.1** | 57.1% |

Without MTP the same setup measures a flat **~56 t/s** decode from context 0 to 130,816.

## Generator chunk size is the biggest prefill lever we found

30,261-token prompts, one value at a time, second run of each:

| `-gcs` | 1024 | 2048 | 4096 | 8192 |
|---|---:|---:|---:|---:|
| prefill t/s | 2,204 | 2,806 | 3,360 | **3,882** |

Two things to know:

- **The first run at any new chunk size pays a one-time kernel autotune.** `-gcs 8192` read 1,674 t/s
  on its first run and 3,882 t/s on its second. This is exactly why the pair matters; a single-run
  sweep would have concluded the opposite.
- We had been running 2048 everywhere (the library default), leaving 20-38% on the table.

We have **not** tested 16384. That is the obvious next experiment.

## A measurement-integrity warning about `eval/perf.py`

The official harness is not a direct long-context measurement. In `eval/perf.py`:

```python
pre_time = 0
if length >= chunk_size * 2:
    pre_time = (length // 2) / results[length // 2]   # inherited, not measured
results[length] = length / (pre_time + t.interval)
```

Every length >= 2 x chunk_size is inherited from a shorter measurement, which is why its published
curve looks flat from 4096 to 131072. With `-chunk_size 2048` only the rows below 4096 are measured;
the rest are extrapolation. Use it for shape, not for headline numbers — and prefer a runner that
sends a fresh prompt per length.

Its no-MTP result matches ours where it *is* measuring: 2,909-3,234 t/s prefill, ~56 t/s decode.

## Warm KV reuse, measured

This is the feature that makes an interactive server pleasant, and it is two separate mechanisms.

**In VRAM (prompt caching).** Same 32,026-token prompt twice inside one Generator:

| run | prompt tokens | cached | hit | wall |
|---|---:|---:|---:|---:|
| cold | 32,026 | 0 | 0% | 15.62 s |
| repeat | 32,026 | 32,000 | **99.9%** | **0.45 s** |

**From pinned host RAM (`-ccs`).** Same prompt, but a second, different 60,031-token request in
between to force the first one's pages out of VRAM:

| step | prompt | cached | hit | prefill t/s | wall | tier entries | live pages evicted | pages restored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A cold | 32,026 | 0 | 0% | 2,100.8 | 15.62 s | 0 | 0 | 0 |
| A again | 32,026 | 32,000 | 99.9% | — | 0.45 s | 0 | 0 | 0 |
| B (different, 60K) | 60,031 | 0 | 0% | 3,362.0 | 18.33 s | 104 | 104 | 0 |
| **A after eviction** | 32,026 | 32,000 | **99.9%** | — | **0.58 s** | 208 | 208 | **104** |
| A once more | 32,026 | 32,000 | 99.9% | — | 0.39 s | 209 | 209 | 104 |

`alloc_tier_pages = 104` is the proof it came from host RAM and not from a surviving VRAM page: the
engine counts restores separately from VRAM hits. Against the fair baseline -- the 9.72 s re-prefill
of the `-ccs 0` control below -- the restore is **16.8x faster**. (Against the 15.62 s cold row it
reads 27x, but that row also paid the one-time kernel autotune.)

**Control: the identical sequence with `-ccs 0`.** Same prompts, same eviction pressure (104 live
pages evicted in both runs), only the tier differing:

| step | `-ccs 8` | `-ccs 0` |
|---|---|---|
| A cold | 15.62 s | 9.85 s |
| A again | 32,000 cached (99.9%), 0.45 s | 32,000 cached (99.9%), 0.44 s |
| B different (60,031 tok, evicts A) | 18.33 s, 104 pushed | 18.12 s, 104 evicted, no tier |
| **A after eviction** | **32,000 cached (99.9%), 0.58 s** | **0 cached (0%), 9.72 s** |
| A once more | 0.39 s | 32,000 cached, 0.40 s |
| `stashes_stranded` | **0** | 6 |

So the tier turns a 9.72 s re-prefill into a 0.58 s restore — **16.8x** — and without it the
Gated-DeltaNet checkpoints are orphaned by KV eviction (`stashes_stranded: 6`). The `-ccs 8` cold row
is worse only because it was the first generation after load and absorbed the one-time kernel autotune;
the B rows agree (18.33 vs 18.12 s). Design notes in `docs/RUNBOOK.md`.

## KV cache

The engine reports its own cache sizing, which matches the arithmetic exactly:

| quantity | value |
|---|---|
| `bytes_per_token` (reported) | **10,752.0** |
| layers with a KV cache | 12, all `CacheLayer_qsa_quant` |
| cache at 262,144 tokens | 2,818,572,288 B (2.63 GiB) |
| VRAM used with cache loaded | 73,589 / 81,920 MiB |
| recurrent (`GDNLayerState` + `PLELayerState`) | 572,395,568 B (546 MiB), constant with context |

Per full-attention layer per token: K 256 B + V 256 B + scales 64 B + QSA indexer planes 320 B = 896 B,
times 12 layers. The QSA planes stay fp16 regardless of `-cq`, so they are 36% of the cache and cap
the Q4 saving at ~62% versus fp16, not 75%.

| context | Q4 KV cache |
|---|---:|
| 262K | 2.63 GiB |
| 500K | 5.01 GiB |
| 1M | 10.0 GiB |

Sizing guidance and the split-versus-unified argument are in `docs/RUNBOOK.md`.

## Model geometry (from the pack's own `config.json`)

| item | value |
|---|---|
| layers | 48 — 36 linear attention (Gated DeltaNet) + 12 full attention |
| `full_attention_interval` | 4 |
| KV heads / head_dim | 2 / 256 |
| QSA indexer | head_dim 128, compress ratio 4, budget 2048 |
| native max position | 262144 |
| experts | 512, top-10 |
| vision | `vision_k6.safetensors` (561 MB) |
| n-gram/PLE table | `ngram_embedding.safetensors` (36.36 GiB, host RAM via `-ngr`) |
## Verified end to end, 2026-09-25

One account, one box, no manual steps beyond `scripts/`:

| step | result |
|---|---|
| `scripts/restore.py` | re-attached an **orphaned** assignment after the Colab CLI dropped its local record for the third time that day (VM alive and still billing); confirmed 79.3 GiB VRAM, 167.1 GiB RAM, cc 8.0, python 3.13.15 |
| `scripts/serve.sh` | loaded the 4.05 bpw pack at `cache_size 500224` in **259.5 s**; `/health` 200 at **76,437 / 81,920 MiB** VRAM |
| through the public tunnel | `/v1/models` **401 without the key, 200 with it**; a real chat completion answered in **3.1 s** |

Runtime flags for that run: `-cq 4 -ndt 4 -gcs 4096 -ccs 16 -rcs 16 -ngr`.

Two facts about the served model that cost real time to find:

- **It is a reasoning model with thinking on by default.** A `max_tokens` under ~1K returns a
  truncated reasoning trace rather than an answer; this looks like a broken server and is not one.
  (`api_server.py` has since turned thinking off unless a request asks for it.)
- **`Generator.generate()` returns `(completions, last_results)`.** Reading `text` out of
  `last_results` yields only the *final fragment* of the completion, so an early `api_server.py`
  answered `" inputs"` to a real question. The server now unpacks the tuple and asks for
  `completion_only=True`. This is the same class of bug as the per-request `Generator`: invisible,
  and it does not raise.

## Qwen3.8-27B on the A100-40G (2026-09-27)

One Colab A100-SXM4-40GB (40,960 MiB VRAM, 83.5 GiB RAM) against `turboderp/Qwen3.8-27B-exl3` @
`SC_4.00bpw_H5_V6` (sha `516bf129059031c6da9416768ea6b7a1be00a8fc`, 16,384,243,706 B), ExLlamaV3
1.5.1, **Q8 KV and a 262,144-token cache, no YaRN**, MTP `ndt=4`, `-gcs 8192`. Measured with
`python scripts/bench_speed.py`: chat completions through the frontend on 3020, a unique nonce on
every cold prompt, the server's own timings.

| measure | value |
|---|---:|
| billing to READY | 5 min 51 s (load 114.3 s; a 43.6 s warm-up since cut to ~8K tokens) |
| VRAM in use | 26,746 MiB |
| host RAM in use | 36.7 of 83.5 GiB |
| cold prefill, 4K (second of a pair) | **2,935 t/s** |
| cold prefill, 32K (second of a pair) | **2,544 t/s** |
| cold prefill, 119K (both of a pair) | **1,435 t/s** (82.8 s) |
| decode, short context, 512 tokens | **62.9 t/s** |
| decode, 32K context, 512 tokens | **50.4 t/s** |
| decode, ~128K context (a Codex turn) | 43.9 t/s |
| a follow-up: 32,512 cached + 78 new tokens | 1.1 s before the first token |
| a Codex turn: 127,744 cached + 510 new tokens | 2.9 s before the first token |

- **The last two rows explain the rail's low "prefill" on agent turns.** There it can read 70-180
  t/s. Almost everything is cached, and what is left is a cost paid once per turn, divided by a few
  hundred new tokens. That cost covers:
  - the recurrent state for the cached prefix;
  - attention over the whole context for the new tokens;
  - the first decode step.
- **Why decode is below Flash-Next's (75-97 t/s on the 80G card).** This is a dense 27B: every token
  reads ~13 GB of weights, where Flash-Next's 6B active parameters read ~3 GB. And the 40 GB card
  has 1.55 TB/s of bandwidth against the 80 GB card's 2.04.
- **The recipe has since moved on, unloaded.** It now asks for Q4 KV and two 409,600-token
  conversations (YaRN 1.5625, cache 819,200). That setting has not been loaded yet. Q4 reads half the
  KV per token, so long-context decode should come out a little faster.

## Strata v0.1.41 on the A100-40G (2026-10-09)

One paid A100-SXM4-40GB trial (40,960 MiB VRAM, 83.5 GiB host RAM), NVIDIA driver 580.82.07,
installed CUDA compiler 13.0.88. The engine was built on the VM from official
`Niko1221/Strata` commit `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`; its built binary SHA-256 was
`81c5e113c48292a723d2cb4ff2613d7be4a79ee559f32292cd6ff07670e10271`. The existing
`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` IQ3_S revision
`2a55d75962e22f7a4a1d9963ab6eae3678537831` was unchanged. Runtime config:
262,144 context, int8 KV, 32,768 resident KV positions, GPU vision, MTP spec 4. GPU assignment to
ready took about **29 min** (one observation, including source build); the n-gram table was warmed
after the engine loaded.

Python standard-library source text was sliced without repetition *within this run*, with unique
nonces. All table rows below use the engine's own timing fields and actual API token usage, not
estimated target lengths; prompt KV cache reported zero. The corpus comprised 64 fixed files,
SHA-256 `119ce3f85eb8c2792a1be276c36756560eeb407874f34cda15dd9ee52613848f`.

| Actual input tokens | Output tokens | Prefill t/s | Decode t/s |
|---:|---:|---:|---:|
| 12,878 | 16 | 1,320.9 | 106.7 |
| 26,816 | 16 | 1,492.8 | 115.0 |
| 26,451 | 16 | 1,591.8 | 96.1 |
| 52,323 | 16 | 1,951.0 | 116.7 |
| 56,363 | 16 | 2,332.8 | 99.4 |
| 110,921 | 16 | 2,515.8 | 104.0 |
| 110,072 | 16 | 2,566.9 | 68.7 |
| 2,900 | **512** | 939.1 | **87.6** |
| 21,672 | **512** | 2,068.8 | **93.0** |

An additional 12,160-token / 2,438.5 prefill t/s row is **excluded from headline results**:
its source prefix had already appeared in an aborted run. Zero KV-cache tokens do not establish
that Strata's row or expert caches were cold. The 16-token decode rates are not substitutes for
the two 512-token decode probes. Samples were not repeated enough to estimate variance.

The prior v0.1.24 trial recorded 58.8 decode t/s after a short prompt and 66.1 after about 41K
input tokens. Comparing those with 87.6 and 93.0 above suggests **+49% and +41%**, respectively,
but the runs used different prompts, contexts and VMs: **not a same-box A/B**. Likewise, the old
prefill observations (1,448 at 16K; 2,082 at 36K; 2,492 at 70K; 2,497 at 144K) do not establish
an across-the-board prefill improvement for v0.1.41.

Functional smoke checks passed for chat text, Unicode streaming, a 256×128 red-left/blue-right
image fixture, function-tool call and result round-trip, and `/v1/responses` final text and
streaming. Responses was tested as a stateless full-history API; response-ID retrieval/cancel/
compact, a full Codex `apply_patch` workflow, full-window context load, and concurrency were **not**
tested. The paid GPU was stopped; Colab reported zero assignments and zero hourly rate afterward.

## Project Maya v1.0.30 Maya-L on A100-80G High-RAM (2026-10-10)

One direct, text-only VM trial used a verified NVIDIA A100-SXM4-80GB (81,920 MiB VRAM),
167.1 GiB host RAM, Colab `shape=hm` and 6.77 CU/h. The official
[`mw00/project-maya` v1.0.30](https://github.com/mw00/project-maya/releases/tag/v1.0.30)
source was pinned to `c189f7ac8114ca039e9643314d1fead014fc6ff5`. Before download,
[`peasantsmith/GLM-5.3-Flash-Maya-GGUF`](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF)
`main` resolved to `f16ee4ba50d8fd2bb726ea1862ca6ca14ba188da`; upstream's installer used
`main` URLs but verified the four published per-shard SHA-256 hashes. The new collabosm adapter
pins the URL revision as well. The four GGUF shards totalled 156.319 GB. The routed experts use
IQ3_S gate/up and IQ4_XS down, with Q5_K for
sensitive tensors and Q6_K attention: **Maya-S and Maya-M were not selected**, because their
2-bit components do not meet the requested 3-bit floor. Setup used `--no-vision` and 32,768
context; this trial makes no vision claim.

The pinned source build, download and pack setup returned exit 0 in **76.8 minutes**. The first
load warmed expert tiers for roughly another **8 minutes**. Colab Python 3.13 lacked `ensurepip`,
so the project-private venv required `--without-pip` and a pinned pip bootstrap. Port 8080 was
occupied by another VM process, so the Maya server bound to loopback port 18080. The API returned
an actual text completion and `/v1/status` engine timings:

| Actual input | Output | Cached prompt | Prefill | Decode | GPU memory used |
|---:|---:|---:|---:|---:|---:|
| 11,515 tokens | 256 tokens | 0 | **1,084.1 tok/s** | **16.2 tok/s** | 81,169 / 81,920 MiB |

Engine time was 10,621.7 ms prefill and 15,781.5 ms decode; 26.42 s wall time. This is **one
sample**, not a median or a model-quality assessment. The completion ended at the requested
256-token length. The model was accessed directly on the VM, not through a collabosm recipe,
tunnel, or Codex. Maya v1.0.30's server source exposes Chat Completions but has no
`/v1/responses` route; do not infer that Codex Responses works. After the trial, `colab stop`
terminated the session; the account reported no active sessions, zero assignments and zero hourly
rate. Any collabosm Maya recipe remains **not end-to-end verified** until its own provisioning,
proxy, stop and API surface are run on a separately authorized session.

**Subsequent offline recipe change (2026-10-10):** the configured window is now 131,072 tokens
(128K), with FP16 KV unchanged and `STRATA_GLM_RESERVE_MB=2048` in its persistent engine config.
The larger state is allocated before the expert cache is sized, and the 2 GiB reserve replaces
the default 700 MiB outside that pool. Thus the GPU expert cache gets less space; the model's
experts and quantization are unchanged. The model's native context metadata is corrected to
1,048,576, separate from the configured 128K window. **No GPU was started for this change.**
All measurements above remain evidence for the previous 32K/default-reserve configuration,
not measured throughput, memory use, or full-window validation of the new 128K recipe.

## What Colab charges, and what it counts as use (2026-09-25/26)

| item | value | source |
|---|---|---|
| A100-80G High-RAM rate | **6.77 CU/h** | `/tun/m/ccu-info` `consumptionRateHourly` with this box as the account's only assignment (older notes: 7.52) |
| unattended box reclaimed | **within 25 min** of the last `colab exec` | CLI history: last exec 02:35:37 UTC, assignment list empty at 03:00:05; the service was up and nothing sent a keep-alive |
| box reclaimed **although pinged** | **22 min** after the last `colab exec`, **40 s** after a successful keep-alive ping | CLI history: last exec 21:07:24 UTC (end of provisioning), then only `download`s; the frontend pinged every 3 min while Codex traffic flowed (last ping ok 21:28:41); the tunnel answered 530/1033 from ~21:29:20; assignment list empty at 21:31:10 |
| boxes that lived | 3.7 h and 7 h | kernel use at least every ~15 min, or a `colab console` open; the longest stretch any box survived with no kernel use was 28.8 min (2026-09-25) |
| shape sent for a 40 GB card | none | 13 unpatched `assign` calls in `colab.log` came back `machineShape 0`; `shape=st` has never been sent |

Chat through the tunnel does not count as use, and neither, on its own, does the keep-alive ping:
the second box above was pinged every 3 minutes. The kernel does, as google-colab-cli 0.7.4's README
now says too ("Sessions stay alive as long as the kernel is active"; 0.7.4 dropped the keep-alive
ping altogether). So while a box is in use the frontend runs scripts/heartbeat.py on the kernel
every 10 minutes (`colab_keepalive.py --heartbeat`, `docs/RUNBOOK.md`, cost guardrails). The
heartbeat is not yet proven on a live box: the next session running longer than 30 minutes on tunnel
traffic alone is that test.

The CLI's local session record goes missing about once an hour in every long run, the VM intact.
The CLI drops it when `list_assignments` briefly leaves the endpoint out, and when a call to the VM
is refused (401/404). An expired token would get exactly that refusal: the record keeps the
runtime-proxy token it was registered with, and `RuntimeProxyInfo` carries `tokenExpiresInSeconds`.
Which of the two causes it is not measured yet. The keep-alive now refreshes the record's url and
token from the live assignment on every call, and re-registers a record that is already gone.

## Capacity

The fitted VRAM line, the per-slot recurrent cost, the session table and the reasons the host-RAM
tier cannot extend live capacity are in `docs/CONCURRENCY.md`.

# beellama on the A100-40G: Qwen3.8-Flash-Next GGUF (2026-11-21)

Goal: run Qwen3.8-Flash-Next (ISTA-DASLab GSQ-RCO GGUF) on the 40 GiB A100 with the
user's beellama fork instead of the ExLlamaV3/Strata engines.

Sizing that decides it (sizes from the HF tree at revision
`2a55d75962e22f7a4a1d9963ab6eae3678537831`; only shard 1 has to be resident, shard 2 is
the 28.8 GB n-gram table and can stay mmap'd):

| Quant | shard 1 | shard 2 | total | task avg |
|---|---:|---:|---:|---:|
| Q2_0 | 37.6 GB | 28.8 GB | 66.4 GB | 89.07 |
| IQ2_XS | 39.2 GB | 28.8 GB | 68.0 GB | 89.16 |
| IQ3_XXS | 47.0 GB | 28.8 GB | 75.8 GB | ~91.0 |
| IQ3_S | 54.8 GB | 28.8 GB | 83.6 GB | ~93.1 |

The A100-40G box has 39 GiB of VRAM, ~83 GiB of host RAM and 236 GB of disk (measured
2026-11-21), so every size fits on disk; Q2_0's shard 1 almost fits VRAM by itself.

Recipes added (engine = beellama, `llama.sh`):

| Recipe | placement | status |
|---|---|---|
| `a100-40g/qwen38-fn-q20-beellama` | `--fit on` (fit-target 4096 MiB), ctx 8192, b 512/ub 128, cache 0 | test 2026-11-21 |
| `a100-40g/qwen38-fn-q20-ncmoe-beellama` | deterministic: `-ncmoe 8`, no fit | prepared |
| `a100-40g/qwen38-fn-iq3s-beellama` | `--fit on`, IQ3_S | prepared |

`llama.sh` now forwards `fit`, `fit-target`, `n-cpu-moe`, `moe-cache-mib` and
`vram-reserve-mib` from the recipe environment. The `FALLBACK_ACCELERATORS` of these
recipes is empty on purpose: a fallback would land on the A100 High-RAM 80 GB and the test
would no longer be about the 40 GiB card.

## Expert cache measurements (RTX 4070, OLMoE-1B-7B q2_k)

- `--moe-cache-mib -1` sizes the cache after the compute buffers are reserved:
  `MoE cache auto: 2224.00 MiB for 2224.00 MiB of host experts (100.0%)`, hit rate
  51.8-68.8% on short ubatches.
- `--moe-host-pinned` (default on) keeps host experts in `CUDA_Host`
  (`cudaMallocHost`): 34.3/34.4/34.3 t/s vs 33.9/33.8/33.9 t/s for pageable memory
  (+1.5%, three runs each).
- Limit: the CUDA host buffer accepts 4 GiB per allocation, and the loader allocates all
  host experts in one buffer. Above that (e.g. Flash-Next with ~17 GiB of host experts)
  the allocation falls back to pageable memory; a chunked `cudaHostRegister` arena is the
  next step.

# Qwen3.8-Flash-Next Q2_0 on the A100-40G with beellama (2026-11-21)

First working configuration of Flash-Next GGUF on the 39 GiB card with the user's fork,
after the load crash was traced to an inconsistent (incremental) build of the Colab zip
(see `docs/FN40-CRASH-ROOTCAUSE.md` in the beellama repo).

Engine: beellama sm_80, release `collabosm-a100-clean-20261121` (clean build).
Model: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` revision `2a55d759…`, Q2_0
(shard 1 37.6 GB + shard 2 28.8 GB, 66.4 GB total, both on `/content`).
Box: A100 40 GB (40,960 MiB), 83 GiB host RAM, 236 GB disk.

## Configuration

    -ngl 99 --n-cpu-moe 2 --moe-cache-mib -1        # 2 layers' experts in RAM, the rest cached in VRAM
    -c 32768 --load-mode mmap --lazy-mode on       # the 28.8 GB n-gram table stays mmap'd
    -b 2048 -ub 512 -fa on                         # prefill throughput
    KV F16 (server default at measurement time; the recipe sets q8_0)

`--moe-cache-mib -1` is the automatic size: it is resolved **after** the first
`sched_reserve()`, so it uses the VRAM the compute buffers left, capped by the host expert
bytes (see `docs/PLAN-strata-parity.md`, P1/P2).

## Measured (two runs, same config, same VM)

| Metric | Run 1 | Run 2 |
|---|---:|---:|
| Load (page cache warm) | 17 s | 17 s |
| VRAM after load | 37,838 MiB | 37,838 MiB |
| Decode, short (24 prompt / 128 out) | 54.08 t/s | 53.91 t/s |
| Prefill, 4,622 tokens cold | 1,157.8 t/s | 1,157.7 t/s |
| Decode after that prefill | 46.41 t/s | 46.42 t/s |
| Prefill, ~30k tokens cold | 1,175.6 t/s | 1,175.5 t/s |
| Decode after that prefill | 44.50 t/s | 45.71 t/s |

## What the expert cache bought (same card, same model)

| Placement | VRAM | decode short | prefill 4k | decode 7k |
|---|---:|---:|---:|---:|
| `-ncmoe 8`, cache off | 31,600 MiB | 38.7 t/s | 314 t/s | 31.8 t/s |
| `-ncmoe 8`, cache auto | 37,002 MiB | 50.6 t/s | 354 t/s | 43.3 t/s |
| `-ncmoe 4`, cache auto | 36,998 MiB | 52.0 t/s | 443 t/s | 46.0 t/s |
| `-ncmoe 2`, cache auto, b512/ub128 | 36,994 MiB | 54.3 t/s | 513 t/s | 46.8 t/s |
| `-ncmoe 2`, cache auto, b2048/ub512 | 37,094 MiB | 55.7 t/s | 1,159.6 t/s | 46.4 t/s |
| `-ncmoe 2`, cache auto, b4096/ub1024 | 37,228 MiB | 52.8 t/s | 1,482.7 t/s | 33.2 t/s (4k) |


> Correction (2026-11-21, later): the runs above used the server default **F16** KV cache.
> The recipe sets LLAMA_KV_TYPE=q8_0; the q8_0 numbers are being re-measured on the same A100
> before the recipe facts are updated.

Reading: the automatic cache is worth +30% to +87% decode; the batch/ubatch raise is worth
+126% prefill, at the cost of a slightly worse decode at 4k with ub 1024. The chosen
`b 2048 / ub 512` keeps both.

Reference on the same class of card (Strata engine, IQ3_S): prefill 1,320–2,567 t/s,
decode 87.6–93.0 t/s. So beellama now sits near the low end of Strata's prefill and about
half its decode — with Q2_0 instead of IQ3_S.

## Recipe

`a100-40g/qwen38-fn-q20-beellama` (status `verified`), engine beellama, clean zip
`collabosm-a100-clean-20261121`, `LLAMA_STREAM_VRAM=1` for the HF->VRAM path.

## Strata vs beellama, same A100-40G, same Q2_0 weights (2026-11-21)

Same VM (`fn40keep`, A100 40 GB, 83 GiB RAM), same model files
(`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` Q2_0 in `/content/fn40/real`), context 32768,
same prompts (short; 4,662-token code block; ~28.6k-token code block), greedy.

| Metric | Strata 0.2.17 (prebuilt engine) | beellama sm_80 (clean build) | beellama / Strata |
|---|---:|---:|---:|
| Load (fresh process, page cache warm) | 197 s experts + cache fill, ready at 210 s | 17 s | 0.08× |
| VRAM after load | 38,220 MiB | 37,838 MiB | — |
| Decode, short (64 prompt / 128 out vs 24 / 128) | **85.9 t/s** | 54.0 t/s | 63% |
| Prefill, 4.6k tokens | **2,394.7 t/s** | 1,157.8 t/s | 48% |
| Decode after 4.6k | **67.4 t/s** | 46.4 t/s | 69% |
| Prefill, ~28.6k tokens | **4,629.3 t/s** | 1,175.6 t/s | 25% |
| Decode after 28.6k | **75.0 t/s** | 45.1 t/s | 60% |

Strata settings: `--max-context 32768 --kv int8`, `--expert-cache auto` (24,576 experts,
31.64 GiB of VRAM), `--prefill auto`, DFlash2 drafter. beellama settings:
`-ncmoe 2 --moe-cache-mib -1 -c 32768 -b 2048 -ub 512 -fa on`, KV F16, no spec decode.

What the numbers say:

- beellama loads **12× faster** when the page cache is warm (17 s vs 210 s): mmap plus the
  36 GB n-gram table left on disk, against Strata copying 31.6 GiB of experts into VRAM at
  0.33 GiB/s.
- Strata is **1.6× faster on short decode and 1.5× at long context**, and **2-4× faster on
  prefill**, where the gap grows with the prompt: its prefill scales 2,395 → 4,629 t/s from
  4.6k to 28.6k tokens, while beellama stays flat at ~1.17k.
- Both fit the card with essentially the same VRAM (38.2 vs 37.8 GiB).

The prefill scaling with context is the clearest next target: beellama's GPU-resident
experts are only refreshed per batch, while Strata pipelines the next layer's experts over
PCIe during the current layer's attention. That is the P5 item in
`docs/PLAN-strata-parity.md` (beellama repo).

## A100-40G: QSA sparse with q8_0 KV and the MTP sidecar (2026-11-21)

Same A100-40G, Qwen3.8-Flash-Next Q2_0, `-ncmoe 2`, `-c 32768 -b 2048 -ub 1024 -fa on`,
`--load-mode mmap --lazy-mode on`, KV `q8_0`, `-np` default (4 slots). No-spec configs:
3 runs each, cache 6144 MiB. MTP: 1 run, auto cache + `--vram-reserve-mib 4096`.
Engine: `feat/strata-parity` @ `e0ee7098a`, Ubuntu 24.04 chroot build (sm_80).

### QSA sparse attention with a Q8_0 KV cache

| config | short tg | pp4k | decode 4k | pp28k | decode 28k |
|---|---:|---:|---:|---:|---:|
| base-f16 (release) | 56.75 | 1,461.6 | 47.08 | 1,481.5 | 44.26 |
| base-q8 (release) | 55.73 | 1,484.5 | 46.04 | 1,513.9 | 39.62 |
| new-f16 (QSA build) | 56.24 | 1,462.3 | 47.35 | 1,481.7 | 44.22 |
| new-q8 (QSA build) | 55.59 | 1,485.8 | 46.40 | **1,516.4** | **43.53** |

- QSA q8 vs base q8: prefill **+0.2%** at 28k (within run noise), decode after 28k
  **+9.9%** (43.53 vs 39.62), short and 4k decode unchanged.
- The phase-1 target (+25% long prefill) was **not met**. The MoE cache is bypassed for
  batches larger than 32 tokens (`src/llama-moe-cache.cpp`), so the 28k prefill uploads
  roughly 39.7 GB of host experts; that dominates over the attention saving.
- Greedy output is identical to the base on short/4k; at 28k+dec the new-q8 text differs
  from the base (sparse gather reduces a different set of masked columns). The kernel
  passes the 28/28 `test-backend-ops` NMSE matrix on CUDA.
- VRAM: 37,550 MiB for both base-q8 and new-q8 (cache 6144).

### MTP with the unsloth sidecar

`-md MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf --spec-type draft-mtp
--spec-draft-n-max 2 --spec-draft-type-k q8_0 --spec-draft-type-v q8_0`,
`--moe-cache-mib -1 --vram-reserve-mib 4096`.

| config | VRAM ready | pp4k | decode 4k | pp28k | decode 28k | acceptance |
|---|---:|---:|---:|---:|---:|---:|
| no-spec q8 | 37,550 | 1,484.5 | 46.04 | 1,513.9 | 39.62 | — |
| MTP sidecar n-max 2 | 39,052 | 1,227.2 | **61.47** | 1,293.0 | **58.86** | **80.7%** (138/171) |

- Decode: **+33%** after 4k, **+49%** after 28k; short decode 53.96 vs 55.73 (no gain).
- Prefill: **-17%** at 4k and **-15%** at 28k. The target computes the NextN hidden
  state for every prompt token so the draft can catch up once; that is the cost.
- Fixed cache 2048 or 4096 reaches 40,404 MiB and OOMs on the first MTP request;
  `n-max 4` fails at load. Auto cache plus reserve 4096 works at 39,052 MiB.
- The QSA-patched sidecar (`compress_ratios[48]` 0 -> 4) **fails to load**; use the
  original file.
- Per-request acceptance from the server log: 0.70-0.86; the adaptive draft-max
  controller settles on 2.

### What is next

The two levers with evidence:
1. Use the MoE cache during prefill when the host-expert working set fits the cache
   (the 2 host layers fit in 6 GB). Prototype in progress; target is +10% at 28k.
2. Reserve the sidecar weights in the auto-size so MTP can keep a real expert cache
   instead of the current reserve-only layout. Prototype in progress.

## A100-40G — A/B das flags Strata (2026-11-21)

Engine `parity5` (`feat/strata-opt` @ `f5b72f42a`), 2 runs per config, Qwen3.8-Flash-Next Q2_0,
Q8_0/Q8_0, `b 2048 / ub 1024`, `-ncmoe 2`, cache 6144 MiB unless noted. Same prompts as the
earlier parity table.

| config | short tg | pp4k | pp28k | decode 28k |
|---|---:|---:|---:|---:|
| base-q8 (release) | 48.31* | 1,410.8 | 1,511.3 | 39.63 |
| p0, prefill flag off | 56.34 | 1,486.1 | 1,514.0 | 43.55 |
| p1, `LLAMA_MOE_CACHE_PREFILL=1` | 56.76 | 1,556.8 | **1,567.4** | 44.11 |
| m1, p1 + `LLAMA_MOE_MAP_DEVICE=1` | **59.72** | 1,565.4 | 1,574.0 | 44.40 |
| MTP, `LLAMA_MTP_RESERVE_WEIGHTS=0` | 55.58 | 1,231.3 | 1,292.2 | 53.14 |
| MTP, `LLAMA_MTP_RESERVE_WEIGHTS=1` | **59.95** | 1,234.6 | 1,292.3 | **57.70** |
| MTP, reserve 1 + `LLAMA_MTP_KV_STORE_ONLY=1` | 56.80 | 1,241.9 | 1,311.6 | 58.53 |
| overlap 1 + ring 0/1 (cache 1024) | 56.6–56.8 | 1,500–1,502 | 1,536–1,538 | ~44.1 |
| `LLAMA_QSA_INDEXER_BOUNDS` 0/1 | 59.8/59.8 | 1,564/1,580 | 1,573/1,568 | 44.4/44.3 |
| `LLAMA_GDN_CONV_L2` 0/1 | 59.6/60.6 | 1,564/1,567 | 1,573/1,576 | 44.3/44.3 |

\* base short caught the cold first run (39.8 t/s); the second run was 56.9.

Verdict:

- `LLAMA_MOE_CACHE_PREFILL`: **+3.5%** at 28k and **+4.8%** at 4k, decode unchanged. Real but below
  the +10% experiment threshold: with `-ncmoe 2` only two layers' experts (~1.5 GB) are host, so
  removing their recurring upload is a small part of the 28k prefill. The recipe enables it.
- `LLAMA_MOE_MAP_DEVICE`: +0.4% prefill, **+5.2% short decode**, +0.7% at 28k. Marginal; left off
  in the recipes until a decode-focused workload needs it.
- `LLAMA_MTP_RESERVE_WEIGHTS`: **+8.6% decode at 28k** (53.14 → 57.70) and **+7.9% short**. The
  reserve makes the auto cache account for the sidecar; the MTP recipe enables it.
- `LLAMA_MTP_KV_STORE_ONLY`: +1% (noise). Left off.
- `LLAMA_MOE_UPLOAD_OVERLAP`/`LLAMA_MOE_PINNED_RING`: no measurable effect at cache 1024. Off.
- `LLAMA_QSA_INDEXER_BOUNDS` and `LLAMA_GDN_CONV_L2`: no measurable effect on the Flash-Next
  prefill/decode. Off.
- MTP overall vs no-spec: decode **+31%** at 28k (57.70 vs 44.11), **+37%** at 4k (64.77 vs 47.14),
  short +5.6%; prefill **-17%** (1,292 vs 1,567). VRAM 39,052 MiB ready / 39,740 peak.

The 28k prefill gap to Strata (4,629 vs 1,567) is **not** explained by the H2D of the two host
layers or by these flags. The next step is the nsys/scheduler attribution (`GGML_SCHED_SPLIT_TIMING`,
`LLAMA_GRAPH_PREP_PROFILE`), not more default-off flags. Recipes now use the `parity5` zip; the
base recipe enables `LLAMA_MOE_CACHE_PREFILL=1` and the MTP recipe `LLAMA_MTP_RESERVE_WEIGHTS=1`.

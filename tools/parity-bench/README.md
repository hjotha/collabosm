# parity-bench

Harness reprodutível para as fases 0/1 do plano de paridade com a Strata na A100-40G.
Substitui os scripts ad-hoc `/tmp/opencode/vm_test_parity.py` e
`/tmp/opencode/vm_bench_parity.py` (que usavam caminhos fixos, `pkill` global e
`SIGKILL`, perdendo as linhas de hit/miss do MoE cache).

Python 3 stdlib apenas. Nada aqui compila, baixa modelo ou toca na GPU por conta própria.
Este harness vive em `collabosm/tools/parity-bench/`; os comandos abaixo partem da raiz do
repositório (`cd tools/parity-bench`), e os caminhos `/content/...` são da VM de teste, não do
computador que roda o app.

## Arquivos

| arquivo | função |
|---|---|
| `bench.py` | sobe o `llama-server`, mede os 5 casos fixos, VRAM, load e estatísticas do MoE cache; emite JSON + tabela Markdown |
| `ablate.py` | gera as variações de Fase 0 sobre uma config base e emite JSON comparativo |
| `examples/moe_cache_sample.log` | log de exemplo com as linhas reais de `llama_moe_cache` (usado no `--self-test`) |
| `examples/moe_cache_disabled_sample.log` | variante "cache auto desligado por orçamento" |
| `tests/fake_server.py` | `llama-server` falso (só `/health` e `/v1/chat/completions`) para testar o harness sem GPU |
| `tests/smoke_cpu.py` | teste de integração CPU-only: sucesso, falha de load, crash no meio dos casos e ablação |

## Protocolo

Cada **repetição** sobe um `llama-server` novo (load medido por repetição) e executa,
nesta ordem:

1. amostra a VRAM do device (baseline) via `nvidia-smi`;
2. `Popen` do servidor com
   `LD_LIBRARY_PATH=<bin-dir>:/usr/local/cuda/lib64:/usr/lib64-nvidia`
   (o valor é **substituído**, não herdado; `env` da config é aplicado depois);
3. espera `GET /health` responder **200** (estado de loading retorna 503) até
   `--load-timeout` (default 360 s); falha de load é registrada e o harness segue;
4. o amostrador de pico já está rodando desde o `Popen` a **1 Hz** (captura o pico de
   load); depois do health mede a VRAM em repouso;
5. envia `--warmup-requests` requisições curtas descartadas (default **1**) para
   capturar CUDA graphs antes das medições — `0` reproduz o protocolo antigo;
6. envia os casos fixos, na ordem `short`, `pp4k`, `pp4k+dec`, `pp28k`, `pp28k+dec`;
7. para o amostrador e encerra o servidor com **SIGTERM** (timeout
   `--shutdown-timeout`, default 60 s; só depois SIGKILL). O SIGTERM é obrigatório:
   as linhas `llama_moe_cache: ... hits/misses/...` são emitidas no destrutor do
   cache (`src/llama-moe-cache.cpp:476`), ou seja, **no shutdown**;
8. lê o log, faz o parse das linhas de MoE cache e grava tudo no JSON.

São **2 repetições** por config por default (`--runs`). A tabela usa a mediana;
`load_s` traz mediana/min/max e a VRAM de pico é o máximo das repetições.

### Casos fixos

| caso | prompt | `max_tokens` | prompt medido |
|---|---|---|---|
| `short` | "Write a python function that reverses a string, briefly." | 128 | ~50 tokens |
| `pp4k` | 100 blocos de código + "Summarize..." | 16 | 4.662 tokens |
| `pp4k+dec` | mesmo bloco + " Explain the time complexity." | 128 | 4.662 + ~7 |
| `pp28k` | 620 blocos de código + "Summarize..." | 16 | ~28,6k tokens |
| `pp28k+dec` | mesmo bloco + " Explain the time complexity." | 128 | ~28,6k + ~7 |

Os blocos foram calibrados para o tokenizer do Qwen3.8-Flash (100 → 4.662 tokens;
620 → ~28,6k) nos scripts antigos. O valor autoritativo é `prompt_n` no JSON: se o
filler ou o tokenizer mudar, ajuste `CODE_BLOCKS_4K`/`CODE_BLOCKS_28K` em `bench.py`.

Payload de cada requisição (OpenAI chat completions):

```json
{"model": "q", "messages": [{"role": "user", "content": "<prompt>"}],
 "max_tokens": 128, "temperature": 0, "stream": false, "cache_prompt": false,
 "chat_template_kwargs": {"enable_thinking": false}}
```

- `max_tokens` é alias de `n_predict` no schema do servidor
  (`tools/server/server-schema.cpp:64`).
- `cache_prompt` é campo do schema (`server-schema.cpp:51`), então vale no endpoint
  OpenAI deste fork.
- `enable_thinking` **não** é campo de topo do schema; é lido de
  `chat_template_kwargs` (`tools/server/server-common.cpp:1779-1796`). Por isso vai
  aninhado — é a forma suportada de desligar o thinking.

### VRAM

`vram_ready_mib` e `vram_peak_mib` são a memória **usada do device** (device-wide),
não o RSS do processo. O harness também grava `vram_baseline_mib` e os deltas
`vram_ready_delta_mib` / `vram_peak_delta_mib` (usado − baseline). Rode na VM
dedicada; qualquer outro processo na GPU contamina os absolutos (os deltas continuam
comparáveis).

### Parse do log

Formatos reais (ver `src/llama-moe-cache.cpp:602-616` e `src/llama-context.cpp:2530`):

```
llama_moe_cache: ubatch <= 8: hits = 372, misses = 169, hit rate = 68.76%, uploaded = 120.50 MiB
llama_moe_cache: ubatch  > 8: hits = 96, misses = 512, hit rate = 15.79%, uploaded = 3400.00 MiB
llama_moe_cache: large batches: 96 experts copied from the cache, 3400.00 MiB
llama_context: MoE cache auto: 6144.00 MiB for 17408.25 MiB of host experts (35.3% of the experts can be cached)
llama_context: MoE cache auto: 17408.25 MiB of host experts, but only 100.00 MiB of device memory is free after the compute buffers (reserve 1536.00 MiB), cache disabled
llama_context: MoE cache auto: no host-resident MoE experts, cache disabled
llama_moe_cache: CUDA0 MoE cache size =  6144.00 MiB for 17408.25 MiB of host experts
llama_context: MoE cache plan: CUDA0 = 6144.00 MiB
```

O JSON guarda as ocorrências (`stats`, `auto`, `sizes`, `plans`, `large_batch_copy`)
e um `summary` agregado. Se uma linha não existe (bucket sem tráfego, cache desligado),
o campo fica `null` — nada é inferido.

### Heartbeat

Uma thread imprime `[heartbeat] Ns elapsed, phase: ...` a cada **15 s**. A VM aborta
execuções sem saída por ~30 s; não remova.

## Flags do fork (verificadas no código)

Conferidas em `common/arg.cpp` (leitura apenas) nesta branch:

| flag | semântica | onde |
|---|---|---|
| `--moe-cache-mib N` | cache de experts em MiB; `-1` = auto (dimensiona após o `sched_reserve`) | `arg.cpp:4685` |
| `--vram-reserve-mib N` | VRAM mantida livre por device **quando** `--moe-cache-mib -1`; default 1536 | `arg.cpp:4701` |
| `--n-cpu-moe N` / `-ncmoe` | mantém os experts das N primeiras camadas na CPU | `arg.cpp:4675` |
| `-lzm/--lazy-mode on\|off\|auto` | leitura on-demand de tensores (exige mmap) | `arg.cpp:4618` |
| `-lm/--load-mode auto\|none\|mmap\|mlock\|mmap+mlock\|dio` | modo de load | `arg.cpp:4599` |
| `-fa/--flash-attn on\|off\|auto` | flash attention | `arg.cpp:3218` |
| `--jinja` | template Jinja | `arg.cpp:5910` |
| `--no-warmup` | não aquece no load | `arg.cpp:3432` |
| `-ctk/-ctv q8_0` | tipo do KV cache | args upstream |
| `-b/-ub`, `-c`, `-ngl`, `--alias`, `--host/--port` | batch/ubatch, contexto, offload, alias, bind | args upstream |

`docs/beellama-args.md` (v0.5.0) documenta KVarN/SINQ e flags Bee, mas **não** lista
`--moe-cache-mib`/`--vram-reserve-mib`; a fonte de verdade dessas flags é o
`common/arg.cpp`. Flags gerenciadas pelo harness (`-m`, `--port`, `-b`, `-ub`, `-c`,
`-ctk`, `-ctv`, `--moe-cache-mib`, ...) são **rejeitadas** dentro de `extra_args` para
não quebrar o A/B silenciosamente.

## Rodando na A100 com o zip `beellama/bin`

O zip entregue tem o layout `bin/llama-server` + libs ao lado. Extraia um diretório
por build:

```bash
unzip -q beellama-bin.zip -d /content/pb-new        # -> /content/pb-new/bin/llama-server
# se o zip preservar a pasta do repo, o caminho vira /content/pb-new/beellama/bin
ls /content/pb-new/bin/llama-server /content/pb-new/beellama/bin/llama-server 2>/dev/null
chmod +x /content/pb-new/bin/llama-server
ldd /content/pb-new/bin/llama-server | grep 'not found' || echo "libs ok"
```

`--model` é o **shard 1**; o loader resolve o shard 2 no mesmo diretório
(`...-00001-of-00002.gguf` → `...-00002-of-00002.gguf`). Mantenha os dois.

### Configs em JSON (recomendado: compara build nova × baseline)

`configs.json`:

```json
[
  {"name": "base-f16", "bin_dir": "/content/pb-base/bin", "kv": "f16",
   "moe_cache_mib": 6144, "batch": 2048, "ubatch": 512},
  {"name": "new-f16", "bin_dir": "/content/pb-new/bin", "kv": "f16",
   "moe_cache_mib": 6144, "batch": 2048, "ubatch": 512},
  {"name": "new-q8-auto", "bin_dir": "/content/pb-new/bin", "kv": "q8_0",
   "moe_cache_mib": -1, "vram_reserve_mib": 1536, "batch": 2048, "ubatch": 512},
  {"name": "new-f16-lazyoff", "bin_dir": "/content/pb-new/bin", "kv": "f16",
   "moe_cache_mib": 6144, "lazy_mode": "off"}
]
```

Campos: `name`, `bin_dir` (opcional; sobrepõe `--bin-dir`), `kv` (`f16|q8_0`),
`moe_cache_mib`, `batch`, `ubatch`, `ctx`, `ngl`, `fa`, `load_mode`, `lazy_mode`
(`on|off|auto`), `n_cpu_moe`, `vram_reserve_mib`, `runs`, `port`, `extra_args`
(lista), `env` (objeto). Defaults: `ctx=32768`, `ngl=99`, `fa=on`, `load_mode=mmap`,
`lazy_mode=on`, `n_cpu_moe=2`, `moe_cache_mib=6144`, `b=2048`, `ub=512`,
`extra_args=["--jinja","--no-warmup"]` (os extras da config são somados a esses).

Comando (Fase 0 + Fase 1 no mesmo JSON):

```bash
cd tools/parity-bench  # a partir da raiz do collabosm
python3 bench.py \
  --bin-dir /content/pb-new/bin \
  --model /content/fn40/real/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --configs /content/parity/configs.json \
  --out /content/parity/parity.json \
  --reference base-f16 --candidate new-f16 \
  --gpu-index 0 --runs 2
```

O stdout traz as tabelas Markdown; o JSON final fica em `--out`. Logs brutos do
servidor vão para `<out-dir>/logs/<stem>/<config>_run<N>.log` (use `--log-dir` para
mudar). O JSON é reescrito após cada config, então um abort no meio preserva o parcial.

### Só uma config (CLI, sem JSON)

```bash
python3 bench.py --bin-dir /content/pb-new/bin --model /caminho/shard1.gguf \
  --out /content/parity/one.json \
  --config 'name=new-f16,kv=f16,cache=6144,b=2048,ub=512,extra=--metrics --no-moe-host-pinned'
```

`extra=` precisa ser o **último** campo (o resto da string vira argumentos via
`shlex.split`); o parser rejeita `extra` no meio. Chaves aceitas: `name`, `kv`,
`cache`/`moe_cache_mib`, `b`/`batch`, `ub`/`ubatch`, `ctx`, `ngl`, `fa`, `lazy`,
`load_mode`, `ncmoe`/`n_cpu_moe`, `reserve`/`vram_reserve_mib`, `build`/`bin_dir`,
`runs`, `port`. Também há `--config-json '<objeto>'` (repetível).

## Critérios de aceite

### Fase 0 — baseline

Mesma config, build nova × build baseline (`--reference base-f16 --candidate new-f16`,
flags idênticas). O bloco `comparison.acceptance` do JSON calcula:

- `phase0_baseline_within_5pct`: **PASS** se, em todos os casos, `pp` e `tg` ficarem
  dentro de **±5%** do baseline (mediana das repetições). Se `pp` do `short` for o
  único fora — prompt minúsculo, ruído alto — registre isso explicitamente no laudo
  em vez de esconder.
- Se as flags do candidato e da referência forem diferentes, esse check retorna
  `null` (N/A) com `same_flags: false`; é o caso das ablações de kv/lazy/ub.

**80% do prefill atribuído**: isso é atribuição de perfil, não uma métrica que o
bench mede. Procedimento reprodutível:

```bash
nsys profile -t cuda,nvtx -o /content/parity/pp4k --force-overwrite true \
  python3 bench.py --bin-dir /content/pb-new/bin --model /caminho/shard1.gguf \
  --out /content/parity/pp4k.json --cases pp4k --runs 1 --warmup-requests 0 \
  --config 'name=pp4k,kv=f16,cache=6144,b=2048,ub=512'
nsys stats --report cuda_gpu_kern_sum /content/parity/pp4k.nsys-rep
```

Aceite: ≥80% do tempo de GPU do prefill de 4.662 tokens atribuído aos kernels do
caminho esperado (MMQ/MMVQ, atenção/MoE do fork) e não a fallbacks/`dequantize`/cópias
extras. O `bench.py` garante o mesmo prompt, batch e flags para o perfil.

### Fase 1 — ganho

Com a build nova e as **mesmas flags** do baseline (`--reference`/`--candidate`):

| check no JSON | alvo |
|---|---|
| `phase1_prefill_long_gain_ge_25pct` | `pp28k` ≥ **+25%** vs baseline |
| `phase1_decode_long_ge_55tps` | `tg` do `pp28k+dec` ≥ **55 t/s** |
| `phase1_short_regression_le_3pct` | `tg` do `short` ≥ **−3%** vs baseline |

`acceptance.pass` é `true` só se todos os checks avaliados passarem. Checks de casos que
não foram medidos (por `--cases`) retornam `null` (N/A), não FAIL. O JSON traz
`pp_delta_pct`/`tg_delta_pct` por caso e por config, além dos valores absolutos.

## Ablações (Fase 0)

`ablate.py` gera, nesta ordem determinística:

```
base, kv-f16, kv-q8_0, lazy-on, lazy-off, ub-512, ub-1024,
ncmoe-1, ncmoe-2, ncmoe-3, reserve-1536, reserve-2048
```

Cada variante herda todos os knobs da base e muda **um** eixo. Variantes que resolvem
para exatamente a mesma linha de comando da base (ex.: `kv-f16` com base `kv=f16`)
viram `alias_of: "base"` e **não são reexecutadas** — economiza ~3 min por alias e o
JSON diz explicitamente que os números foram reusados.

```bash
python3 ablate.py \
  --bin-dir /content/pb-new/bin \
  --model /content/fn40/real/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --out /content/parity/ablate.json \
  --moe-cache-mib -1 --vram-reserve-mib 1536 \
  --runs 2 --gpu-index 0
```

Saída: `<out>` (comparativo, schema `parity-bench-ablate/1`) + `<out>.bench.json`
(bruto do `bench.py`, mesmo schema do bench). A tabela de deltas cobre `pp4k`,
`pp28k`, `short tg`, decode pós-28k, load, VRAM de pico e hit rate do MoE.

**Atenção**: `--vram-reserve-mib` só tem efeito com `--moe-cache-mib -1`. Com cache
fixo, `reserve-1536/2048` são inertes; o JSON emite warning e os resultados devem
sair iguais (é um bom teste de ruído do harness).

Os alvos de Fase 1 no bloco `ablation.variants[].targets_vs_base` são **informativos**
nesse contexto: as ablações mudam flags, então elas não são o A/B de build da Fase 1.
A regra de ±5% da Fase 0 também só vale para A/B com flags idênticas.

## Saída JSON (resumo)

```json
{
  "schema": "parity-bench/1",
  "host": {"gpus": {...}},
  "protocol": {"model": "...", "cases": [...], "reference": "...", "candidate": "..."},
  "configs": [
    {"name": "new-f16", "config": {...}, "alias_of": null,
     "runs": [
       {"run": 1, "ok": true, "complete": true, "load_s": 63.2,
        "vram_ready_mib": 36400, "vram_peak_mib": 38900,
        "cases": {"short": {"prompt_n": 51, "cache_n": 0, "pp_tps": 812.4,
                            "tg_tps": 61.2, "text_sha256": "...", "text": "..."}},
        "moe_cache": {"summary": {"small_hit_rate_pct": 68.76, "large_hits": 96}},
        "server_exit": {"graceful": true, "exit_code": 0}}
     ],
     "aggregate": {"cases": {"short": {"pp_tps_median": 812.4,
                                       "tg_tps_median": 61.2}},
                   "moe": {...}, "load_s": {...}}}
  ],
  "determinism": {"per_case": {"short": {"groups": {"<sha12>": ["new-f16"]}}}},
  "comparison": {"cases": {...}, "acceptance": {"checks": [...], "pass": true}}
}
```

- `text` é o texto completo; `determinism` agrupa por `sha256` e sinaliza
  `run_to_run_identical: false` quando as repetições divergem.
- `aggregate.moe.uploaded_mib_per_run` é a mediana por repetição (use esta para
  comparar braços); `uploaded_mib_total` é a soma das repetições.
- `comparison.acceptance.checks[].pass` pode ser `null` (N/A), nunca "verdadeiro por
  omissão".
- `--gpu-index N` seleciona o device no `nvidia-smi` **e** exporta
  `CUDA_VISIBLE_DEVICES=N` para o servidor, então o índice é o físico.

## Testes sem GPU

```bash
cd tools/parity-bench  # a partir da raiz do collabosm
python3 -m py_compile bench.py ablate.py tests/fake_server.py tests/smoke_cpu.py
python3 bench.py --self-test          # parse do log, CLI, validação, agregação, markdown
python3 ablate.py --self-test         # nomes determinísticos, aliases, render
python3 tests/smoke_cpu.py            # integração real do ciclo de vida (server falso)
python3 bench.py --dry-run --bin-dir /content/pb-new/bin --model /caminho/shard1.gguf \
  --out /tmp/dry.json --config 'name=new-f16,kv=q8_0,cache=-1,ub=1024'
```

`tests/smoke_cpu.py` cobre: load OK, load que falha (registra e segue), crash no meio
dos casos (casos restantes viram erro, harness não cai), warmup, contagem de
requisições, parse do log no SIGTERM, determinismo, `comparison` e `ablate` completo.
Ele usa `tests/fake_server.py` como `llama-server`; nenhum binário real é executado.

## Cuidados

- **Não use `pkill llama-server`**: o harness mata só o PID/grupo que subiu. Se uma
  porta estiver ocupada, ele anda para a próxima (`--port-base`, default 8090).
- Tempo estimado por repetição na A100: load 40–90 s + casos ~80–120 s (depende do
  hit rate). 3 configs × 2 runs ≈ 15–25 min. Para laudo final prefira `--runs 3`
  (com 2, a "mediana" é a média dos dois valores).
- `--model` não é baixado; aponte para o shard 1 já presente na VM.
- `nvidia-smi` ausente não derruba nada: os campos de VRAM ficam `null`.
- O harness **não** mexe no repositório `beellama.cpp` nem compila nada.

# Fase 0 — instrumentação do prefill de ~28,6k tokens (Qwen3.8-Flash-Next, A100-40G)

Objetivo: atribuir **≥80%** do custo do prefill entre GDN, QSA/atenção, MoE MMQ,
cópias H2D/D2H e host idle, e decidir com número — não com intuição — se as fases 3
(pipeline do PLE) e 4 (overlap dos experts) do `docs/PLAN-strata-parity.md` têm
evidência neste modelo/configuração.

Nada aqui compila, roda CUDA ou toca no `llama-server` local. Os scripts foram
validados com `bash -n`, `py_compile`, `--dry-run` e um trace SQLite sintético;
a execução real é sua, na A100, quando o benchmark de lá terminar.

## Arquivos

| arquivo | papel |
|---|---|
| `profile_prefill.sh` | orquestra os runs `clean` e `trace` (nsys), prompt, `/proc`, resumo e aceitação |
| `ablate_lazy.py` | A/B `--lazy-mode on` vs `off` (cold/warm) e driver HTTP do prefill |
| `summarize_nsys.py` | lê o SQLite do nsys e produz `class_summary.csv/.md` por classe |
| `README.md` | este protocolo |

Copie os **quatro juntos** para a A100 (o shell procura os `.py` no próprio diretório).

## Comando exato para rodar depois

Na A100, com o benchmark de lá encerrado e a GPU livre:

```bash
cd /content/profiling   # diretório onde os 4 arquivos foram copiados
./profile_prefill.sh \
  --server /content/llama-parity/beellama/bin/llama-server \
  --model  /content/fn40/real/Q2_0 \
  --outdir /content/profiling/phase0-a100 \
  --moe-cache-mib -1 \
  --runs both
```

> **`--moe-cache-mib`:** o pedido original dizia `6144`, mas com `-ncmoe 2` e
> `-ub 1024` isso pede ~44,8 GiB (34,5 GiB de pesos + 6 GiB de cache + 4 GiB de
> KV/buffers) e **não cabe nos 40 GiB**. O `measured_q20.md` usou `-1` (auto),
> que dimensiona o cache depois do `sched_reserve` e cabe (~39,0 GiB). O script
> aborta no preflight se a conta não fechar. A alternativa é
> `--ncmoe 8 --moe-cache-mib 6144` (mais experts no host, menos pesos na GPU).
> O default do script continua 6144 para reproduzir o pedido literal; na A100
> use `-1`.

Isso faz dois runs com o mesmo prompt calibrado (28.600 ± 0,6%, ~±170 tokens;
o número exato fica em `prompt_meta.json`/`client.json`):

1. `clean/` — sem nsys e sem envs de diagnóstico: é o **número de aceitação**.
2. `trace/` — `nsys profile` + instrumentação env-gated: é a **atribuição**.

Cada run sobe um servidor novo, espera `/health`, faz **uma** requisição
`/completion` com `cache_prompt=false`, `n_predict=1`, e derruba o servidor com
SIGTERM (para imprimir as estatísticas do MoE cache e fechar o `.nsys-rep`).

Se a GPU estiver ocupada por outro processo, o script **aborta**. Use
`--allow-busy-gpu` só de propósito. Para validar sem GPU/modelo:

```bash
./profile_prefill.sh --dry-run
```

A/B do PLE (rodar depois do profile, ou no lugar dele):

```bash
python3 ablate_lazy.py ablate \
  --server /content/llama-parity/beellama/bin/llama-server \
  --model  /content/fn40/real/Q2_0 \
  --outdir /content/profiling/lazy-ab \
  --moe-cache-mib -1 \
  --reps 2 --drop-caches auto
```

`--drop-caches auto` usa `echo 3 > /proc/sys/vm/drop_caches` quando roda como root.
Sem root, o braço "cold" pode estar quente e o resultado não vale.

## Flags e envs (nomes conferidos no código)

Flags de produção usadas pelos dois scripts:

```
-ngl 99 -ncmoe 2 --moe-cache-mib 6144 -c 32768 -b 2048 -ub 1024 -fa on \
--load-mode mmap --lazy-mode on -ctk q8_0 -ctv q8_0
```

`--moe-cache-mib` é parâmetro (`--moe-cache-mib -1` = auto). O default 6144 é o
do pedido; na A100-40G use `-1` (veja o box acima).

(`common/arg.cpp`: `-ngl` L4732, `-ncmoe` L4675, `--moe-cache-mib` L4685,
`--load-mode` L4599, `--lazy-mode` L4618, `-ctk/-ctv` L3909/L3922.)

Envs do run `trace` (todas existem no código):

| env | onde | o que imprime |
|---|---|---|
| `GGML_SCHED_SPLIT_TIMING=1` | `ggml/src/ggml-backend.cpp:2645` | `[SPLIT] backend=… nodes=… wall_ms=…`, com um `ggml_backend_synchronize` por split |
| `GGML_SCHED_SPLIT_SPAN=1` | `ggml/src/ggml-backend.cpp:2556` | `[SPLITSPAN] … t_iter_us t_postsync_us t_postcopy_us` (sem sync) |
| `LLAMA_GRAPH_PREP_PROFILE=1` | `src/llama-context.cpp:4461` | `[GRAPHPREP] … apply/check/reset/build/alloc/inputs_ms` por ubatch |
| `LLAMA_F5_STAGE_TIMING=1` | `src/llama-context.cpp:4634` | `[F5STAGE] n_tokens=… graph_compute_host_ms=…` (submissão, assíncrona) |
| `GGML_CUDA_GRAPH_DEBUG=1` | `ggml/src/ggml-cuda/ggml-cuda.cu:3747` | eventos de captura/update de CUDA graph + VRAM livre |

`GGML_SCHED_HASH_STAGED=1` existe (`ggml-backend.cpp:2560`) mas **não é usado**:
para tensores ≤ 1 MiB ele faz `ggml_backend_tensor_get` síncrono no meio do
split (`f5_hash_staged_print`, L2408), o que serializa a GPU e contamina o
tempo. Fica só como ferramenta de correção, não de medição.

## Fatos do modelo (lidos do GGUF, não estimados)

| item | valor |
|---|---|
| layers | 48 (36 GDN + 12 QSA; `compress_ratios` > 0 nas layers 3,7,…,47) |
| experts | 512 por layer, 10 usados por token |
| tipo dos experts | `Q2_0` = id **42** (confirmado em `ffn_{up,gate,down}_exps` das 48 layers) |
| experts em host com `-ncmoe 2` | 2 × 707,8 MB = **1,42 GB** |
| `per_layer_token_embd` (PLE) | **28,80 GB**, tipo 20 (IQ4_NL), no shard 2 |
| linhas de PLE por token | `n_embd / n_embd_per_layer = 2560/160 = 16` |
| cache MoE | 6144 MiB cobre os 1,42 GB de experts host |

Duas consequências que a medição precisa confirmar ou refutar:

1. **Prefill não usa o cache de experts.** `max_batch = 32`
   (`src/llama-moe-cache.cpp:231`) e `get_slot_map` devolve `nullptr` quando
   `n_tokens > 32` (L486). Com `-ub 1024`, cada ubatch reenvia os 1,42 GB de
   experts das layers 0–1 pela PCIe. Em 28 ubatches (28.600/1024), isso é
   **~39,7 GB de H2D**. Se o trace mostrar ~40 GB de HtoD, o classificador e o
   modelo estão coerentes; se mostrar ~1,4 GB, o cache está sendo usado no
   prefill e essa leitura está errada — nesse caso, pare e investigue.
2. **`--lazy-mode on` não lê o PLE no load.** Com lazy ranges, o mmap não usa
   `MAP_POPULATE` e faz `MADV_WILLNEED` só no complemento
   (`src/llama-mmap.cpp:512-537`). Com `off`, o shard 2 inteiro (28,8 GB) é
   populado no load. É por isso que o A/B compara `load_s` **separado** de
   `prompt_ms`.

## Contexto medido (para sanidade, não para aceitação)

Do `measured_q20.md`/`measured_compare.md`, mesma VM e mesmos pesos:

| medição | valor |
|---|---:|
| prefill 28,6k, `-ub 512`, `--moe-cache-mib -1` | 1.175,6 t/s → **24,3 s** |
| prefill 4,6k, mesmo config | 1.157,8 t/s |
| prefill 4k, `-ub 1024` (tabela de placement) | 1.482,7 t/s |
| Strata, mesmo prompt de 28,6k | 4.629,3 t/s → 6,2 s |

Se o seu `clean` com `-ub 1024` der muito abaixo de ~1.300 t/s, alguma flag
mudou (cache, `-ncmoe`, KV, página fria) e o trace não é comparável ao
baseline acima. O alvo da Fase 0 é explicar os ~24 s (ou o que o `clean` der),
não reproduzir a Strata.

## Protocolo

### Run `clean` (aceitação)

- servidor novo, página do GGUF e do PLE **frias** (`drop_caches` quando root);
- sem nsys, sem envs de diagnóstico;
- uma requisição de 28,6k tokens com `cache_prompt=false`;
- registra `timings.prompt_ms`, `prompt_n`, `cache_n`, wall do cliente,
  `read_bytes`/`majflt` do processo e `nvidia-smi` antes/depois.

### Run `trace` (atribuição)

- mesmas condições, com `nsys profile --trace=cuda,osrt --sample=none
  --cpuctxsw=process-tree` e as envs da tabela acima;
- o `.nsys-rep` é exportado para SQLite e o `summarize_nsys.py` gera
  `class_summary.csv` e `class_summary.md`;
- a janela do request é delimitada por marcadores de epoch do cliente
  (`epoch_start_ns`/`epoch_end_ns`), convertidos com
  `TARGET_INFO_SESSION_START_TIME.utcEpochNs`. Se a janela não tiver kernel
  nenhum, o script cai para a última rajada do trace e avisa.

## Como ler

### `GGML_SCHED_SPLIT_TIMING` / `SPLITSPAN`

Um grafo de prefill tem normalmente dois splits: `CPU` (gather do PLE e
entradas) e `CUDA0` (o resto). Com `SPLIT_TIMING`:

```
[SPLIT] backend=CUDA0 nodes=900 wall_ms=45.000
```

é o wall do split **incluindo** o `ggml_backend_synchronize` que a instrumentação
adiciona. Serve para saber quanto tempo o split levou, mas **serializa** os
ubatches; o total do run `trace` pode ficar acima do `clean`.

Com `SPLITSPAN` (sem sync), por split:

- `wait_us = t_postsync_us - t_iter_us` → espera pelo evento do backend anterior;
- `copy_us = t_postcopy_us - t_postsync_us` → cópias de entrada no host;
- `submit_us = t_iter_us(next split) - t_postcopy_us` → submissão (CUDA) ou
  computação (CPU) do split atual.

Some por backend no `class_summary.md` (seção "Splits"). Um `CPU` com
`submit_us` alto é o custo do PLE no host — a evidência direta da Fase 3.

### `GRAPHPREP` / `F5STAGE`

`[GRAPHPREP]` cobre `apply`, `check`, `reset`, `build`, `alloc` e `inputs` do
grafo, por ubatch. `[F5STAGE] graph_compute_host_ms` é a **submissão** assíncrona
(`llama_context::graph_compute` retorna antes da GPU terminar). Sem
`SPLIT_TIMING` é só submissão; com `SPLIT_TIMING` ligado ele inclui a espera dos
`ggml_backend_synchronize` por split. Não é o tempo de GPU. Somados, dão o
custo de host por ubatch.

### nsys

- `class_summary.md`: GDN, QSA, MoE, H2D, D2H, host idle, outro GPU, com ms,
  % do wall e instâncias.
- `kernels_top.csv`: os kernels que dominam; confira se algum nome caiu em
  "outro GPU" por engano.
- `memcpy_by_kind.csv`: HtoD/DtoH por tempo e bytes.
- `summary.json`: tudo, inclusive `gpu_busy_ms`, `host_busy_ms`,
  `five_class_pct` e o veredito de cobertura.
- `nsys-stats/`: relatórios padrão do nsys (`cuda_gpu_kern_sum`,
  `cuda_gpu_mem_time_sum`, `cuda_api_sum`) para conferência manual.

Classificação dos kernels (ver `summarize_nsys.py`):

| classe | padrão |
|---|---|
| GDN | `gated_delta_net`, `ssm_conv`, `ssm_scan`, `ssm_ssd`, `l2_norm_f32` |
| QSA/atenção | `flash_attn`, `fattn`, `lightning_indexer`, `top_k_mask`, `k_set_rows`, `cumsum_kernel` |
| MoE/FFN | `mul_mat_q` com `(ggml_type)42`, `topk_moe`, `moe_weighted_reduction`, `mm_ids_helper`, `add_id_kernel`, `mul_mat_vec_q_moe` |
| outro GPU | todo o resto (projeções densas, norms, rope, GLU) |

O id 42 vem do GGUF (`model-info`), não é hardcoded. Os 8 tensores Q2_0 que não
são experts (`attn_q`, `ple_key`, `attn_gate`) somam 50 MB contra 34 GB de
experts — 0,15%, desprezível, mas está documentado.

SQL manual, se precisar:

```sql
-- kernels por nome (demangledName é id em StringIds)
SELECT s.value, count(*), sum(k.end-k.start)/1e6 AS ms
FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName
GROUP BY 1 ORDER BY 3 DESC LIMIT 20;

-- memcpy por tipo
SELECT e.label, count(*), sum(m.bytes)/1048576.0 AS MiB, sum(m.end-m.start)/1e6 AS ms
FROM CUPTI_ACTIVITY_KIND_MEMCPY m JOIN ENUM_CUDA_MEMCPY_OPER e ON e.id=m.copyKind
GROUP BY 1;
```

Os timestamps do SQLite são **ns desde o início da coleta**; o epoch do começo
está em `TARGET_INFO_SESSION_START_TIME.utcEpochNs`.

## Critério de aceite da Fase 0

1. **Baseline ±5%**: `prompt_ms` do run `trace` dentro de ±5% do `clean`.
   O `acceptance.md` calcula e marca PASS/FAIL. Se falhar, o `nsys`/os envs
   mudaram o comportamento; use o `clean` como denominador do wall e trate as
   frações do trace como relativas (os kernels continuam corretos; os gaps do
   host podem estar inflados pelo sync por split).
2. **80% atribuído**: `five_class_pct ≥ 80%` em `summary.json` /
   `class_summary.md`, somando GDN + QSA + MoE + H2D + D2H + host idle.
3. **Sanidade dos bytes**: HtoD ≈ 39,7 GB (28 ubatches × 1,42 GB), conforme a
   previsão acima. Divergência grande invalida a leitura do MoE.
4. **Sem host idle negativo** e sem `other_gpu` absurdo (> 40%): se acontecer,
   o classificador perdeu kernels e o resultado não vale.

Só com 1–3 passando o número serve para decidir as fases seguintes.

## A/B do PLE (`ablate_lazy.py`)

Para cada modo (`on`, `off`), em ordem alternada entre repetições:

1. `drop_caches` (cold);
2. servidor novo (KV vazio);
3. prefill #1 = **cold** (páginas do GGUF/PLE fora do page cache);
4. prefill #2 = **warm** (mesmas páginas, já quentes; KV é reprocessado porque
   `cache_prompt=false`, e o script aborta se `cache_n != 0`);
5. `off` paga 28,8 GB no load; `on` paga as linhas sob demanda no prefill —
   por isso `load_s` e `prompt_ms` são reportados separadamente.

Decisão (mediana de `prompt_ms`, `--reps 2`):

| condição | veredito |
|---|---|
| `off` > 20% mais rápido que `on` no **cold** | `EVIDENCIA_FASE3` — o I/O do PLE pesa; pipeline tem alvo |
| diferença < 5% no cold **e** no warm | `SEM_EVIDENCIA` — não construir a Fase 3 por isso |
| entre 5% e 20% | `INCONCLUSIVO` — repetir/alongar o prompt antes de decidir |

O veredito está em `lazy-ab/summary.md` e `run.json`. O ganho de páginas quentes
dentro do modo `on` (`warm_gain_on`) separa “é I/O de página” de “é CPU/latência
de get_rows”.

## Critério da Fase 4 (overlap de experts)

O teto do ganho do overlap é o tempo de H2D **serializado** no mesmo stream:

- `copy_h2d_pct = ms de H2D / wall` no `class_summary.md`;
- `>= 10%` → há evidência para a Fase 4 (o teto justifica o trabalho);
- `< 5%` → não há evidência; o ganho máximo é menor que o ruído entre runs;
- entre 5% e 10% → meça o `streamId` das cópias e dos kernels no SQLite: se já
  estiverem em streams diferentes, o overlap existe e o teto real é menor.

Para este modelo, a previsão é ~40 GB de H2D (28 ubatches × 1,42 GB) porque o
cache de experts é bypassado no prefill (`n_tokens > 32`). **Antes de partir
para a Fase 4**, teste a hipótese mais barata: com `--moe-cache-mib 6144` e só
1,42 GB de experts host, o cache caberia todos eles; se o bypass for a causa
dos 40 GB, permitir o cache em lotes grandes (ou subir `max_batch`) elimina o
H2D sem mexer no motor. O trace mostra se o H2D é estrutural ou não.

## Critério da Fase 3 (pipeline do PLE)

`host_idle` inclui as page faults do PLE; para separar o que é I/O do que é
CPU, use:

- `[SPLIT] backend=CPU wall_ms` (com `SPLIT_TIMING`): custo do gather do PLE
  por ubatch, incluindo as faltas de página;
- `read_bytes`/`majflt` do processo durante o request (`client.json`);
- o A/B `ablate_lazy.py` (cold vs warm), que é o teste decisivo.

O código já faz `llama_prefetch_rows` (`madvise(MADV_WILLNEED)`) para o ubatch
corrente (`src/models/qwen4exp.cpp:1436`); a Fase 3 seria prefetchar o **próximo**
ubatch enquanto o atual computa. Se o custo do split CPU for pequeno, não vale.

## O que NÃO concluir

- **Smoke truncado não mede performance.** O teste de header + `truncate` do
  `docs/FN40-CRASH-ROOTCAUSE.md` prova o caminho de load em 96 MB; ele não diz
  nada sobre prefill, H2D ou hit rate.
- **Hit rate isolado não decide nada.** O número de 68,76% do plano é de
  `ubatch <= 8` (decode). No prefill com `-ub 1024` o cache é **bypassado**
  (`max_batch=32`); um hit rate alto no fim do run não contradiz os ~40 GB de
  H2D do prefill. Olhe `hits/misses` **e** os bytes HtoD.
- **Soma de kernels ≠ wall.** Kernels e cópias podem sobrepor; o script usa a
  **união** dos intervalos (`gpu_busy_ms`), não a soma.
- **`off` ganhar no load não é evidência da Fase 3.** O critério é o prefill;
  o custo de startup é reportado à parte e pode inverter a decisão prática.
- **Bytes H2D não provam que a Fase 4 vale.** É preciso ver a GPU **idle**
  durante a cópia (mesmo stream). Se a cópia já estiver sobreposta, o teto do
  overlap é menor que o tempo de cópia.
- **Uma corrida só não é resultado.** Varie `--reps`, olhe a dispersão e
  registre clock/power (`nvidia-smi-*.txt`). A A100 muda de clock entre runs.
- **Não compare modelos/máquinas diferentes** com estes números: as frações
  dependem de quantização, `-ncmoe`, cache e disco.

## Se o nsys não for a ferramenta certa

Para a Fase 0 ele **é** a ferramenta certa: precisamos de tempo por kernel,
cópias e gaps de host no mesmo eixo, com overhead baixo. `ncu` não substitui:
ele faz replay kernel a kernel, tem overhead de 10–100×, não vê o wall do
prefill nem o idle entre kernels, e não atribui cópias H2D no contexto do
request. Use `ncu` só **depois**, para os 2–3 kernels que o nsys apontar como
dominantes (ex.: `--set full --kernel-name regex:mul_mat_q`).

Se `nsys` não existir ou não conseguir injetar no chroot da A100:

1. O script detecta e cai para **log-only** (`--nsys auto`), mantendo
   `[SPLIT]`, `[GRAPHPREP]`, `[F5STAGE]`, `/proc/<pid>/io`, `majflt` e
   `nvidia-smi`. A atribuição por classe fica limitada a CPU split vs CUDA
   split + bytes lidos, mas ainda responde à Fase 3.
2. Alternativa: rodar o `nsys` do host (fora do chroot) com
   `--trace=cuda,osrt` no binário do chroot; o CUPTI precisa do driver do host,
   que é o mesmo.
3. Última alternativa: `nsys` de outra máquina com A100 (o trace é portável),
   mantendo o mesmo build, modelo e flags.

`--nsys on` falha alto em vez de cair para log-only; `--nsys auto` (default) cai.

## Testes sem GPU

```bash
python3 tests/test_schema.py    # classificador + contadores num SQLite sintetico
```

Os outros dois arquivos em `tests/` (`make_fake_model.py`, `fake_server.py`) permitem
ensaiar o fluxo inteiro sem GPU; o `README` não depende deles.

## Limitações conhecidas

- `GGML_SCHED_SPLIT_TIMING=1` adiciona um sync por split. O default do script é
  `--split-timing auto` = ligado no trace (foi o pedido do plano), e o
  `acceptance.md` mede o desvio. Se o desvio passar de 5%, repita o trace com
  `--split-timing off` para a timeline limpa (perde só o `[SPLIT] wall_ms`).
- `host_idle` = wall − GPU busy − CPU do processo. CPU que roda em paralelo com
  a GPU conta como host CPU, não como idle; por isso o idle é conservador.
- `k_set_rows` também é usado para escrever KV; pode haver pequena
  misatribuição entre QSA e “outro GPU”. O `kernels_top.csv` mostra o peso.
- O prompt é código repetido calibrado por `/tokenize`; ele não replica a
  distribuição de tokens de um prompt real, só o **tamanho**. Para comparar
  com produção, troque `--prompt-file` por um prompt real de mesmo tamanho.
- `--lazy-mode off` exige RAM para 28,8 GB de PLE + margem; o script aborta se
  `MemAvailable` não bastar. Não rode o braço `off` com a A100 fazendo outra
  coisa.

## Saída esperada

```
phase0-a100/
  model_info.json          # layers, expert type, PLE, shards
  prompt.txt               # prompt calibrado (sha256 no client.json)
  prompt_meta.json
  acceptance.md            # clean vs trace, ±5%
  clean/
    server.log  client.json  drive.log  load_s.txt  nvidia-smi-*.txt
  trace/
    prefill.nsys-rep  prefill.sqlite
    server.log  client.json  drive.log
    class_summary.csv  class_summary.md  summary.json
    kernels_top.csv  memcpy_by_kind.csv  osrt_top.csv
    nsys-stats/
```

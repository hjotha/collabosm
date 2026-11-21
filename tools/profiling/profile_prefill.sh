#!/usr/bin/env bash
#
# Fase 0 do docs/PLAN-strata-parity.md: atribuir o custo do prefill de ~28.6k
# tokens do Qwen3.8-Flash-Next no A100-40G entre GDN, QSA/atencao, MoE MMQ,
# copias H2D/D2H e host idle.
#
# O script NAO compila nada e NAO roda benchmark: sobe o llama-server com as
# flags de producao, faz UMA requisicao de prefill sem cache e coleta:
#   - run `clean`: sem nsys e sem envs de diagnostico -> numero de aceitacao
#   - run `trace`: nsys + instrumentacao env-gated -> timeline e classes
#
# Uso rapido (na A100, quando o benchmark de la tiver terminado):
#   ./profile_prefill.sh --moe-cache-mib -1
#   (--allow-busy-gpu so se a GPU estiver ocupada de proposito)
#
# Validacao sem GPU:
#   ./profile_prefill.sh --dry-run
#
# Nomes de env/flag foram conferidos no codigo (common/arg.cpp, src/llama-context.cpp,
# ggml/src/ggml-backend.cpp, ggml/src/ggml-cuda/ggml-cuda.cu). Veja README.md.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_DRIVER="$SCRIPT_DIR/ablate_lazy.py"
PY_SUMMARY="$SCRIPT_DIR/summarize_nsys.py"

SERVER="${SERVER:-/content/llama-parity/beellama/bin/llama-server}"
MODEL="${MODEL:-/content/fn40/real/Q2_0}"
PORT="${PORT:-8097}"
TARGET_TOKENS="${TARGET_TOKENS:-28600}"
NCMOE="${NCMOE:-2}"
MOE_CACHE_MIB="${MOE_CACHE_MIB:-6144}"
GPU_TOTAL_MIB="${GPU_TOTAL_MIB:-0}"
OUTDIR=""
RUNS="both"
NSYS_MODE="auto"
SPLIT_TIMING="auto"
DROP_CACHES="auto"
ALLOW_BUSY_GPU=0
DRY_RUN=0
FORCE=0
SOURCE_DIR=""
CHECK_HELP=0
KEEP_SERVER=0
NO_STATS=0
LOAD_TIMEOUT="${LOAD_TIMEOUT:-3600}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-3600}"
SETTLE="${SETTLE:-5}"

# envs verificadas no codigo
ENV_TRACE=(
  "GGML_SCHED_SPLIT_TIMING=1"
  "GGML_SCHED_SPLIT_SPAN=1"
  "LLAMA_GRAPH_PREP_PROFILE=1"
  "LLAMA_F5_STAGE_TIMING=1"
  "GGML_CUDA_GRAPH_DEBUG=1"
)
ENV_CLEAN=(
  "LLAMA_LOG_COLORS=0"
)

SERVER_FLAGS=(
  --host 127.0.0.1
  -np 1
  --no-warmup
  --metrics
  --no-webui
  --alias fn40
)

# flags que o dry-run valida contra o fonte ou contra o --help
REQUIRED_FLAGS=(
  -m -ngl -ncmoe --moe-cache-mib -c -b -ub -fa --load-mode --lazy-mode
  -ctk -ctv --host -np --no-warmup --metrics --no-webui --alias --port
)

ok()   { printf '  OK   %s\n' "$*"; }
warn() { printf '  WARN %s\n' "$*"; }
fail() { printf '  FAIL %s\n' "$*" >&2; exit 2; }
info() { printf '  ..   %s\n' "$*"; }

usage() {
  sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'
  cat <<EOF

Opcoes:
  --server PATH        llama-server (default: $SERVER)
  --model PATH         diretorio ou shard 1 do GGUF (default: $MODEL)
  --outdir DIR         diretorio de saida (default: /tmp/opencode/profiling/phase0-<ts>)
  --port N             porta local (default: 8097)
  --target-tokens N    tokens do prompt (default: 28600)
  --ncmoe N            layers com experts no host (default: 2)
  --moe-cache-mib N    cache de experts na GPU; -1 = auto (default: 6144)
  --gpu-total-mib N    VRAM total para o check do dry-run (ex.: 40960)
  --runs both|clean|trace   (default: both)
  --nsys auto|on|off   (default: auto)
  --split-timing auto|on|off  (default: auto = on no trace)
  --drop-caches auto|yes|no   (default: auto = sim se root)
  --allow-busy-gpu     nao aborta se houver processo na GPU
  --source-dir DIR     fonte do beellama para validar flags/envs
  --check-server-help  roda `llama-server --help` no preflight
  --keep-server        nao mata o servidor no fim (debug)
  --no-stats           nao roda os relatorios padrao do nsys
  --load-timeout S     timeout do /health (default: 3600)
  --request-timeout S  timeout do prefill (default: 3600)
  --settle S           espera apos /health (default: 5)
  --force              permite outdir nao vazio
  --dry-run            valida tudo sem GPU, sem modelo e sem subir servidor
  -h, --help
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --server) SERVER="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --outdir) OUTDIR="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --target-tokens) TARGET_TOKENS="$2"; shift 2;;
    --ncmoe) NCMOE="$2"; shift 2;;
    --moe-cache-mib) MOE_CACHE_MIB="$2"; shift 2;;
    --gpu-total-mib) GPU_TOTAL_MIB="$2"; shift 2;;
    --runs) RUNS="$2"; shift 2;;
    --nsys) NSYS_MODE="$2"; shift 2;;
    --split-timing) SPLIT_TIMING="$2"; shift 2;;
    --drop-caches) DROP_CACHES="$2"; shift 2;;
    --allow-busy-gpu) ALLOW_BUSY_GPU=1; shift;;
    --source-dir) SOURCE_DIR="$2"; shift 2;;
    --check-server-help) CHECK_HELP=1; shift;;
    --keep-server) KEEP_SERVER=1; shift;;
    --no-stats) NO_STATS=1; shift;;
    --load-timeout) LOAD_TIMEOUT="$2"; shift 2;;
    --request-timeout) REQUEST_TIMEOUT="$2"; shift 2;;
    --settle) SETTLE="$2"; shift 2;;
    --force) FORCE=1; shift;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) fail "opcao desconhecida: $1 (use --help)";;
  esac
done

PROD_FLAGS=(
  -ngl 99
  -ncmoe "$NCMOE"
  --moe-cache-mib "$MOE_CACHE_MIB"
  -c 32768
  -b 2048
  -ub 1024
  -fa on
  --load-mode mmap
  --lazy-mode on
  -ctk q8_0
  -ctv q8_0
)

case "$RUNS" in both|clean|trace) ;; *) fail "--runs invalido: $RUNS";; esac
case "$NSYS_MODE" in auto|on|off) ;; *) fail "--nsys invalido: $NSYS_MODE";; esac
case "$SPLIT_TIMING" in auto|on|off) ;; *) fail "--split-timing invalido: $SPLIT_TIMING";; esac
case "$DROP_CACHES" in auto|yes|no) ;; *) fail "--drop-caches invalido: $DROP_CACHES";; esac

if [ -z "$OUTDIR" ]; then
  OUTDIR="/tmp/opencode/profiling/phase0-$(date +%Y%m%d-%H%M%S)"
fi

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

have() { command -v "$1" >/dev/null 2>&1; }

port_is_free() {
  python3 - "$PORT" <<'PY'
import socket, sys
p = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(("127.0.0.1", p))
except OSError:
    sys.exit(1)
finally:
    s.close()
PY
}

gpu_compute_apps() {
  have nvidia-smi || return 0
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
}

server_pid_by_port() {
  local pid=""
  if have ss; then
    pid=$(ss -ltnp 2>/dev/null | awk -v p=":${PORT}" '$4 ~ p"$" {print}' \
          | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2 || true)
  fi
  if [ -z "$pid" ] && have pgrep; then
    pid=$(pgrep -f -- "llama-server.*--port ${PORT}" | head -1 || true)
  fi
  printf '%s' "$pid"
}

wait_health() {
  local t0=$SECONDS
  while [ $((SECONDS - t0)) -lt "$LOAD_TIMEOUT" ]; do
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      return 0
    fi
    local pid
    pid="$(server_pid_by_port)"
    if [ -n "$pid" ] && [ ! -d "/proc/$pid" ]; then
      return 1
    fi
    sleep 2
  done
  return 1
}

kill_server() {
  local pid="$1"
  [ -n "$pid" ] || return 0
  if [ -d "/proc/$pid" ]; then
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 120); do
      [ -d "/proc/$pid" ] || break
      sleep 1
    done
    if [ -d "/proc/$pid" ]; then
      kill -KILL "$pid" 2>/dev/null || true
    fi
  fi
}

nsys_export_sqlite() {
  local rep="$1" db="$2"
  # a sintaxe de --force-overwrite mudou entre versoes
  if nsys export -t sqlite -f true -o "$db" "$rep" >/dev/null 2>&1; then return 0; fi
  if nsys export --type=sqlite --force-overwrite=true -o "$db" "$rep" >/dev/null 2>&1; then return 0; fi
  nsys export -t sqlite -o "$db" "$rep" >/dev/null 2>&1
}

nsys_profile_cmd() {
  local out="$1"
  echo nsys profile \
    --trace=cuda,osrt \
    --sample=none \
    --cpuctxsw=process-tree \
    --cuda-memory-usage=false \
    --stats=false \
    --force-overwrite=true \
    --kill=sigterm \
    -o "$out"
}

check_vram_budget() {
  # $1 = model_info.json, $2 = total MiB, $3 = strict (1 = falha se nao couber)
  python3 - "$1" "$NCMOE" "$MOE_CACHE_MIB" "$2" "$3" <<'PY'
import json, sys
info = json.load(open(sys.argv[1]))
ncmoe = int(sys.argv[2])
cache_mib = int(sys.argv[3])
total_mib = int(sys.argv[4])
strict = int(sys.argv[5])
per = [v for _k, v in sorted(info.get("expert_bytes_per_layer", {}).items(),
                             key=lambda kv: int(kv[0]))]
if not per:
    print("  ..   modelo sem experts host-residentes; check de VRAM pulado")
    sys.exit(0)
expert = sum(per)
per_layer = per[0] if per else 0
host = min(ncmoe * per_layer, expert) if per_layer else 0
total = sum(f["bytes"] for f in info["files"])
ple = (info.get("ple") or {}).get("bytes", 0)
gpu_weights = max(0, total - ple - host)
cache = host if cache_mib < 0 else cache_mib * 1024 * 1024
margin = 4096 * 1024 * 1024
need = gpu_weights + cache + margin
mib = lambda b: b / (1024.0 * 1024.0)
print("  ..   VRAM: pesos GPU %.0f + cache %.0f + margem 4096 = %.0f MiB (total %d MiB, host experts %.0f MiB)" % (
    mib(gpu_weights), mib(cache), mib(need), total_mib, mib(host)))
if total_mib and need > total_mib * 1024 * 1024:
    print("  FAIL nao cabe: faltam %.0f MiB; use --moe-cache-mib -1 (auto) ou --ncmoe 8" % (
        mib(need - total_mib * 1024 * 1024)))
    sys.exit(1)
if total_mib:
    print("  OK   cabe; folga %.0f MiB" % mib(total_mib * 1024 * 1024 - need))
sys.exit(0)
PY
}

# ---------------------------------------------------------------------------
# dry-run
# ---------------------------------------------------------------------------

validate_flags_from_source() {
  local src="$1" missing=0
  [ -f "$src/common/arg.cpp" ] || { warn "sem $src/common/arg.cpp"; return 1; }
  for f in "${REQUIRED_FLAGS[@]}"; do
    if ! grep -Fq "\"$f\"" "$src/common/arg.cpp"; then
      printf '  FAIL flag %s nao encontrada em common/arg.cpp\n' "$f" >&2
      missing=1
    fi
  done
  for e in "${ENV_TRACE[@]}"; do
    local name="${e%%=*}"
    if ! grep -Rqs "$name" "$src/src" "$src/ggml/src"; then
      printf '  FAIL env %s nao encontrada no fonte\n' "$name" >&2
      missing=1
    fi
  done
  [ "$missing" -eq 0 ] && ok "flags e envs conferidas em $src"
  return "$missing"
}

validate_flags_from_help() {
  local bin="$1" help missing=0
  [ -x "$bin" ] || { warn "sem binario para --help"; return 1; }
  help="$( cd "$(dirname "$bin")" && "./$(basename "$bin")" --help 2>&1 || true )"
  [ -n "$help" ] || { warn "--help vazio"; return 1; }
  for f in "${REQUIRED_FLAGS[@]}"; do
    printf '%s\n' "$help" | grep -Fq -- "$f" || {
      printf '  FAIL flag %s ausente no --help do binario\n' "$f" >&2
      missing=1
    }
  done
  [ "$missing" -eq 0 ] && ok "flags conferidas no --help do binario"
  return "$missing"
}

validate_env_in_libs() {
  local bindir="$1" missing=0 found=0
  for e in "${ENV_TRACE[@]}"; do
    local name="${e%%=*}"
    found=0
    for lib in "$bindir"/libllama.so* "$bindir"/libggml*.so*; do
      [ -f "$lib" ] || continue
      if strings "$lib" 2>/dev/null | grep -Fq "$name"; then found=1; break; fi
    done
    if [ "$found" -eq 0 ]; then
      warn "env $name nao encontrada nas libs de $bindir (pode ser build antigo)"
    fi
  done
  return 0
}

do_dry_run() {
  echo "=== DRY-RUN profile_prefill.sh ==="
  bash -n "$0" && ok "bash -n $0"
  python3 -m py_compile "$PY_DRIVER" "$PY_SUMMARY" && ok "python3 -m py_compile"
  python3 "$PY_DRIVER" --help >/dev/null && ok "ablate_lazy.py --help"
  python3 "$PY_SUMMARY" --help >/dev/null && ok "summarize_nsys.py --help"

  if [ -z "$SOURCE_DIR" ] && [ -d /home/hjotha/beellama.cpp ]; then
    SOURCE_DIR=/home/hjotha/beellama.cpp
  fi
  if [ -n "$SOURCE_DIR" ] && [ -d "$SOURCE_DIR" ]; then
    validate_flags_from_source "$SOURCE_DIR" || fail "validacao no fonte falhou"
  else
    warn "sem --source-dir; validacao de flags no fonte pulada"
  fi

  if [ -x "$SERVER" ]; then
    ok "server encontrado: $SERVER"
    if [ "$CHECK_HELP" -eq 1 ]; then
      validate_flags_from_help "$SERVER" || fail "flags divergem do binario"
    else
      info "--check-server-help nao passado; --help do binario nao foi executado"
    fi
    validate_env_in_libs "$(dirname "$SERVER")"
  else
    warn "server ausente agora: $SERVER (esperado fora da A100)"
  fi

  if [ -e "$MODEL" ]; then
    local mi_json="${TMPDIR:-/tmp}/model-info.$$.json"
    if python3 "$PY_DRIVER" model-info --model "$MODEL" >"$mi_json"; then
      python3 - "$mi_json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print("  OK   modelo: %d shard(s), %s layers, expert type %s (%s), PLE %.2f GiB" % (
    len(d["files"]), d.get("n_layers"), d.get("expert_type_name"),
    d.get("expert_type"), (d.get("ple") or {}).get("bytes", 0) / 1024**3))
PY
      ok "cabecalho GGUF legivel"

      local total_mib="$GPU_TOTAL_MIB"
      if [ "$total_mib" -eq 0 ] && have nvidia-smi; then
        total_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
      fi
      if [ -n "$total_mib" ] && [ "$total_mib" -gt 0 ]; then
        check_vram_budget "$mi_json" "$total_mib" 0 || \
          warn "a config pedida nao cabe nesta GPU; na A100 use --moe-cache-mib -1 ou --ncmoe 8"
      else
        info "VRAM total desconhecida; passe --gpu-total-mib 40960 para conferir o orcamento"
      fi
    else
      fail "cabecalho GGUF invalido: $MODEL"
    fi
    rm -f "$mi_json"
  else
    warn "modelo ausente agora: $MODEL (esperado fora da A100)"
  fi

  if port_is_free; then ok "porta $PORT livre"; else fail "porta $PORT ocupada"; fi

  if have nsys; then ok "nsys: $(nsys --version 2>&1 | head -1)"; else warn "nsys ausente (trace caira para log-only)"; fi
  if have ncu; then info "ncu presente (nao sera usado; veja README)"; fi
  if have nvidia-smi; then
    info "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
    apps="$(gpu_compute_apps)"
    if [ -n "$apps" ]; then warn "GPU ocupada: $apps"; else ok "GPU sem processos de compute"; fi
  else
    warn "nvidia-smi ausente (esperado fora da A100)"
  fi

  echo
  echo "--- comando que sera executado no run clean ---"
  echo "env ${ENV_CLEAN[*]} $SERVER -m $MODEL ${PROD_FLAGS[*]} ${SERVER_FLAGS[*]} --port $PORT"
  echo
  echo "--- comando que sera executado no run trace ---"
  echo "env ${ENV_TRACE[*]} $(nsys_profile_cmd "$OUTDIR/trace/prefill") -- $SERVER -m $MODEL ${PROD_FLAGS[*]} ${SERVER_FLAGS[*]} --port $PORT"
  echo
  echo "--- depois do request ---"
  echo "ablate_lazy.py drive --port $PORT --prompt-file $OUTDIR/prompt.txt --target-tokens $TARGET_TOKENS --server-pid PID --out-json $OUTDIR/<run>/client.json"
  echo "summarize_nsys.py --sqlite $OUTDIR/trace/prefill.sqlite --outdir $OUTDIR/trace --drive-json $OUTDIR/trace/client.json --server-log $OUTDIR/trace/server.log --expert-type <id>"
  echo
  echo "OK: dry-run (nada foi executado, nenhuma GPU foi tocada)"
  exit 0
}

# ---------------------------------------------------------------------------
# preflight real
# ---------------------------------------------------------------------------

preflight() {
  echo "== preflight =="
  [ -x "$SERVER" ] || fail "llama-server nao executavel: $SERVER"
  [ -e "$MODEL" ] || fail "modelo ausente: $MODEL"
  SERVER="$(cd "$(dirname "$SERVER")" && pwd)/$(basename "$SERVER")"
  MODEL="$(cd "$(dirname "$MODEL")" && pwd)/$(basename "$MODEL")"
  have python3 || fail "python3 ausente"
  have curl || fail "curl ausente"
  port_is_free || fail "porta $PORT ocupada"

  local apps
  apps="$(gpu_compute_apps)"
  if [ -n "$apps" ] && [ "$ALLOW_BUSY_GPU" -eq 0 ]; then
    fail "GPU ocupada (a A100 pode estar no benchmark):
$apps
use --allow-busy-gpu somente se for proposital"
  fi

  mkdir -p "$OUTDIR"
  if [ -n "$(ls -A "$OUTDIR" 2>/dev/null || true)" ] && [ "$FORCE" -eq 0 ]; then
    fail "outdir nao vazio: $OUTDIR (use --force)"
  fi

  python3 "$PY_DRIVER" model-info --model "$MODEL" >"$OUTDIR/model_info.json"
  EXPERT_TYPE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["expert_type"])' "$OUTDIR/model_info.json")"
  PLE_GIB="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("%.1f"%((d.get("ple") or {}).get("bytes",0)/1024**3))' "$OUTDIR/model_info.json")"
  ok "expert type $EXPERT_TYPE, PLE ${PLE_GIB} GiB"

  # orcamento de VRAM: pesos GPU + cache de experts + margem para KV/buffers.
  # necessario porque --moe-cache-mib 6144 com -ncmoe 2 nao cabe nos 39 GiB.
  if have nvidia-smi; then
    local total_mib
    total_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    [ -n "$total_mib" ] && [ "$total_mib" -gt 0 ] || total_mib="$GPU_TOTAL_MIB"
    if [ -n "$total_mib" ] && [ "$total_mib" -gt 0 ]; then
      check_vram_budget "$OUTDIR/model_info.json" "$total_mib" 1 || fail "configuracao de VRAM nao cabe na GPU (veja acima)"
      ok "orcamento de VRAM conferido"
    else
      warn "VRAM total desconhecida; orcamento nao conferido"
    fi
  else
    warn "sem nvidia-smi; orcamento de VRAM nao conferido"
  fi

  validate_flags_from_help "$SERVER" || fail "flags divergem do binario"

  if [ "$NSYS_MODE" != "off" ]; then
    if ! have nsys; then
      [ "$NSYS_MODE" = "on" ] && fail "--nsys on mas nsys ausente"
      warn "nsys ausente; run trace sera log-only"
      NSYS_MODE=off
    else
      # probe barato: valida licenca/injecao sem CUDA
      local probe="$OUTDIR/_nsys_probe"
      if ( cd "$(dirname "$SERVER")" && \
           nsys profile --trace=cuda,osrt --sample=none --cpuctxsw=process-tree \
             --cuda-memory-usage=false --stats=false --force-overwrite=true \
             --kill=sigterm -o "$probe" -- "$SERVER" --version >/dev/null 2>&1 ); then
        ok "nsys operacional ($(nsys --version 2>&1 | head -1))"
      else
        [ "$NSYS_MODE" = "on" ] && fail "nsys nao conseguiu perfilar --version"
        warn "nsys falhou no probe; run trace sera log-only"
        NSYS_MODE=off
      fi
      rm -f "$probe".* 2>/dev/null || true
    fi
  fi

  local free_kib
  free_kib=$(df -Pk "$OUTDIR" | awk 'NR==2 {print $4}')
  [ "$free_kib" -ge $((5*1024*1024)) ] || fail "menos de 5 GiB livres em $OUTDIR"
  ok "espaco livre: $((free_kib/1024/1024)) GiB"
}

# ---------------------------------------------------------------------------
# um run (clean ou trace)
# ---------------------------------------------------------------------------

run_one() {
  local name="$1"
  local dir="$OUTDIR/$name"
  mkdir -p "$dir"
  local log="$dir/server.log"

  local -a envs
  local nsys_prefix=()
  if [ "$name" = "trace" ]; then
    envs=("${ENV_TRACE[@]}")
    if [ "$NSYS_MODE" != "off" ]; then
      nsys_prefix=(nsys profile --trace=cuda,osrt --sample=none --cpuctxsw=process-tree
                   --cuda-memory-usage=false --stats=false --force-overwrite=true
                   --kill=sigterm -o "$dir/prefill")
    fi
  else
    envs=("${ENV_CLEAN[@]}")
  fi

  if [ "$DROP_CACHES" = "yes" ] || { [ "$DROP_CACHES" = "auto" ] && [ "$(id -u)" -eq 0 ]; }; then
    sync
    if echo 3 > /proc/sys/vm/drop_caches 2>/dev/null; then
      ok "[$name] drop_caches feito (PLE/pesos frios)"
    else
      warn "[$name] drop_caches falhou; o braco cold pode estar quente"
    fi
  else
    warn "[$name] sem drop_caches (nao-root ou --drop-caches no)"
  fi

  local nvidia_before="$dir/nvidia-smi-before.txt"
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,clocks.sm,clocks.mem,power.draw,temperature.gpu \
    --format=csv,noheader >"$nvidia_before" 2>&1 || true

  local cmd=("$SERVER" -m "$MODEL" "${PROD_FLAGS[@]}" "${SERVER_FLAGS[@]}" --port "$PORT")
  {
    echo "# RUN: $name"
    echo "# DATE: $(date -Is)"
    echo "# CMD: ${nsys_prefix[*]} ${cmd[*]}"
    echo "# ENV: ${envs[*]}"
    echo "#"
  } >"$log"

  local t0
  t0=$(date +%s)
  # shellcheck disable=SC2086
  env "${envs[@]}" "${nsys_prefix[@]}" "${cmd[@]}" >>"$log" 2>&1 &
  local wrapper_pid=$!

  if ! wait_health; then
    tail -60 "$log" >&2 || true
    kill_server "$wrapper_pid"
    fail "[$name] servidor nao chegou a /health em ${LOAD_TIMEOUT}s"
  fi
  local load_s=$(( $(date +%s) - t0 ))
  local pid
  pid="$(server_pid_by_port)"
  [ -n "$pid" ] || { kill_server "$wrapper_pid"; fail "[$name] nao achei o pid do llama-server"; }
  ok "[$name] health em ${load_s}s (pid $pid)"
  echo "$pid" >"$dir/server.pid"
  echo "$load_s" >"$dir/load_s.txt"
  sleep "$SETTLE"

  if ! python3 "$PY_DRIVER" drive \
      --port "$PORT" \
      --prompt-file "$OUTDIR/prompt.txt" \
      --target-tokens "$TARGET_TOKENS" \
      --calibrate \
      --server-pid "$pid" \
      --out-json "$dir/client.json" \
      --request-timeout "$REQUEST_TIMEOUT" 2>&1 | tee "$dir/drive.log"; then
    kill_server "$pid"
    fail "[$name] request de prefill falhou (veja $dir/drive.log)"
  fi

  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,clocks.sm,clocks.mem,power.draw,temperature.gpu \
    --format=csv,noheader >"$dir/nvidia-smi-after.txt" 2>&1 || true

  if [ "$KEEP_SERVER" -eq 1 ]; then
    warn "[$name] --keep-server: pid $pid continua rodando"
  else
    kill_server "$pid"
    # espera o nsys finalizar o .nsys-rep
    if [ "${#nsys_prefix[@]}" -gt 0 ]; then
      for _ in $(seq 1 300); do
        [ -d "/proc/$wrapper_pid" ] || break
        sleep 1
      done
      wait "$wrapper_pid" 2>/dev/null || true
      for _ in $(seq 1 180); do
        [ -f "$dir/prefill.nsys-rep" ] && break
        sleep 1
      done
    fi
  fi

  if [ "$name" = "trace" ] && [ "${#nsys_prefix[@]}" -eq 0 ]; then
    local wall_ms
    wall_ms="$(python3 -c 'import json,sys; print((json.load(open(sys.argv[1])).get("timings") or {}).get("prompt_ms") or 0)' "$dir/client.json")"
    python3 "$PY_SUMMARY" --log-only --outdir "$dir" --drive-json "$dir/client.json" \
      --server-log "$log" --wall-ms "$wall_ms" | tee "$dir/log_summary.stdout"
  fi

  if [ "$name" = "trace" ] && [ "${#nsys_prefix[@]}" -gt 0 ]; then
    if [ ! -f "$dir/prefill.nsys-rep" ]; then
      warn "[$name] nsys nao gerou $dir/prefill.nsys-rep; caindo para log-only"
      local wall_ms
      wall_ms="$(python3 -c 'import json,sys; print((json.load(open(sys.argv[1])).get("timings") or {}).get("prompt_ms") or 0)' "$dir/client.json")"
      python3 "$PY_SUMMARY" --log-only --outdir "$dir" --drive-json "$dir/client.json" \
        --server-log "$log" --wall-ms "$wall_ms" | tee "$dir/log_summary.stdout"
    else
      ok "[$name] nsys-rep: $(du -h "$dir/prefill.nsys-rep" | cut -f1)"
      if nsys_export_sqlite "$dir/prefill.nsys-rep" "$dir/prefill.sqlite"; then
        ok "[$name] sqlite exportado"
        local wall_ms
        wall_ms="$(python3 -c 'import json,sys; print((json.load(open(sys.argv[1])).get("timings") or {}).get("prompt_ms") or 0)' "$dir/client.json")"
        python3 "$PY_SUMMARY" \
          --sqlite "$dir/prefill.sqlite" \
          --outdir "$dir" \
          --drive-json "$dir/client.json" \
          --server-log "$log" \
          --expert-type "$EXPERT_TYPE" \
          --wall-ms "$wall_ms" \
          | tee "$dir/class_summary.stdout"
      else
        warn "[$name] export sqlite falhou; rode 'nsys export -t sqlite' manualmente"
      fi
      if [ "$NO_STATS" -eq 0 ]; then
        mkdir -p "$dir/nsys-stats"
        ( cd "$dir/nsys-stats" && \
          nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum \
            --format csv --force-export=true --output . "$dir/prefill.nsys-rep" \
            >nsys-stats.log 2>&1 ) || warn "[$name] nsys stats opcional falhou"
      fi
    fi
  fi
}

# ---------------------------------------------------------------------------
# aceitacao: clean vs trace
# ---------------------------------------------------------------------------

acceptance() {
  local clean="$OUTDIR/clean/client.json" trace="$OUTDIR/trace/client.json"
  local out="$OUTDIR/acceptance.md"
  {
    echo "# Aceitacao da Fase 0"
    echo
    echo "- data: $(date -Is)"
    echo "- host: $(hostname)"
    echo "- server: \`$SERVER\`"
    echo "- modelo: \`$MODEL\`"
    echo "- flags: \`${PROD_FLAGS[*]}\`"
    echo "- prompt alvo: $TARGET_TOKENS tokens"
    echo
  } >"$out"

  if [ -f "$clean" ] && [ -f "$trace" ]; then
    python3 - "$clean" "$trace" >>"$out" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
t = json.load(open(sys.argv[2]))
cm = (c.get("timings") or {}).get("prompt_ms")
tm = (t.get("timings") or {}).get("prompt_ms")
print("| run | tokens | prompt_ms | t/s |")
print("|---|---:|---:|---:|")
for name, d in (("clean", c), ("trace", t)):
    ti = d.get("timings") or {}
    print("| %s | %s | %.1f | %.2f |" % (
        name, ti.get("prompt_n"), ti.get("prompt_ms") or 0,
        ti.get("prompt_per_second") or 0))
if cm and tm:
    delta = (tm - cm) / cm * 100.0
    print()
    print("Overhead do trace (nsys + envs de diagnostico): **%.2f%%**" % delta)
    print()
    print("Criterio de baseline ±5%%: **%s**" % ("PASS" if abs(delta) <= 5.0 else "FAIL"))
    print()
    print("Se FAIL, use `clean` como denominador do wall e trate as fracoes do trace")
    print("como relativas (os kernels do nsys continuam corretos; os gaps do host")
    print("podem estar inflados pelo synchronize por split).")
PY
  else
    echo "Runs clean/trace incompletos; nada a comparar." >>"$out"
  fi

  if [ -f "$OUTDIR/trace/class_summary.md" ]; then
    {
      echo
      echo "## Atribuicao (trace)"
      echo
      sed -n '/| classe | ms |/,$p' "$OUTDIR/trace/class_summary.md" | head -30
    } >>"$out"
  fi

  echo
  cat "$out"
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if [ "$DRY_RUN" -eq 1 ]; then
  do_dry_run
fi

echo "=== profile_prefill.sh ==="
echo "outdir: $OUTDIR"
preflight

if [ "$SPLIT_TIMING" = "auto" ]; then
  SPLIT_TIMING=on
fi
if [ "$SPLIT_TIMING" = "off" ]; then
  # remove apenas GGML_SCHED_SPLIT_TIMING; SPAN continua barato
  filtered=()
  for e in "${ENV_TRACE[@]}"; do
    [ "$e" = "GGML_SCHED_SPLIT_TIMING=1" ] || filtered+=("$e")
  done
  ENV_TRACE=("${filtered[@]}")
fi

case "$RUNS" in
  clean) run_one clean ;;
  trace) run_one trace ;;
  both) run_one clean; run_one trace ;;
esac

acceptance

echo
echo "=== arquivos ==="
find "$OUTDIR" -maxdepth 2 -type f -printf '  %p (%s bytes)\n' | sort
echo
echo "Repetir depois:"
echo "  $0 --server $SERVER --model $MODEL --outdir $OUTDIR-r2 --runs both"

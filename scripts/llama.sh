#!/usr/bin/env bash
# T4 setup based on the tested colab-qwen configuration; this integration is unmeasured.
llama_install() {
  local base=/content/collabosm-llama tarfile bin
  mkdir -p "$base/llama" "$MODEL_DIR" || return 1
  status "stage=runtime"
  bin=$(find "$base/llama" -maxdepth 3 -type f -name llama-server -print -quit)
  if [ -z "$bin" ]; then
    tarfile=$(mktemp) || return 1
    curl -fL --retry 2 -o "$tarfile" 'https://github.com/ggml-org/llama.cpp/releases/download/b12318/llama-b12318-bin-ubuntu-cuda-12.8-x64.tar.gz' && tar -xzf "$tarfile" -C "$base/llama" || { rm -f "$tarfile"; return 1; }
    rm -f "$tarfile"
    bin=$(find "$base/llama" -maxdepth 3 -type f -name llama-server -print -quit)
  fi
  [ -n "$bin" ] || return 1
  export LD_LIBRARY_PATH="$(dirname "$bin"):/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}"
  if ldd "$bin" "$(dirname "$bin")/libggml-cuda.so" 2>&1 | grep -q 'not found'; then
    tarfile=$(mktemp) || return 1
    curl -fL --retry 2 -o "$tarfile" 'https://github.com/ggml-org/llama.cpp/releases/download/b12318/cudart-llama-b12318-bin-ubuntu-cuda-12.8-x64.tar.gz' && tar -xzf "$tarfile" --strip-components=1 -C "$(dirname "$bin")" || { rm -f "$tarfile"; return 1; }
    rm -f "$tarfile"
  fi
  status "stage=weights"
  python - <<'PY' || return 1
import os
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id=os.environ['MODEL_REPO'], revision=os.environ['MODEL_REVISION'], filename=os.environ['LLAMA_FILE'], local_dir=os.environ['MODEL_DIR'])
PY
}
llama_start() {
  local bin
  bin=$(find /content/collabosm-llama/llama -maxdepth 3 -type f -name llama-server -print -quit)
  [ -n "$bin" ] && [ -f "$MODEL_DIR/$LLAMA_FILE" ] || return 1
  export LD_LIBRARY_PATH="$(dirname "$bin"):/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}"
  LLAMA_PROC='collabosm-llama/llama.*/llama-server'
  nohup "$bin" -m "$MODEL_DIR/$LLAMA_FILE" --alias "$MODEL_ID" -ngl 99 -fa on -c "${LLAMA_CONTEXT:-32768}" --parallel 1 -b 2048 -ub 256 -ctk q8_0 -ctv q8_0 --spec-type draft-mtp --spec-draft-n-max 2 --fit off --no-kv-offload --jinja --host 127.0.0.1 --port "$PORT" --api-key-file "$KEY_FILE" > "$LOG" 2>&1 < /dev/null &
}

#!/usr/bin/env bash
# T4 setup using the user's beellama.cpp fork with local KVarN4 tiers (no remote-attn).
llama_install() {
  local base=/content/collabosm-llama bin
  mkdir -p "$base/llama" "$base/slot-cache" "$MODEL_DIR" || return 1
  status "stage=runtime"
  bin=$(find "$base/llama" -maxdepth 4 -type f -name llama-server -print -quit)
  if [ -z "$bin" ]; then
    if [ ! -f /content/beellama.zip ] && ls /content/beellama.zip.part* >/dev/null 2>&1; then
      cat /content/beellama.zip.part* > /content/beellama.zip && rm -f /content/beellama.zip.part*
    fi
    if [ ! -f /content/beellama.zip ] && [ -n "${BEELLAMA_ZIP_URL:-}" ]; then
      curl -fL --retry 2 -o /content/beellama.zip "$BEELLAMA_ZIP_URL" || return 1
    fi
    if [ -f /content/beellama.zip ]; then
      if command -v unzip >/dev/null 2>&1; then
        unzip -q -o /content/beellama.zip -d "$base/llama" || return 1
      else
        python3 - "$base/llama" <<'PY' || return 1
import os, stat, sys, zipfile
dest = sys.argv[1]
with zipfile.ZipFile("/content/beellama.zip") as zf:
    for info in zf.infolist():
        target = os.path.join(dest, info.filename)
        if stat.S_ISLNK(info.external_attr >> 16):
            os.makedirs(os.path.dirname(target), exist_ok=True)
            if os.path.lexists(target):
                os.unlink(target)
            os.symlink(zf.read(info).decode(), target)
        else:
            zf.extract(info, dest)
PY
      fi
      bin=$(find "$base/llama" -maxdepth 4 -type f -name llama-server -print -quit)
      [ -n "$bin" ] && chmod +x "$bin" || return 1
    else
      echo "ERROR: /content/beellama.zip not found" >&2
      return 1
    fi
  fi
  export LD_LIBRARY_PATH="$(dirname "$bin"):/usr/local/cuda/lib64:/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}"
  if ldd "$bin" "$(dirname "$bin")/libggml-cuda.so" 2>&1 | grep -q 'not found'; then
    ldd "$bin" "$(dirname "$bin")/libggml-cuda.so" 2>&1 | grep 'not found' >&2
    return 1
  fi
  status "stage=weights"
  python - <<'PY' || return 1
import os
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id=os.environ['MODEL_REPO'], revision=os.environ['MODEL_REVISION'], filename=os.environ['LLAMA_FILE'], local_dir=os.environ['MODEL_DIR'])
PY
}

llama_write_ini() {
  local ini=/content/collabosm-llama/t4-production.ini
  cat > "$ini" <<EOF
; CollabOSM T4 beellama preset (local KVarN4 tiers, no remote-attn)
version = 1

[*]
load-mode = none
parallel = 1
device = CUDA0
flash-attn = on
fit = off
ctx-checkpoints = 1
metrics = true
slots = true
cache-prompt = true
context-shift = false
gpu-layers = 99
cache-type-k = kvarn4
cache-type-v = kvarn4
spec-draft-type-k = kvarn4
spec-draft-type-v = kvarn4
cache-ram = ${LLAMA_CACHE_RAM:-4096}
slot-save-auto = true
slot-save-max-count = 16
slot-save-max-mb = 12288
slot-save-path = /content/collabosm-llama/slot-cache/
temp = 1.0
top-p = 0.95
top-k = 20
min-p = 0.05
reasoning-effort = low

kvarn-window-chunk = 4096
cuda-graph-recovery-headroom-mb = 18
ovf-packed-view = true
ovf-cont-kv = true
ovf-tail-local-cuda = true
ovf-qwht-cuda = true
vk-get-rows-q4-0-f16vec = false
slot-save-split-prefix = true
native-snapshot-auto-save = true
native-snapshot-auto-restore = true
adaptive-direct-pure-disk-restore = true

[$MODEL_ID]
model = $MODEL_DIR/$LLAMA_FILE
alias = $MODEL_ID,qwen-3.8-27b,Qwen3.8-27B
load-on-startup = true
slot-save-auto = true
slot-save-block = 256
reasoning-effort = low

spec-type = draft-mtp
spec-dm-controller = off
spec-draft-p-min = 0.65
spec-draft-p-min-s = 0.80
spec-draft-p-min-m = 0.70
spec-draft-p-min-l = 0.65
spec-draft-p-min-xl = 0.50
spec-draft-p-min-xxl = 0.50
spec-draft-p-min-xxxl = 0.50

; Tier S: 0 - 38912 (b2048/ub256 N=6)
ctx-size-s = ${TIER_S_CTX:-38912}
mtp-s-max-tokens = ${TIER_S_CTX:-38912}
batch-size-s = 2048
ubatch-size-s = 256
cache-type-k-s = kvarn4
cache-type-v-s = kvarn4
spec-draft-type-k-s = kvarn4
spec-draft-type-v-s = kvarn4
spec-draft-n-max-s = 6

; Tier M: 38913 - 56320 (b1152/ub384 N=4)
ctx-size-m = ${TIER_M_CTX:-56320}
mtp-m-max-tokens = ${TIER_M_CTX:-56320}
batch-size-m = 1152
ubatch-size-m = 384
cache-type-k-m = kvarn4
cache-type-v-m = kvarn4
spec-draft-type-k-m = kvarn4
spec-draft-type-v-m = kvarn4
spec-draft-n-max-m = 4

; Tier L: 56321 - 75776 (b2048/ub256 N=2)
ctx-size-l = ${TIER_L_CTX:-75776}
batch-size-l = 2048
ubatch-size-l = 256
cache-type-k-l = kvarn4
cache-type-v-l = kvarn4
spec-draft-type-k-l = kvarn4
spec-draft-type-v-l = kvarn4
spec-draft-n-max-l = 2

; Tier XL: 75777 - 122112 (b2048/ub256 N=0, local CUDA)
ctx-size-xl = ${TIER_XL_CTX:-122112}
xl-max-tokens = ${TIER_XL_CTX:-122112}
batch-size-xl = 2048
ubatch-size-xl = 256
cache-type-k-xl = kvarn4
cache-type-v-xl = kvarn4
spec-draft-type-k-xl = kvarn4
spec-draft-type-v-xl = kvarn4
spec-draft-n-max-xl = 0

; Tier XXL: 122113 - 163840 (b512/ub64 N=0, local CUDA)
ctx-size-xxl = ${TIER_XXL_CTX:-163840}
xxlong-max-tokens = ${TIER_XXL_CTX:-163840}
batch-size-xxl = 512
ubatch-size-xxl = 64
cache-type-k-xxl = kvarn4
cache-type-v-xxl = kvarn4
spec-draft-type-k-xxl = kvarn4
spec-draft-type-v-xxl = kvarn4
spec-draft-n-max-xxl = 0

; Tier XXXL: 163841 - ${LLAMA_CONTEXT:-204800} (b512/ub64 N=0, local CUDA T4 max ~14.7/15.3 GiB)
ctx-size-xxxl = ${LLAMA_CONTEXT:-204800}
xxxlong-max-tokens = ${LLAMA_CONTEXT:-204800}
batch-size-xxxl = 512
ubatch-size-xxxl = 64
cache-type-k-xxxl = kvarn4
cache-type-v-xxxl = kvarn4
spec-draft-type-k-xxxl = kvarn4
spec-draft-type-v-xxxl = kvarn4
spec-draft-n-max-xxxl = 0
EOF
}

llama_start() {
  local bin
  bin=$(find /content/collabosm-llama/llama -maxdepth 4 -type f -name llama-server -print -quit)
  [ -n "$bin" ] && [ -f "$MODEL_DIR/$LLAMA_FILE" ] || return 1
  export LD_LIBRARY_PATH="$(dirname "$bin"):/usr/local/cuda/lib64:/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}"
  llama_write_ini || return 1
  LLAMA_PROC='collabosm-llama/llama.*/llama-server'
  nohup "$bin" --models-preset /content/collabosm-llama/t4-production.ini --models-max 1 --host 127.0.0.1 --port "$PORT" --api-key-file "$KEY_FILE" > "$LOG" 2>&1 < /dev/null &
}

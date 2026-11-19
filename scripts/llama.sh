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
  if [ "${LLAMA_STREAM_VRAM:-1}" = "1" ]; then
    echo "[llama] direct HF->VRAM streaming enabled (--hf-stream-vram); skipping pre-download to disk"
    return 0
  fi
  status "stage=weights"
  python - <<'PY' || return 1
import concurrent.futures, os, urllib.request
from huggingface_hub import hf_hub_download, hf_hub_url

repo = os.environ["MODEL_REPO"]
rev = os.environ["MODEL_REVISION"]
files = [os.environ["LLAMA_FILE"]] + os.environ.get("LLAMA_EXTRA_FILES", "").split()
dest_dir = os.environ["MODEL_DIR"]

def parallel_download(fname) -> bool:
    dest = os.path.join(dest_dir, fname)
    os.makedirs(os.path.dirname(dest) or dest_dir, exist_ok=True)
    url = hf_hub_url(repo_id=repo, revision=rev, filename=fname)
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=20) as r:
        final_url = r.geturl()
        size = int(r.headers.get("Content-Length", "0") or 0)
        accept_ranges = r.headers.get("Accept-Ranges", "").lower()
    if size <= 0 or "bytes" not in accept_ranges:
        return False
    if os.path.isfile(dest) and os.path.getsize(dest) == size:
        return True
    part = dest + ".part"
    with open(part, "wb") as fh:
        fh.truncate(size)
    workers = 8
    chunk = (size + workers - 1) // workers
    def fetch(idx: int):
        start = idx * chunk
        end = min(size - 1, start + chunk - 1)
        if start > end:
            return
        rreq = urllib.request.Request(final_url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(rreq, timeout=120) as resp:
            if resp.status != 206:
                raise RuntimeError(f"expected 206, got {resp.status}")
            with open(part, "r+b") as out:
                out.seek(start)
                while True:
                    buf = resp.read(1024 * 1024)
                    if not buf:
                        break
                    out.write(buf)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(fetch, range(workers)))
    os.replace(part, dest)
    return True

for fname in files:
    try:
        if not parallel_download(fname):
            raise RuntimeError("range unsupported")
    except Exception as e:
        print(f"[llama] parallel download fallback ({e}), using hf_hub_download", flush=True)
        hf_hub_download(repo_id=repo, revision=rev, filename=fname, local_dir=dest_dir)
PY
}

llama_write_ini() {
  local ini=/content/collabosm-llama/t4-production.ini key value
  # Keep hand-run serve.sh and older VM environments on the catalog's defaults.
  while IFS='=' read -r key value; do
    [[ -n "${!key:-}" ]] || printf -v "$key" '%s' "$value"
  done < "$(dirname -- "${BASH_SOURCE[0]}")/llama-defaults.env" || return 1
  cat > "$ini" <<EOF
; CollabOSM beellama preset (local KV tiers, no remote-attn)
version = 1

[*]
load-mode = ${LLAMA_LOAD_MODE:-none}
lazy-mode = ${LLAMA_LAZY_MODE:-auto}
parallel = 1
device = CUDA0
flash-attn = on
fit = off
ctx-checkpoints = 1
metrics = true
slots = true
cache-prompt = true
context-shift = false
gpu-layers = ${LLAMA_GPU_LAYERS:-99}
cache-type-k = $LLAMA_KV_TYPE
cache-type-v = $LLAMA_KV_TYPE
spec-draft-type-k = ${LLAMA_SPEC_KV_TYPE:-$LLAMA_KV_TYPE}
spec-draft-type-v = ${LLAMA_SPEC_KV_TYPE:-$LLAMA_KV_TYPE}
cache-ram = ${LLAMA_CACHE_RAM:-8192}
slot-save-auto = true
slot-save-max-count = ${LLAMA_SLOT_SAVE_MAX_COUNT:-64}
slot-save-max-mb = ${LLAMA_SLOT_SAVE_MAX_MB:-81920}
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
alias = ${LLAMA_ALIASES:-$MODEL_ID,qwen-3.8-27b,Qwen3.8-27B}
load-on-startup = true
no-warmup = true
slot-save-auto = true
slot-save-block = 256
reasoning-effort = low

spec-type = ${LLAMA_SPEC_TYPE:-draft-mtp}
spec-dm-controller = off
spec-draft-p-min = $LLAMA_MTP_P_MIN
spec-draft-p-min-s = $LLAMA_MTP_P_MIN
spec-draft-p-min-m = $LLAMA_MTP_P_MIN

; Tier S
ctx-size-s = $TIER_S_CTX
mtp-s-max-tokens = $TIER_S_CTX
batch-size-s = $LLAMA_BATCH
ubatch-size-s = $LLAMA_UBATCH
cache-type-k-s = $LLAMA_KV_TYPE
cache-type-v-s = $LLAMA_KV_TYPE
spec-draft-type-k-s = ${LLAMA_SPEC_KV_TYPE:-$LLAMA_KV_TYPE}
spec-draft-type-v-s = ${LLAMA_SPEC_KV_TYPE:-$LLAMA_KV_TYPE}
spec-draft-n-max-s = $LLAMA_MTP_DRAFT

; Tier M
ctx-size-m = $TIER_M_CTX
mtp-m-max-tokens = $TIER_M_CTX
batch-size-m = $LLAMA_BATCH
ubatch-size-m = $LLAMA_UBATCH
cache-type-k-m = $LLAMA_KV_TYPE
cache-type-v-m = $LLAMA_KV_TYPE
spec-draft-type-k-m = ${LLAMA_SPEC_KV_TYPE:-$LLAMA_KV_TYPE}
spec-draft-type-v-m = ${LLAMA_SPEC_KV_TYPE:-$LLAMA_KV_TYPE}
spec-draft-n-max-m = $LLAMA_MTP_DRAFT

; Tier L
ctx-size-l = $LLAMA_CONTEXT
batch-size-l = $LLAMA_BATCH
ubatch-size-l = $LLAMA_UBATCH
cache-type-k-l = $LLAMA_KV_TYPE
cache-type-v-l = $LLAMA_KV_TYPE
spec-draft-type-k-l = ${LLAMA_SPEC_KV_TYPE:-$LLAMA_KV_TYPE}
spec-draft-type-v-l = ${LLAMA_SPEC_KV_TYPE:-$LLAMA_KV_TYPE}
spec-draft-n-max-l = $LLAMA_MTP_DRAFT_L

ctx-size-xl = 0
ctx-size-xxl = 0
ctx-size-xxxl = 0
EOF
}

llama_start() {
  local bin model_url
  bin=$(find /content/collabosm-llama/llama -maxdepth 4 -type f -name llama-server -print -quit)
  [ -n "$bin" ] || return 1
  if [ "${LLAMA_STREAM_VRAM:-1}" != "1" ] && [ ! -f "$MODEL_DIR/$LLAMA_FILE" ]; then
    return 1
  fi
  export LD_LIBRARY_PATH="$(dirname "$bin"):/usr/local/cuda/lib64:/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}"
  llama_write_ini || return 1
  model_url="https://huggingface.co/${MODEL_REPO}/resolve/${MODEL_REVISION:-main}/${LLAMA_FILE}"
  LLAMA_PROC='collabosm-llama/llama.*/llama-server'
  local stream_args=()
  if [ ! -f "$MODEL_DIR/$LLAMA_FILE" ]; then
    stream_args=(
      --model-url "$model_url"
      --hf-stream-vram
      --hf-download-threads "${LLAMA_HF_THREADS:-16}"
      --hf-stream-chunk-mb "${LLAMA_HF_CHUNK_MB:-8}"
      --hf-stream-disk-mode "${LLAMA_HF_DISK_MODE:-background}"
    )
  fi
  nohup "$bin" \
    --model "$MODEL_DIR/$LLAMA_FILE" \
    "${stream_args[@]}" \
    --alias "${LLAMA_ALIASES:-$MODEL_ID,qwen-3.8-27b,Qwen3.8-27B}" \
    --load-mode "${LLAMA_LOAD_MODE:-none}" \
    --lazy-mode "${LLAMA_LAZY_MODE:-auto}" \
    --parallel 1 \
    --device CUDA0 \
    --flash-attn on \
    --fit off \
    --ctx-checkpoints 1 \
    --metrics \
    --slots \
    --cache-prompt \
    --no-context-shift \
    --gpu-layers "${LLAMA_GPU_LAYERS:-99}" \
    --cache-type-k "$LLAMA_KV_TYPE" \
    --cache-type-v "$LLAMA_KV_TYPE" \
    --spec-draft-type-k "${LLAMA_SPEC_KV_TYPE:-$LLAMA_KV_TYPE}" \
    --spec-draft-type-v "${LLAMA_SPEC_KV_TYPE:-$LLAMA_KV_TYPE}" \
    --cache-ram "${LLAMA_CACHE_RAM:-8192}" \
    --slot-save-auto \
    --slot-save-max-count "${LLAMA_SLOT_SAVE_MAX_COUNT:-64}" \
    --slot-save-max-mb "${LLAMA_SLOT_SAVE_MAX_MB:-81920}" \
    --slot-save-path /content/collabosm-llama/slot-cache/ \
    --slot-save-block 256 \
    --slot-save-split-prefix \
    --native-snapshot-auto-save \
    --native-snapshot-auto-restore \
    --adaptive-direct-pure-disk-restore \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.05 \
    --reasoning-effort low \
    --kvarn-window-chunk 4096 \
    --cuda-graph-recovery-headroom-mb 18 \
    --ovf-packed-view \
    --ovf-cont-kv \
    --ovf-tail-local-cuda \
    --ovf-qwht-cuda \
    --no-vk-get-rows-q4-0-f16vec \
    --no-warmup \
    --spec-type "${LLAMA_SPEC_TYPE:-draft-mtp}" \
    --spec-dm-controller off \
    --spec-draft-p-min "$LLAMA_MTP_P_MIN" \
    --spec-draft-p-min-s "$LLAMA_MTP_P_MIN" \
    --spec-draft-p-min-m "$LLAMA_MTP_P_MIN" \
    --ctx-size-s "$TIER_S_CTX" \
    --mtp-s-max-tokens "$TIER_S_CTX" \
    --batch-size-s "$LLAMA_BATCH" \
    --ubatch-size-s "$LLAMA_UBATCH" \
    --cache-type-k-s "$LLAMA_KV_TYPE" \
    --cache-type-v-s "$LLAMA_KV_TYPE" \
    --spec-draft-type-k-s "${LLAMA_SPEC_KV_TYPE:-$LLAMA_KV_TYPE}" \
    --spec-draft-type-v-s "${LLAMA_SPEC_KV_TYPE:-$LLAMA_KV_TYPE}" \
    --spec-draft-n-max-s "$LLAMA_MTP_DRAFT" \
    --ctx-size-m "$TIER_M_CTX" \
    --mtp-m-max-tokens "$TIER_M_CTX" \
    --batch-size-m "$LLAMA_BATCH" \
    --ubatch-size-m "$LLAMA_UBATCH" \
    --cache-type-k-m "$LLAMA_KV_TYPE" \
    --cache-type-v-m "$LLAMA_KV_TYPE" \
    --spec-draft-type-k-m "${LLAMA_SPEC_KV_TYPE:-$LLAMA_KV_TYPE}" \
    --spec-draft-type-v-m "${LLAMA_SPEC_KV_TYPE:-$LLAMA_KV_TYPE}" \
    --spec-draft-n-max-m "$LLAMA_MTP_DRAFT" \
    --ctx-size-l "$LLAMA_CONTEXT" \
    --batch-size-l "$LLAMA_BATCH" \
    --ubatch-size-l "$LLAMA_UBATCH" \
    --cache-type-k-l "$LLAMA_KV_TYPE" \
    --cache-type-v-l "$LLAMA_KV_TYPE" \
    --spec-draft-type-k-l "${LLAMA_SPEC_KV_TYPE:-$LLAMA_KV_TYPE}" \
    --spec-draft-type-v-l "${LLAMA_SPEC_KV_TYPE:-$LLAMA_KV_TYPE}" \
    --spec-draft-n-max-l "$LLAMA_MTP_DRAFT_L" \
    --ctx-size-xl 0 \
    --ctx-size-xxl 0 \
    --ctx-size-xxxl 0 \
    --host 127.0.0.1 \
    --port "$PORT" \
    --api-key-file "$KEY_FILE" > "$LOG" 2>&1 < /dev/null &
}

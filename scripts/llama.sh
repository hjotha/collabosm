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
import concurrent.futures, os, urllib.request
from huggingface_hub import hf_hub_download, hf_hub_url

repo = os.environ["MODEL_REPO"]
rev = os.environ["MODEL_REVISION"]
fname = os.environ["LLAMA_FILE"]
dest_dir = os.environ["MODEL_DIR"]
dest = os.path.join(dest_dir, fname)

def parallel_download() -> bool:
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

try:
    if not parallel_download():
        raise RuntimeError("range unsupported")
except Exception as e:
    print(f"[llama] parallel download fallback ({e}), using hf_hub_download", flush=True)
    hf_hub_download(repo_id=repo, revision=rev, filename=fname, local_dir=dest_dir)
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
no-warmup = true
slot-save-auto = true
slot-save-block = 256
reasoning-effort = low

spec-type = draft-mtp
spec-dm-controller = off
spec-draft-p-min = 0.65
spec-draft-p-min-s = 0.65
spec-draft-p-min-m = 0.65

; Tier S: 0 - 150528 (b2048/ub256 N=2, p_min=0.65; ~126 MiB free VRAM)
ctx-size-s = ${TIER_S_CTX:-150528}
mtp-s-max-tokens = ${TIER_S_CTX:-150528}
batch-size-s = 2048
ubatch-size-s = 256
cache-type-k-s = kvarn4
cache-type-v-s = kvarn4
spec-draft-type-k-s = kvarn4
spec-draft-type-v-s = kvarn4
spec-draft-n-max-s = 2

; Tier M: aligned with Tier S (no intermediate step before Tier L)
ctx-size-m = ${TIER_M_CTX:-150528}
mtp-m-max-tokens = ${TIER_M_CTX:-150528}
batch-size-m = 2048
ubatch-size-m = 256
cache-type-k-m = kvarn4
cache-type-v-m = kvarn4
spec-draft-type-k-m = kvarn4
spec-draft-type-v-m = kvarn4
spec-draft-n-max-m = 2

; Tier L: 150529 - ${LLAMA_CONTEXT:-200704} (b2048/ub256 N=0, local CUDA; ~126-130 MiB free VRAM)
ctx-size-l = ${LLAMA_CONTEXT:-200704}
batch-size-l = 2048
ubatch-size-l = 256
cache-type-k-l = kvarn4
cache-type-v-l = kvarn4
spec-draft-type-k-l = kvarn4
spec-draft-type-v-l = kvarn4
spec-draft-n-max-l = 0

ctx-size-xl = 0
ctx-size-xxl = 0
ctx-size-xxxl = 0
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

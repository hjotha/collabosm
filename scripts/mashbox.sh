#!/usr/bin/env bash
# The pinned MASHBOX distribution verifies its catalog's weights and kernels.
MASHBOX_VENV=/content/collabosm-mashbox
export XDG_CACHE_HOME="${MODEL_DIR}"

mashbox_install() {
  status "stage=runtime"
  python3 -m pip install uv &&
  python3 -m uv venv --allow-existing --python 3.12 "$MASHBOX_VENV" &&
  python3 -m uv pip install --python "$MASHBOX_VENV/bin/python" "mashbox[tpu]==${MASHBOX_VERSION}" &&
  JAX_PLATFORMS=tpu "$MASHBOX_VENV/bin/python" - <<'PY' || return 1
import jax

devices = jax.devices("tpu")
print("JAX TPU devices:", devices, [device.device_kind for device in devices], flush=True)
if len(devices) != 1 or "v6" not in devices[0].device_kind.lower():
    raise SystemExit("MASHBOX recipe requires a single TPU v6e chip")
PY
}

mashbox_start() {
  nohup env JAX_PLATFORMS=tpu "$MASHBOX_VENV/bin/mashbox" run "$MODEL_ID" \
    --host 127.0.0.1 --port "$PORT" --api-key-file "$KEY_FILE" \
    --ctx "$MASHBOX_CONTEXT" --slots 1 --tensor-parallel-size 1 \
    > "$LOG" 2>&1 &
}

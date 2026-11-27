#!/usr/bin/env bash
# Installed by bootstrap.sh on a CUDA Colab runtime.
set -euo pipefail

media_install() {
  local venv=/content/collabosm-media
  echo "stage=runtime" > /content/STATUS
  if [[ "${MEDIA_PIPELINE:?}" == pika ]]; then
    venv=/content/collabosm-pika-venv
  fi
  local py="$venv/bin/python"
  if [[ "$MEDIA_PIPELINE" == pika ]]; then
    local free_bytes
    free_bytes=$(df -PB1 /content | awk 'NR == 2 {print $4}')
    if (( free_bytes < 225000000000 )) && [[ ! -x "$venv/bin/pika-h3-pro" ]]; then
      echo "Pika H3 Flex requires at least 225 GB of free disk before installation."
      exit 1
    fi
    python3 -m pip install --no-cache-dir uv
    python3 -m uv venv --allow-existing --python 3.12 --seed "$venv"
    local src=/content/collabosm-pika
    [[ -d "$src/.git" ]] || git clone https://github.com/Pika-Labs/Pika-H3-Flex.git "$src"
    git -C "$src" checkout --detach f961d7ed09a5932b7533559211b7d233e8f1c8e8
    cd "$src"
    "$py" -m pip install --no-cache-dir -r requirements.txt
    "$py" tools/apply_diffusers_patches.py
    "$py" -m pip install --no-deps -e .
  else
    # Reuse Colab's CUDA PyTorch and let its pip manage the venv without ensurepip.
    python3 -m venv --without-pip --system-site-packages "$venv"
    "$py" -c 'import torch; assert torch.cuda.is_available(), "Colab CUDA PyTorch is unavailable"'
    python3 -m pip --python "$py" install --no-cache-dir 'diffusers==0.38.0' 'transformers==4.57.6' \
      'accelerate==1.12.0' 'bitsandbytes==0.49.2' 'huggingface-hub==0.36.2' sentencepiece imageio-ffmpeg
    echo "stage=weights" > /content/STATUS
    "$py" /content/media_server.py --download
  fi
}

media_install

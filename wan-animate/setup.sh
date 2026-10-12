#!/usr/bin/env bash
# One-time setup on a GPU machine: clones Wan2.2, installs deps, downloads Animate-14B weights.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v nvidia-smi >/dev/null; then
  echo "No NVIDIA GPU found. Wan2.2-Animate-14B needs a CUDA GPU (80 GB A100/H100 recommended)." >&2
  exit 1
fi

[ -d Wan2.2 ] || git clone https://github.com/Wan-Video/Wan2.2.git

pip install -U pip
pip install "torch>=2.4.0" torchvision
# flash_attn often fails to build first; install everything else, then try it last.
grep -v flash_attn Wan2.2/requirements.txt > /tmp/wan_reqs.txt
pip install -r /tmp/wan_reqs.txt
# Animate preprocessing has its own requirements file in newer checkouts.
[ -f Wan2.2/requirements_animate.txt ] && pip install -r Wan2.2/requirements_animate.txt
pip install flash_attn --no-build-isolation || echo "flash_attn failed to install; continuing without it."

pip install "huggingface_hub[cli]"
if [ ! -d Wan2.2-Animate-14B ]; then
  if ! huggingface-cli download Wan-AI/Wan2.2-Animate-14B --local-dir ./Wan2.2-Animate-14B; then
    echo "Hugging Face download failed, trying ModelScope..." >&2
    pip install modelscope
    modelscope download Wan-AI/Wan2.2-Animate-14B --local_dir ./Wan2.2-Animate-14B
  fi
fi

echo "Setup done. Next: ./spawn_character.sh --mode replace --video in.mp4 --image character.png"

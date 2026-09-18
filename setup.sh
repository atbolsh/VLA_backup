#!/usr/bin/env bash
# Install only. Does not start Gradio. Run on the rented GPU box.
set -euo pipefail

echo "============================================================"
echo " robo-env setup (EO-1 + LIBERO)"
echo " Installs the venv, weights, and LIBERO assets."
echo " After this finishes: bash launch.sh"
echo "============================================================"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

_try() { echo "+ $*"; "$@"; }

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "WARNING: nvidia-smi not found. EO-1 needs a CUDA GPU." >&2
else
  echo "GPU:"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
fi

if [[ -f "$HERE/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$HERE/.env"
  set +a
fi

PY=""
for c in python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1; then
    PY="$c"
    break
  fi
done
[[ -n "$PY" ]] || { echo "Need python3.10+" >&2; exit 1; }
echo "Using $($PY --version)"

[[ -d .venv ]] || "$PY" -m venv .venv
VENV_PY="${HERE}/.venv/bin/python"
# shellcheck disable=SC1091
source .venv/bin/activate
export PATH="${HERE}/.venv/bin:${PATH}"
export PYTHONNOUSERSITE=1
hash -r 2>/dev/null || true
"$VENV_PY" -m pip install -U pip wheel

CU128="https://download.pytorch.org/whl/cu128"
CU126="https://download.pytorch.org/whl/cu126"
TORCH_RUNG=""

if _try "$VENV_PY" -m pip install "torch==2.7.0" "torchvision==0.22.0" --index-url "$CU128"; then
  TORCH_RUNG="2.7.0+cu128"
elif _try "$VENV_PY" -m pip install "torch==2.8.0" --index-url "$CU128"; then
  _try "$VENV_PY" -m pip install torchvision --index-url "$CU128"
  TORCH_RUNG="2.8.0+cu128"
elif [[ "${FORCE_SETUP:-}" == "1" ]] && _try "$VENV_PY" -m pip install "torch==2.7.0" "torchvision==0.22.0" --index-url "$CU126"; then
  TORCH_RUNG="2.7.0+cu126"
else
  echo "Could not install a CUDA torch wheel. Set FORCE_SETUP=1 to try cu126." >&2
  exit 1
fi

_try "$VENV_PY" -m pip install "transformers>=4.49,<4.58"
if ! _try "$VENV_PY" -m pip install -r requirements.txt; then
  echo "requirements.txt failed; installing a lean set, then lerobot --no-deps."
  _try "$VENV_PY" -m pip install accelerate safetensors "huggingface-hub>=0.34.2,<1.0" pillow numpy einops \
    sentencepiece protobuf qwen-vl-utils python-dotenv requests pyyaml \
    "transformers>=4.49,<4.58" "datasets>=2.19.0,<=3.6.0" "diffusers>=0.27.2,<0.39" \
    "opencv-python-headless>=4.9.0" "av>=14.2.0" "draccus==0.10.0" \
    gymnasium transforms3d "gradio>=4.44,<6" imageio imageio-ffmpeg tqdm tyro
  _try "$VENV_PY" -m pip install "lerobot==0.3.3" --no-deps
fi

_try "$VENV_PY" -m pip install "huggingface-hub>=0.34.2,<1.0" "diffusers>=0.27.2,<0.39"

if [[ "$TORCH_RUNG" == "2.7.0+cu128" ]]; then
  _try "$VENV_PY" -m pip install "torch==2.7.0" "torchvision==0.22.0" --index-url "$CU128"
elif [[ "$TORCH_RUNG" == "2.8.0+cu128" ]]; then
  _try "$VENV_PY" -m pip install "torch==2.8.0" --index-url "$CU128"
  _try "$VENV_PY" -m pip install torchvision --index-url "$CU128"
elif [[ "$TORCH_RUNG" == "2.7.0+cu126" ]]; then
  _try "$VENV_PY" -m pip install "torch==2.7.0" "torchvision==0.22.0" --index-url "$CU126"
fi

_try "$VENV_PY" -m pip install -e "$HERE"

# LIBERO: do not install their requirements.txt (ancient torch/transformers).
_try "$VENV_PY" -m pip install "hydra-core>=1.2" easydict "bddl==1.0.1" future cloudpickle \
  "gym>=0.25,<0.27" matplotlib "robosuite==1.4.1" "mujoco==3.3.2"

if [[ ! -d vendor/LIBERO/.git ]]; then
  mkdir -p vendor
  _try git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git vendor/LIBERO
fi
_try "$VENV_PY" -m pip install -e vendor/LIBERO --no-deps
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${HERE}/.libero}"
_try "$VENV_PY" scripts/write_libero_home_config.py

"$VENV_PY" - <<'PY'
import torch, sys
print("torch", torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
try:
    import libero
    from libero.libero import get_libero_path
    print("libero bddl", get_libero_path("bddl_files"))
except Exception as exc:
    print("LIBERO import warning:", exc, file=sys.stderr)
PY

mkdir -p vendor/eo1_libero weights logs scenes
EO1=https://raw.githubusercontent.com/SHAILAB-IPEC/EO1/main/experiments/2_libero
for f in eval_libero.py data-libero.yaml README.md; do
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL -o "vendor/eo1_libero/${f}.tmp" "${EO1}/${f}" && mv "vendor/eo1_libero/${f}.tmp" "vendor/eo1_libero/${f}" \
      || echo "WARNING: could not refresh vendor/eo1_libero/${f}"
  fi
done

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

"$VENV_PY" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("IPEC-COMMUNITY/EO-1-3B", local_dir="weights/EO-1-3B", local_dir_use_symlinks=False)
print("downloaded IPEC-COMMUNITY/EO-1-3B")
PY

"$VENV_PY" scripts/build_libero_robot_config.py || echo "WARNING: robot_config builder failed; using fallback JSON"

{
  echo "torch=$TORCH_RUNG"
  echo "lerobot=0.3.3"
} > .rung
echo "============================================================"
echo " setup done. Rung:"
cat .rung
echo
echo "Start the playground with:  bash launch.sh"
echo "Scene editor only:          bash launch.sh --scene"
echo "============================================================"

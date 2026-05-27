#!/bin/bash
set -e

REPO_DIR="/workspace/anima-trainer"
MODELS_DIR="$REPO_DIR/models/anima"
PYTHON="/usr/bin/python"
PIP="$PYTHON -m pip"

# ---------------------------------------------------------------------------
# 1. Clone or update repo (with submodules)
# ---------------------------------------------------------------------------
if [ ! -d "$REPO_DIR" ]; then
    echo ">>> Cloning repo..."
    git clone --recurse-submodules https://github.com/tasdelenmuratpdr-hue/anima_trainer "$REPO_DIR"
else
    echo ">>> Updating repo..."
    cd "$REPO_DIR"
    git pull
    git submodule update --init --recursive
fi

cd "$REPO_DIR"

# ---------------------------------------------------------------------------
# 2. Install packages (always use /usr/bin/python)
# ---------------------------------------------------------------------------
echo ">>> Installing sd-scripts requirements..."
cd sd-scripts
$PIP install -r requirements.txt -q --root-user-action=ignore
cd ..

echo ">>> Installing app requirements..."
$PIP install "gradio>=4.0.0,<6.0.0" toml accelerate -q --root-user-action=ignore

# ---------------------------------------------------------------------------
# 3. Download models if missing
# ---------------------------------------------------------------------------
mkdir -p "$MODELS_DIR/dit" "$MODELS_DIR/text_encoder" "$MODELS_DIR/vae"

if [ ! -f "$MODELS_DIR/text_encoder/qwen_3_06b_base.safetensors" ]; then
    echo ">>> Downloading Qwen3 text encoder (1.19 GB)..."
    wget -q --show-progress -O "$MODELS_DIR/text_encoder/qwen_3_06b_base.safetensors" \
        "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/text_encoders/qwen_3_06b_base.safetensors"
fi

if [ ! -f "$MODELS_DIR/vae/qwen_image_vae.safetensors" ]; then
    echo ">>> Downloading VAE (254 MB)..."
    wget -q --show-progress -O "$MODELS_DIR/vae/qwen_image_vae.safetensors" \
        "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/vae/qwen_image_vae.safetensors"
fi

# ---------------------------------------------------------------------------
# 4. Launch app
# ---------------------------------------------------------------------------
echo ">>> Starting Anima LoRA Trainer on port 7860..."
export GRADIO_SERVER_NAME=0.0.0.0
export PYTHONIOENCODING=utf-8
$PYTHON app.py

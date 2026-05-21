#!/bin/bash
set -e

REPO_DIR="/workspace/anima-trainer"
MODELS_DIR="$REPO_DIR/models/anima"
STAMP="$REPO_DIR/.deps_installed"

# Clone or update repo
if [ ! -d "$REPO_DIR" ]; then
    echo ">>> Cloning repo..."
    git clone https://github.com/tasdelenmuratpdr-hue/anima_trainer "$REPO_DIR"
else
    echo ">>> Updating repo..."
    cd "$REPO_DIR" && git pull
fi

cd "$REPO_DIR"

# Install dependencies only if not already done
if [ ! -f "$STAMP" ]; then
    echo ">>> Installing sd-scripts requirements..."
    cd sd-scripts && pip install -r requirements.txt -q && cd ..
    pip install "gradio>=4.0.0,<6.0.0" toml -q
    touch "$STAMP"
    echo ">>> Dependencies installed."
else
    echo ">>> Dependencies already installed, skipping."
fi

# Create model directories
mkdir -p "$MODELS_DIR/dit" "$MODELS_DIR/text_encoder" "$MODELS_DIR/vae"

# Download text encoder if missing
if [ ! -f "$MODELS_DIR/text_encoder/qwen_3_06b_base.safetensors" ]; then
    echo ">>> Downloading Qwen3 text encoder (1.19 GB)..."
    wget -q --show-progress -O "$MODELS_DIR/text_encoder/qwen_3_06b_base.safetensors" \
        "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/text_encoders/qwen_3_06b_base.safetensors"
fi

# Download VAE if missing
if [ ! -f "$MODELS_DIR/vae/qwen_image_vae.safetensors" ]; then
    echo ">>> Downloading VAE (254 MB)..."
    wget -q --show-progress -O "$MODELS_DIR/vae/qwen_image_vae.safetensors" \
        "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/vae/qwen_image_vae.safetensors"
fi

# Note: DiT models are selected and downloaded via the UI (Training tab).
# To pre-download a model, uncomment and edit one of the lines below:
#
# wget -q --show-progress -O "$MODELS_DIR/dit/anima-base-v1.0.safetensors" \
#     "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/diffusion_models/anima-base-v1.0.safetensors"
#
# For a custom model (e.g. novaanimeanima.safetensors), upload it to:
#   $MODELS_DIR/dit/novaanimeanima.safetensors
# and it will appear automatically in the Base Model dropdown.

echo ">>> Starting Anima LoRA Trainer on port 7860..."
export GRADIO_SERVER_NAME=0.0.0.0
export PYTHONIOENCODING=utf-8
python app.py

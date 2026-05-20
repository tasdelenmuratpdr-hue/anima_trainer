#!/bin/bash
cd /workspace/anima-trainer

# Download text encoder if not in network volume
if [ ! -f "models/anima/text_encoder/qwen_3_06b_base.safetensors" ]; then
    echo "Downloading Qwen3 text encoder (1.19 GB)..."
    wget -c --show-progress \
        -O models/anima/text_encoder/qwen_3_06b_base.safetensors \
        "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/text_encoders/qwen_3_06b_base.safetensors"
fi

# Download VAE if not in network volume
if [ ! -f "models/anima/vae/qwen_image_vae.safetensors" ]; then
    echo "Downloading VAE (254 MB)..."
    wget -c --show-progress \
        -O models/anima/vae/qwen_image_vae.safetensors \
        "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/vae/qwen_image_vae.safetensors"
fi

echo "Starting Anima LoRA Trainer..."
export GRADIO_SERVER_NAME=0.0.0.0
python app.py

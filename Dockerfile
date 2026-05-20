FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y git curl wget && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/anima-trainer

COPY . .

RUN cd sd-scripts && pip install -r requirements.txt && cd ..
RUN pip install "gradio>=4.0.0,<6.0.0" toml
RUN accelerate config default

RUN mkdir -p models/anima/dit models/anima/text_encoder models/anima/vae configs logs

EXPOSE 7860

COPY runpod_start.sh /runpod_start.sh
RUN chmod +x /runpod_start.sh
CMD ["/runpod_start.sh"]

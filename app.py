"""
Anima LoRA Trainer — Local / RunPod Gradio UI
"""

import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

import gradio as gr
import toml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
CONFIG_FILE = ROOT / "config.json"
CONFIGS_DIR = ROOT / "configs"
LOGS_DIR = ROOT / "logs"
MODELS_DIR = ROOT / "models" / "anima"
DATASETS_DIR = ROOT / "datasets"
SD_SCRIPTS_DIR = ROOT / "sd-scripts"
QUEUE_FILE = ROOT / "queue.json"

QWEN3_MODEL = MODELS_DIR / "text_encoder" / "qwen_3_06b_base.safetensors"
VAE_MODEL = MODELS_DIR / "vae" / "qwen_image_vae.safetensors"
TRAIN_SCRIPT = SD_SCRIPTS_DIR / "anima_train_network.py"

ACCELERATE_CONFIG = "app_configs/accelerate_gpu.yaml"

BASE_MODEL_URLS = {
    "anima-base-v1.0": "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/diffusion_models/anima-base-v1.0.safetensors",
    "anima-preview3-base": "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/diffusion_models/anima-preview3-base.safetensors",
    "anima-preview": "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/diffusion_models/anima-preview.safetensors",
}

for _d in [CONFIGS_DIR, LOGS_DIR, DATASETS_DIR, MODELS_DIR / "dit",
           MODELS_DIR / "text_encoder", MODELS_DIR / "vae"]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model auto-detection
# ---------------------------------------------------------------------------

def get_available_dit_models() -> list[str]:
    """Scan models/anima/dit/ for any .safetensors files."""
    dit_dir = MODELS_DIR / "dit"
    found = sorted(f.name for f in dit_dir.glob("*.safetensors"))
    # Also include known remote models by name (not filename) if not downloaded
    known = list(BASE_MODEL_URLS.keys())
    for k in known:
        fname = k + ".safetensors"
        if fname not in found:
            found.append(k + "  [not downloaded]")
    return found if found else ["anima-base-v1.0  [not downloaded]"]


def resolve_dit_model(choice: str) -> Path:
    """Given a dropdown choice, return the local path for the DiT model."""
    name = choice.replace("  [not downloaded]", "").strip()
    # If it's a known remote model name (no extension), map to filename
    if not name.endswith(".safetensors"):
        name = name + ".safetensors"
    return MODELS_DIR / "dit" / name


def get_base_model_url(choice: str) -> str | None:
    """Return download URL for known models, None for custom files."""
    name = choice.replace("  [not downloaded]", "").strip()
    return BASE_MODEL_URLS.get(name)


# ---------------------------------------------------------------------------
# Accelerate detection
# ---------------------------------------------------------------------------

def get_accelerate_cmd() -> list[str]:
    acc = shutil.which("accelerate")
    if acc:
        return [acc, "launch"]
    return [sys.executable, "-m", "accelerate", "launch"]


# ---------------------------------------------------------------------------
# Defaults & config persistence
# ---------------------------------------------------------------------------

DEFAULTS = {
    "project_name": "my_lora",
    "base_model": "anima-base-v1.0",
    "image_directory": "",
    "output_directory": "",
    "network_dim": 20,
    "network_alpha": 20,
    "learning_rate": 0.0001,
    "max_train_epochs": 10,
    "resolution": 768,
    "repeats": 10,
    "caption_dropout": 0.1,
    "gpu_index": "0",
    "optimizer_type": "AdamW8bit",
    "lr_scheduler": "cosine_with_restarts",
    "lr_scheduler_num_cycles": 1,
    "lr_warmup_steps": 100,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 1.0,
    "save_every_n_epochs": 1,
    "save_last_n_epochs": 4,
    "mixed_precision": "bf16",
    "gradient_checkpointing": True,
    "seed": 42,
    "noise_offset": 0.03,
    "multires_noise_discount": 0.3,
    "timestep_sampling": "sigmoid",
    "discrete_flow_shift": 1.0,
    "cache_latents": True,
    "cache_text_encoder_outputs": True,
    "vae_chunk_size": 64,
    "vae_disable_cache": True,
    "num_cpu_threads_per_process": 1,
    "last_train_config": "",
    "last_dataset_config": "",
}


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                saved = json.load(f)
            cfg.update({k: v for k, v in saved.items() if k in DEFAULTS})
        except Exception:
            pass
    return cfg


def save_config(cfg: dict):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def detect_gpus() -> list[str]:
    try:
        import torch
        if not torch.cuda.is_available():
            return ["CPU (no CUDA detected)"]
        choices = [f"{i}: {torch.cuda.get_device_name(i)}" for i in range(torch.cuda.device_count())]
        return choices if choices else ["0", "1"]
    except ImportError:
        return ["0", "1"]


GPU_CHOICES = detect_gpus()


def gpu_index_from_choice(choice: str) -> str:
    if not choice:
        return "0"
    return str(choice).split(":")[0].strip()


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def validate_dataset(image_dir: str) -> tuple[int, list[str], list[str]]:
    p = Path(image_dir)
    if not p.exists():
        raise FileNotFoundError(f"Directory not found: {image_dir}")
    if not p.is_dir():
        raise NotADirectoryError(f"Not a directory: {image_dir}")
    all_files = list(p.iterdir())
    image_files = [f for f in all_files if f.suffix.lower() in IMAGE_EXTS and f.is_file()]
    txt_basenames = {f.stem for f in all_files if f.suffix.lower() == ".txt" and f.is_file()}
    missing = [f.name for f in image_files if f.stem not in txt_basenames]
    warnings = []
    if not image_files:
        warnings.append("No image files found in directory.")
    if missing:
        warnings.append(f"{len(missing)} image(s) are missing caption (.txt) files.")
    return len(image_files), missing, warnings


# ---------------------------------------------------------------------------
# TOML config generation
# ---------------------------------------------------------------------------

def create_training_config(
    project_name, output_dir, dit_model_path, qwen3_model_path, vae_model_path,
    network_dim=20, network_alpha=20, learning_rate=1e-4, max_train_epochs=10,
    optimizer_type="AdamW8bit", lr_scheduler="cosine_with_restarts",
    lr_scheduler_num_cycles=1, lr_warmup_steps=100,
    train_batch_size=1, gradient_accumulation_steps=1, max_grad_norm=1.0,
    save_every_n_epochs=1, save_last_n_epochs=4,
    mixed_precision="bf16", gradient_checkpointing=True,
    seed=42, noise_offset=0.03, multires_noise_discount=0.3,
    timestep_sampling="sigmoid", discrete_flow_shift=1.0,
    cache_latents=True, cache_text_encoder_outputs=True,
    vae_chunk_size=64, vae_disable_cache=True,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    config_path = CONFIGS_DIR / f"{project_name}_training_{ts}.toml"
    training_config = {
        "pretrained_model_name_or_path": str(dit_model_path),
        "qwen3": str(qwen3_model_path),
        "vae": str(vae_model_path),
        "network_module": "networks.lora_anima",
        "network_dim": int(network_dim),
        "network_alpha": int(network_alpha),
        "network_train_unet_only": True,
        "learning_rate": float(learning_rate),
        "optimizer_type": optimizer_type,
        "optimizer_args": ["weight_decay=0.1", "betas=[0.9, 0.99]"],
        "lr_scheduler": lr_scheduler,
        "lr_scheduler_num_cycles": int(lr_scheduler_num_cycles),
        "lr_warmup_steps": int(lr_warmup_steps),
        "max_train_epochs": int(max_train_epochs),
        "train_batch_size": int(train_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "max_grad_norm": float(max_grad_norm),
        "seed": int(seed),
        "timestep_sampling": timestep_sampling,
        "discrete_flow_shift": float(discrete_flow_shift),
        "qwen3_max_token_length": 512,
        "t5_max_token_length": 512,
        "mixed_precision": mixed_precision,
        "gradient_checkpointing": bool(gradient_checkpointing),
        "cache_latents": bool(cache_latents),
        "cache_text_encoder_outputs": bool(cache_text_encoder_outputs),
        "vae_chunk_size": int(vae_chunk_size),
        "vae_disable_cache": bool(vae_disable_cache),
        "output_dir": str(output_dir),
        "output_name": project_name,
        "save_model_as": "safetensors",
        "save_precision": "bf16",
        "save_every_n_epochs": int(save_every_n_epochs),
        "save_last_n_epochs": int(save_last_n_epochs),
        "shuffle_caption": False,
        "caption_extension": ".txt",
        "noise_offset": float(noise_offset),
        "multires_noise_discount": float(multires_noise_discount),
        "training_comment": f"Anima LoRA - {datetime.now().strftime('%Y-%m-%d')}",
    }
    with open(config_path, "w") as f:
        toml.dump(training_config, f)
    return str(config_path)


def create_dataset_config(
    project_name, image_dir, resolution=768, repeats=5, caption_dropout_rate=0.1
) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    config_path = CONFIGS_DIR / f"{project_name}_dataset_{ts}.toml"
    dataset_config = {
        "general": {
            "resolution": int(resolution),
            "enable_bucket": True,
            "bucket_no_upscale": False,
            "bucket_reso_steps": 64,
            "min_bucket_reso": 256,
            "max_bucket_reso": 4096,
        },
        "datasets": [{
            "resolution": int(resolution),
            "subsets": [{
                "num_repeats": int(repeats),
                "image_dir": str(image_dir),
                "caption_extension": ".txt",
                "caption_dropout_rate": float(caption_dropout_rate),
            }],
        }],
    }
    with open(config_path, "w") as f:
        toml.dump(dataset_config, f)
    return str(config_path)


# ---------------------------------------------------------------------------
# Configure Training handler
# ---------------------------------------------------------------------------

def configure_training(
    project_name, base_model, image_directory, output_directory,
    network_dim, network_alpha, learning_rate, max_train_epochs,
    resolution, repeats, caption_dropout, gpu_index_choice,
    optimizer_type, lr_scheduler, lr_scheduler_num_cycles, lr_warmup_steps,
    train_batch_size, gradient_accumulation_steps, max_grad_norm,
    save_every_n_epochs, save_last_n_epochs, mixed_precision,
    gradient_checkpointing, seed, noise_offset, multires_noise_discount,
    timestep_sampling, discrete_flow_shift,
    cache_latents, cache_text_encoder_outputs, vae_chunk_size, vae_disable_cache,
    num_cpu_threads_per_process,
) -> tuple[str, str, str]:
    lines = []
    if not project_name.strip():
        return "❌ Project name cannot be empty.", "", ""
    if not image_directory.strip():
        return "❌ Image directory cannot be empty.", "", ""
    if not output_directory.strip():
        return "❌ Output directory cannot be empty.", "", ""

    lines.append(f"Project:          {project_name}")
    lines.append(f"Image directory:  {image_directory}")
    lines.append(f"Output directory: {output_directory}")
    lines.append("")

    try:
        n_images, missing, warnings = validate_dataset(image_directory)
    except (FileNotFoundError, NotADirectoryError) as e:
        return f"❌ {e}", "", ""

    lines.append(f"Images found:     {n_images}")
    if missing:
        lines.append(f"⚠ Missing captions ({len(missing)}):")
        for m in missing[:20]:
            lines.append(f"    • {m}")
        if len(missing) > 20:
            lines.append(f"    … and {len(missing) - 20} more")
    else:
        lines.append("✓ All images have caption files.")
    for w in warnings:
        lines.append(f"⚠ {w}")
    if n_images == 0:
        lines.append("\n❌ Cannot configure — no images found.")
        return "\n".join(lines), "", ""

    batch = max(int(train_batch_size), 1)
    grad = max(int(gradient_accumulation_steps), 1)
    spe = math.ceil((n_images * int(repeats)) / (batch * grad))
    total = spe * int(max_train_epochs)
    lines += ["", "── Step Estimate ─────────────────────────────────────",
              f"  Steps per epoch: {spe}  ({n_images} imgs × {repeats} repeats)",
              f"  Total steps:     {total}  ({spe} × {max_train_epochs} epochs)",
              "──────────────────────────────────────────────────────"]

    lines.append("\nChecking models...")
    dit_model = resolve_dit_model(base_model)
    missing_models = []
    for label, path in [("DiT", dit_model), ("Qwen3", QWEN3_MODEL), ("VAE", VAE_MODEL)]:
        if Path(path).exists():
            lines.append(f"  ✓ {label}: {path}")
        else:
            if label == "DiT":
                lines.append(f"  ℹ {label}: {path}")
                lines.append(f"      (will auto-download when training starts)")
            else:
                lines.append(f"  ✗ {label} missing: {path}")
                missing_models.append(label)
    if missing_models:
        lines.append(f"\n❌ Missing models: {', '.join(missing_models)}")
        lines.append("Run setup_for_linux.sh / setup_for_windows.bat to download them.")
        return "\n".join(lines), "", ""

    lines.append("\nGenerating TOML configs...")
    try:
        train_cfg = create_training_config(
            project_name=project_name, output_dir=output_directory,
            dit_model_path=dit_model, qwen3_model_path=QWEN3_MODEL, vae_model_path=VAE_MODEL,
            network_dim=network_dim, network_alpha=network_alpha, learning_rate=learning_rate,
            max_train_epochs=max_train_epochs, optimizer_type=optimizer_type,
            lr_scheduler=lr_scheduler, lr_scheduler_num_cycles=lr_scheduler_num_cycles,
            lr_warmup_steps=lr_warmup_steps, train_batch_size=train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps, max_grad_norm=max_grad_norm,
            save_every_n_epochs=save_every_n_epochs, save_last_n_epochs=save_last_n_epochs,
            mixed_precision=mixed_precision, gradient_checkpointing=gradient_checkpointing,
            seed=seed, noise_offset=noise_offset, multires_noise_discount=multires_noise_discount,
            timestep_sampling=timestep_sampling, discrete_flow_shift=discrete_flow_shift,
            cache_latents=cache_latents, cache_text_encoder_outputs=cache_text_encoder_outputs,
            vae_chunk_size=vae_chunk_size, vae_disable_cache=vae_disable_cache,
        )
        dataset_cfg = create_dataset_config(
            project_name=project_name, image_dir=image_directory,
            resolution=resolution, repeats=repeats, caption_dropout_rate=caption_dropout,
        )
    except Exception as e:
        lines.append(f"❌ Failed to generate configs: {e}")
        return "\n".join(lines), "", ""

    lines.append(f"  ✓ Training config: {train_cfg}")
    lines.append(f"  ✓ Dataset  config: {dataset_cfg}")

    cfg = {
        "project_name": project_name, "base_model": base_model,
        "image_directory": image_directory, "output_directory": output_directory,
        "network_dim": int(network_dim), "network_alpha": int(network_alpha),
        "learning_rate": float(learning_rate), "max_train_epochs": int(max_train_epochs),
        "resolution": int(resolution), "repeats": int(repeats),
        "caption_dropout": float(caption_dropout),
        "gpu_index": gpu_index_from_choice(gpu_index_choice),
        "optimizer_type": optimizer_type, "lr_scheduler": lr_scheduler,
        "lr_scheduler_num_cycles": int(lr_scheduler_num_cycles),
        "lr_warmup_steps": int(lr_warmup_steps),
        "train_batch_size": int(train_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "max_grad_norm": float(max_grad_norm),
        "save_every_n_epochs": int(save_every_n_epochs),
        "save_last_n_epochs": int(save_last_n_epochs),
        "mixed_precision": mixed_precision, "gradient_checkpointing": bool(gradient_checkpointing),
        "seed": int(seed), "noise_offset": float(noise_offset),
        "multires_noise_discount": float(multires_noise_discount),
        "timestep_sampling": timestep_sampling, "discrete_flow_shift": float(discrete_flow_shift),
        "cache_latents": bool(cache_latents),
        "cache_text_encoder_outputs": bool(cache_text_encoder_outputs),
        "vae_chunk_size": int(vae_chunk_size), "vae_disable_cache": bool(vae_disable_cache),
        "num_cpu_threads_per_process": int(num_cpu_threads_per_process),
        "last_train_config": train_cfg, "last_dataset_config": dataset_cfg,
    }
    save_config(cfg)
    lines.append("\n✓ Configuration complete — ready to train.")
    return "\n".join(lines), train_cfg, dataset_cfg


# ---------------------------------------------------------------------------
# Training runner (streaming generator)
# ---------------------------------------------------------------------------

def _run_training_subprocess(train_cfg, dataset_cfg, gpu_idx, threads, base_model):
    """Core training runner — yields log lines. Used by both direct and queue runs."""
    log_lines: list[str] = []

    def emit(line: str):
        log_lines.append(line)
        return "\n".join(log_lines)

    dit_model = resolve_dit_model(base_model)
    if not dit_model.exists():
        url = get_base_model_url(base_model)
        if not url:
            yield emit(f"❌ Model file not found and no download URL known: {dit_model}")
            return
        yield emit(f"⏳ Downloading base model '{base_model}'...")
        yield emit(f"   Destination: {dit_model}")
        yield emit("")
        os.makedirs(dit_model.parent, exist_ok=True)
        try:
            dl_proc = subprocess.Popen(
                ["wget", "-c", "--show-progress", "-O", str(dit_model), url],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True, bufsize=1,
            )
            for line in iter(dl_proc.stdout.readline, ""):
                yield emit(line.rstrip("\n"))
            dl_proc.wait()
            if dl_proc.returncode != 0:
                yield emit(f"❌ Download failed (exit code {dl_proc.returncode})")
                return
            yield emit("✓ Base model downloaded.")
            yield emit("")
        except FileNotFoundError:
            yield emit("❌ 'wget' not found. Please download the model manually.")
            return

    if not TRAIN_SCRIPT.exists():
        yield emit(f"❌ Training script not found: {TRAIN_SCRIPT}")
        return

    acc_cmd = get_accelerate_cmd()
    cmd = acc_cmd + [
        "--config_file", str(ACCELERATE_CONFIG),
        "--num_cpu_threads_per_process", str(threads),
        "--gpu_ids", gpu_idx,
        str(TRAIN_SCRIPT),
        "--config_file", train_cfg,
        "--dataset_config", dataset_cfg,
    ]

    yield emit(f"Using GPU index: {gpu_idx}")
    yield emit(f"Command: {' '.join(shlex.quote(c) for c in cmd)}")
    yield emit("")
    yield emit("⏳ Launching training process...")

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    saved_cfg = load_config()
    project_name = saved_cfg.get("project_name", "run")
    log_file_path = LOGS_DIR / f"{project_name}_{ts}.log"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_idx
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, bufsize=1, env=env, cwd=str(ROOT),
            encoding="utf-8", errors="ignore",
        )
    except FileNotFoundError:
        yield emit("❌ 'accelerate' not found. Make sure the venv is activated.")
        return

    yield emit(f"✓ Process started (PID: {process.pid}). Loading models...")
    yield emit("")

    with open(log_file_path, "w", encoding="utf-8", errors="ignore") as log_f:
        log_f.write(f"Command: {' '.join(cmd)}\nStarted: {datetime.now().isoformat()}\n\n")
        for line in iter(process.stdout.readline, ""):
            line = line.rstrip("\n")
            log_f.write(line + "\n")
            log_f.flush()
            yield emit(line)

    exit_code = process.wait()
    if exit_code == 0:
        yield emit(f"\n✓ Training completed!\nLoRA saved to: {saved_cfg.get('output_directory', '?')}\nLog: {log_file_path}")
    else:
        yield emit(f"\n✗ Training failed (exit code: {exit_code})\nLog: {log_file_path}")
        try:
            result = subprocess.run(["dmesg", "-T"], capture_output=True, text=True, timeout=5)
            tail = "\n".join(result.stdout.splitlines()[-40:])
            if any(t in tail for t in ("Out of memory", "Killed process", "oom_reaper", "OOM")):
                yield emit("\n💡 OOM detected. Try: network_dim=8 and/or resolution=512")
        except Exception:
            pass


def start_training(custom_config_path: str, gpu_index_choice: str,
                   num_cpu_threads_per_process: int, base_model: str):
    saved_cfg = load_config()
    train_cfg = custom_config_path.strip() if custom_config_path.strip() else saved_cfg.get("last_train_config", "")
    dataset_cfg = saved_cfg.get("last_dataset_config", "")
    log_lines: list[str] = []

    def emit(line):
        log_lines.append(line)
        return "\n".join(log_lines)

    if not train_cfg:
        yield emit("❌ No training config found. Run 'Configure Training' first.")
        return
    if not Path(train_cfg).exists():
        yield emit(f"❌ Training config not found: {train_cfg}")
        return
    if not dataset_cfg or not Path(dataset_cfg).exists():
        yield emit("❌ Dataset config not found. Run 'Configure Training' first.")
        return

    gpu_idx = gpu_index_from_choice(gpu_index_choice)
    threads = max(int(num_cpu_threads_per_process), 1)

    yield from _run_training_subprocess(train_cfg, dataset_cfg, gpu_idx, threads, base_model)


# ---------------------------------------------------------------------------
# Upload Dataset
# ---------------------------------------------------------------------------

def upload_dataset(files, dataset_name: str, captions_zip=None) -> str:
    if not dataset_name.strip():
        return "❌ Dataset name cannot be empty."
    if not files:
        return "❌ No files uploaded."

    dest = DATASETS_DIR / dataset_name.strip()
    dest.mkdir(parents=True, exist_ok=True)

    copied_images = 0
    copied_txts = 0
    skipped = []

    for f in files:
        # Gradio 4.x returns file path strings or objects with .name/.path
        if isinstance(f, str):
            src = Path(f)
        elif hasattr(f, "name"):
            src = Path(f.name)
        elif hasattr(f, "path"):
            src = Path(f.path)
        else:
            src = Path(str(f))

        ext = src.suffix.lower()
        if ext in IMAGE_EXTS or ext == ".txt":
            dst = dest / src.name
            shutil.copy2(str(src), str(dst))
            if ext in IMAGE_EXTS:
                copied_images += 1
            else:
                copied_txts += 1
        else:
            skipped.append(src.name)

    lines = [f"✓ Dataset saved to: {dest}",
             f"  Images:   {copied_images}",
             f"  Captions: {copied_txts}"]
    if skipped:
        lines.append(f"  Skipped (unsupported): {', '.join(skipped[:10])}")

    missing_captions = []
    for img in dest.iterdir():
        if img.suffix.lower() in IMAGE_EXTS:
            if not (dest / (img.stem + ".txt")).exists():
                missing_captions.append(img.name)
    if missing_captions:
        lines.append(f"\n⚠ {len(missing_captions)} image(s) missing captions:")
        for m in missing_captions[:10]:
            lines.append(f"    • {m}")

    lines.append(f"\nDataset path (use in Training tab):\n{dest}")
    return "\n".join(lines)


def list_datasets() -> list[str]:
    if not DATASETS_DIR.exists():
        return []
    return sorted(str(p) for p in DATASETS_DIR.iterdir() if p.is_dir())


# ---------------------------------------------------------------------------
# Job Queue
# ---------------------------------------------------------------------------

_queue_lock = threading.Lock()
_active_job_log: list[str] = []
_queue_running = False


def load_queue() -> list[dict]:
    if QUEUE_FILE.exists():
        try:
            with open(QUEUE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return []


def save_queue(q: list[dict]):
    with open(QUEUE_FILE, "w") as f:
        json.dump(q, f, indent=2)


def add_to_queue(
    project_name, base_model, image_directory, output_directory,
    network_dim, network_alpha, learning_rate, max_train_epochs,
    resolution, repeats, caption_dropout, gpu_index_choice,
    optimizer_type, lr_scheduler, lr_scheduler_num_cycles, lr_warmup_steps,
    train_batch_size, gradient_accumulation_steps, max_grad_norm,
    save_every_n_epochs, save_last_n_epochs, mixed_precision,
    gradient_checkpointing, seed, noise_offset, multires_noise_discount,
    timestep_sampling, discrete_flow_shift,
    cache_latents, cache_text_encoder_outputs, vae_chunk_size, vae_disable_cache,
    num_cpu_threads_per_process,
) -> str:
    if not project_name.strip():
        return "❌ Project name cannot be empty.", format_queue_html(load_queue())
    if not image_directory.strip() or not output_directory.strip():
        return "❌ Image/Output directory cannot be empty.", format_queue_html(load_queue())

    status_msg, train_cfg, dataset_cfg = configure_training(
        project_name, base_model, image_directory, output_directory,
        network_dim, network_alpha, learning_rate, max_train_epochs,
        resolution, repeats, caption_dropout, gpu_index_choice,
        optimizer_type, lr_scheduler, lr_scheduler_num_cycles, lr_warmup_steps,
        train_batch_size, gradient_accumulation_steps, max_grad_norm,
        save_every_n_epochs, save_last_n_epochs, mixed_precision,
        gradient_checkpointing, seed, noise_offset, multires_noise_discount,
        timestep_sampling, discrete_flow_shift,
        cache_latents, cache_text_encoder_outputs, vae_chunk_size, vae_disable_cache,
        num_cpu_threads_per_process,
    )

    if not train_cfg:
        return f"❌ Config failed:\n{status_msg}", format_queue_html(load_queue())

    gpu_idx = gpu_index_from_choice(gpu_index_choice)
    job = {
        "id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "project_name": project_name,
        "base_model": base_model,
        "train_config": train_cfg,
        "dataset_config": dataset_cfg,
        "output_directory": output_directory,
        "gpu_idx": gpu_idx,
        "threads": int(num_cpu_threads_per_process),
        "status": "pending",
        "added": datetime.now().isoformat(),
    }

    with _queue_lock:
        q = load_queue()
        q.append(job)
        save_queue(q)

    n_pending = sum(1 for j in q if j["status"] == "pending")
    return f"✓ Added '{project_name}' to queue. ({n_pending} pending)", format_queue_html(q)


def remove_from_queue(job_id: str) -> tuple[str, str]:
    with _queue_lock:
        q = load_queue()
        q = [j for j in q if j["id"] != job_id or j["status"] == "running"]
        save_queue(q)
    return "✓ Job removed.", format_queue_html(q)


def clear_done_jobs() -> tuple[str, str]:
    with _queue_lock:
        q = load_queue()
        q = [j for j in q if j["status"] not in ("done", "failed")]
        save_queue(q)
    return "✓ Cleared finished jobs.", format_queue_html(q)


def format_queue_html(q: list[dict]) -> str:
    if not q:
        return "Queue is empty."
    lines = []
    status_icons = {"pending": "⏳", "running": "▶", "done": "✓", "failed": "✗"}
    for job in q:
        icon = status_icons.get(job["status"], "?")
        lines.append(f"{icon} [{job['status'].upper()}] {job['project_name']}  (id: {job['id']})")
        lines.append(f"   Base: {job.get('base_model','?')}  |  GPU: {job.get('gpu_idx','?')}  |  Added: {job.get('added','?')[:19]}")
        if job["status"] in ("done", "failed"):
            lines.append(f"   Output: {job.get('output_directory','?')}")
        lines.append("")
    return "\n".join(lines)


def get_queue_status() -> str:
    q = load_queue()
    return format_queue_html(q)


def start_queue(gpu_index_choice: str, num_cpu_threads: int):
    global _queue_running, _active_job_log
    _active_job_log = []

    def emit(line):
        _active_job_log.append(line)
        return "\n".join(_active_job_log)

    with _queue_lock:
        q = load_queue()
        pending = [j for j in q if j["status"] == "pending"]

    if not pending:
        yield emit("No pending jobs in queue."), format_queue_html(load_queue())
        return

    _queue_running = True
    gpu_idx = gpu_index_from_choice(gpu_index_choice)
    threads = max(int(num_cpu_threads), 1)

    total = len(pending)
    for i, job in enumerate(pending, 1):
        yield emit(f"\n{'='*60}"), format_queue_html(load_queue())
        yield emit(f"Starting job {i}/{total}: {job['project_name']}"), format_queue_html(load_queue())
        yield emit(f"{'='*60}"), format_queue_html(load_queue())

        with _queue_lock:
            q = load_queue()
            for j in q:
                if j["id"] == job["id"]:
                    j["status"] = "running"
            save_queue(q)

        success = True
        for log_text in _run_training_subprocess(
            job["train_config"], job["dataset_config"],
            gpu_idx, threads, job["base_model"]
        ):
            _active_job_log = log_text.split("\n")
            if "✗ Training failed" in log_text:
                success = False
            yield log_text, format_queue_html(load_queue())

        with _queue_lock:
            q = load_queue()
            for j in q:
                if j["id"] == job["id"]:
                    j["status"] = "done" if success else "failed"
            save_queue(q)

        yield emit(f"\n{'✓' if success else '✗'} Job {i}/{total} finished: {job['project_name']}"), format_queue_html(load_queue())

    _queue_running = False
    yield emit(f"\n✓ All {total} jobs completed."), format_queue_html(load_queue())


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------

def list_output_files() -> list[str]:
    """Find all .safetensors files in output directories."""
    files = []
    cfg = load_config()
    search_dirs = set()

    out_dir = cfg.get("output_directory", "")
    if out_dir and Path(out_dir).exists():
        search_dirs.add(Path(out_dir))

    # Also scan any output dirs from queue
    q = load_queue()
    for job in q:
        d = Path(job.get("output_directory", ""))
        if d.exists():
            search_dirs.add(d)

    # And the ROOT/output dir
    root_out = ROOT / "output"
    if root_out.exists():
        search_dirs.add(root_out)

    for d in search_dirs:
        for f in sorted(d.glob("**/*.safetensors"), key=lambda x: x.stat().st_mtime, reverse=True):
            files.append(str(f))

    return files


def refresh_downloads() -> tuple[list, str]:
    files = list_output_files()
    if not files:
        return [], "No .safetensors files found in output directories."
    summary = f"Found {len(files)} file(s):\n" + "\n".join(
        f"  {Path(f).name}  ({Path(f).stat().st_size / 1024 / 1024:.1f} MB)" for f in files
    )
    return files, summary


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    cfg = load_config()
    saved_gpu_idx = str(cfg.get("gpu_index", "0"))
    default_gpu = next(
        (c for c in GPU_CHOICES if c.startswith(saved_gpu_idx + ":")),
        GPU_CHOICES[0] if GPU_CHOICES else "0",
    )

    dit_models = get_available_dit_models()
    saved_base = cfg.get("base_model", "anima-base-v1.0")
    default_model = next((m for m in dit_models if saved_base in m), dit_models[0] if dit_models else saved_base)

    with gr.Blocks(title="Anima LoRA Trainer") as demo:
        gr.Markdown(
            """# Citron's Anima LoRA Trainer

Super Simple Gradio UI for training LoRA adapters on the [Anima](https://huggingface.co/circlestone-labs/Anima) diffusion model.

Created by [Citron Legacy](https://x.com/Citron_Legacy) — [GitHub](https://github.com/citronlegacy/citron-anima-lora-trainer-ui)
"""
        )

        last_train_cfg = gr.State(cfg.get("last_train_config", ""))
        last_dataset_cfg = gr.State(cfg.get("last_dataset_config", ""))

        with gr.Tabs():

            # ================================================================
            # TAB 1 — Training
            # ================================================================
            with gr.Tab("Training"):

                with gr.Group():
                    gr.Markdown("### Project & Paths")
                    with gr.Row():
                        project_name = gr.Textbox(label="Project Name", value=cfg["project_name"], placeholder="my_lora")
                        gpu_dropdown = gr.Dropdown(label="GPU", choices=GPU_CHOICES, value=default_gpu)
                    with gr.Row():
                        base_model_dropdown = gr.Dropdown(
                            label="Base Model",
                            choices=dit_models,
                            value=default_model,
                            info="Local .safetensors files auto-detected from models/anima/dit/. Known models download automatically.",
                        )
                    image_directory = gr.Textbox(
                        label="Image Directory (flat folder with images + .txt captions)",
                        value=cfg["image_directory"], placeholder="/path/to/dataset",
                    )
                    output_directory = gr.Textbox(
                        label="Output Directory (where trained LoRA is saved)",
                        value=cfg["output_directory"], placeholder="/path/to/output",
                    )

                with gr.Group():
                    gr.Markdown("### Network")
                    with gr.Row():
                        network_dim = gr.Number(label="Network Dim", value=cfg["network_dim"], precision=0, minimum=1)
                        network_alpha = gr.Number(label="Network Alpha", value=cfg["network_alpha"], precision=0, minimum=1)
                        learning_rate = gr.Number(label="Learning Rate", value=cfg["learning_rate"])
                        max_train_epochs = gr.Number(label="Max Epochs", value=cfg["max_train_epochs"], precision=0, minimum=1)

                with gr.Group():
                    gr.Markdown("### Dataset")
                    with gr.Row():
                        resolution = gr.Number(label="Resolution (px)", value=cfg["resolution"], precision=0, minimum=64)
                        repeats = gr.Number(label="Repeats", value=cfg["repeats"], precision=0, minimum=1)
                        caption_dropout = gr.Slider(label="Caption Dropout", minimum=0.0, maximum=1.0, step=0.05, value=cfg["caption_dropout"])

                gr.Markdown("---")
                gr.Markdown("### Config & Training")

                with gr.Row():
                    configure_btn = gr.Button("Configure Training", variant="secondary", size="lg")
                    train_btn = gr.Button("Start Training", variant="primary", size="lg")

                custom_config_input = gr.Textbox(
                    label="Override Training Config Path (optional — leave blank to use last generated)",
                    value="", placeholder="/path/to/custom_training_config.toml",
                )

                status_box = gr.Textbox(label="Configuration Status", lines=12, interactive=False, show_copy_button=True)
                log_box = gr.Textbox(label="Training Log", lines=25, interactive=False, show_copy_button=True, autoscroll=True)

            # ================================================================
            # TAB 2 — Advanced Settings
            # ================================================================
            with gr.Tab("Advanced Settings"):
                gr.Markdown("_These settings are applied when you click **Configure Training** or **Add to Queue**._\n\nDefaults match the original notebook.")

                with gr.Group():
                    gr.Markdown("### Optimizer & Scheduler")
                    with gr.Row():
                        optimizer_type = gr.Dropdown(
                            label="Optimizer",
                            choices=["AdamW8bit", "AdamW", "Lion", "SGD", "Prodigy"],
                            value=cfg["optimizer_type"],
                        )
                        lr_scheduler = gr.Dropdown(
                            label="LR Scheduler",
                            choices=["cosine_with_restarts", "cosine", "linear", "constant", "constant_with_warmup", "polynomial"],
                            value=cfg["lr_scheduler"],
                        )
                    with gr.Row():
                        lr_scheduler_num_cycles = gr.Number(label="LR Scheduler Num Cycles", value=cfg["lr_scheduler_num_cycles"], precision=0, minimum=1)
                        lr_warmup_steps = gr.Number(label="LR Warmup Steps", value=cfg["lr_warmup_steps"], precision=0, minimum=0)

                with gr.Group():
                    gr.Markdown("### Batch & Gradient")
                    with gr.Row():
                        train_batch_size = gr.Number(label="Train Batch Size", value=cfg["train_batch_size"], precision=0, minimum=1)
                        gradient_accumulation_steps = gr.Number(label="Gradient Accumulation Steps", value=cfg["gradient_accumulation_steps"], precision=0, minimum=1)
                        max_grad_norm = gr.Number(label="Max Grad Norm", value=cfg["max_grad_norm"])

                with gr.Group():
                    gr.Markdown("### Saving")
                    with gr.Row():
                        save_every_n_epochs = gr.Number(label="Save Every N Epochs", value=cfg["save_every_n_epochs"], precision=0, minimum=1)
                        save_last_n_epochs = gr.Number(label="Keep Last N Checkpoints", value=cfg["save_last_n_epochs"], precision=0, minimum=1)

                with gr.Group():
                    gr.Markdown("### Precision & Memory")
                    with gr.Row():
                        mixed_precision = gr.Dropdown(label="Mixed Precision", choices=["bf16", "fp16", "no"], value=cfg["mixed_precision"])
                        vae_chunk_size = gr.Number(label="VAE Chunk Size", value=cfg["vae_chunk_size"], precision=0, minimum=1)
                    with gr.Row():
                        gradient_checkpointing = gr.Checkbox(label="Gradient Checkpointing", value=cfg["gradient_checkpointing"])
                        cache_latents = gr.Checkbox(label="Cache Latents", value=cfg["cache_latents"])
                        cache_text_encoder_outputs = gr.Checkbox(label="Cache Text Encoder Outputs", value=cfg["cache_text_encoder_outputs"])
                        vae_disable_cache = gr.Checkbox(label="VAE Disable Cache", value=cfg["vae_disable_cache"])

                with gr.Group():
                    gr.Markdown("### Noise & Flow")
                    with gr.Row():
                        noise_offset = gr.Number(label="Noise Offset", value=cfg["noise_offset"])
                        multires_noise_discount = gr.Number(label="Multires Noise Discount", value=cfg["multires_noise_discount"])
                        timestep_sampling = gr.Dropdown(label="Timestep Sampling", choices=["sigmoid", "uniform", "logit_normal"], value=cfg["timestep_sampling"])
                        discrete_flow_shift = gr.Number(label="Discrete Flow Shift", value=cfg["discrete_flow_shift"])

                with gr.Group():
                    gr.Markdown("### Misc")
                    with gr.Row():
                        seed = gr.Number(label="Seed", value=cfg["seed"], precision=0)
                        num_cpu_threads = gr.Number(label="CPU Threads Per Process", value=cfg["num_cpu_threads_per_process"], precision=0, minimum=1)

            # ================================================================
            # TAB 3 — Upload Dataset
            # ================================================================
            with gr.Tab("Upload Dataset"):
                gr.Markdown(
                    "### Upload images and captions from your PC\n"
                    "Upload image files (.jpg, .png, .webp, etc.) and matching .txt caption files. "
                    "Files are saved to `datasets/<name>/` inside the trainer folder."
                )
                with gr.Row():
                    upload_name = gr.Textbox(label="Dataset Name", placeholder="my_dataset_v1")
                upload_files = gr.Files(
                    label="Drag & Drop Images + Caption (.txt) Files",
                )
                upload_btn = gr.Button("Save Dataset", variant="primary")
                upload_status = gr.Textbox(label="Upload Status", lines=12, interactive=False)

                gr.Markdown("---")
                gr.Markdown("### Existing Datasets")
                dataset_list = gr.Textbox(
                    label="Datasets in datasets/ folder",
                    value="\n".join(list_datasets()) or "(none yet)",
                    lines=5, interactive=False,
                )
                refresh_datasets_btn = gr.Button("Refresh Dataset List", variant="secondary")

                upload_btn.click(
                    fn=upload_dataset,
                    inputs=[upload_files, upload_name],
                    outputs=[upload_status],
                )
                refresh_datasets_btn.click(
                    fn=lambda: "\n".join(list_datasets()) or "(none yet)",
                    inputs=[],
                    outputs=[dataset_list],
                )

            # ================================================================
            # TAB 4 — Job Queue
            # ================================================================
            with gr.Tab("Job Queue"):
                gr.Markdown(
                    "### Batch Training Queue\n"
                    "Configure each LoRA in the **Training** tab, then click **Add to Queue**. "
                    "When all jobs are added, click **Start Queue** to run them overnight."
                )

                with gr.Row():
                    add_queue_btn = gr.Button("+ Add Current Settings to Queue", variant="secondary", size="lg")
                    start_queue_btn = gr.Button("▶ Start Queue", variant="primary", size="lg")

                with gr.Row():
                    clear_done_btn = gr.Button("Clear Finished Jobs", variant="stop")
                    refresh_queue_btn = gr.Button("Refresh Status")

                queue_status_box = gr.Textbox(
                    label="Queue",
                    value=get_queue_status(),
                    lines=15, interactive=False,
                )
                queue_add_status = gr.Textbox(label="Add Status", lines=3, interactive=False)
                queue_log_box = gr.Textbox(label="Queue Training Log", lines=25, interactive=False, autoscroll=True)

            # ================================================================
            # TAB 5 — Downloads
            # ================================================================
            with gr.Tab("Downloads"):
                gr.Markdown(
                    "### Download trained LoRA files\n"
                    "Scans all configured output directories for .safetensors files."
                )
                refresh_dl_btn = gr.Button("Refresh File List", variant="secondary")
                dl_summary = gr.Textbox(label="Available Files", lines=8, interactive=False)
                dl_files = gr.Files(label="Download", interactive=False)

                refresh_dl_btn.click(
                    fn=refresh_downloads,
                    inputs=[],
                    outputs=[dl_files, dl_summary],
                )

        # ── Shared advanced inputs ────────────────────────────────────────
        adv_inputs = [
            optimizer_type, lr_scheduler, lr_scheduler_num_cycles, lr_warmup_steps,
            train_batch_size, gradient_accumulation_steps, max_grad_norm,
            save_every_n_epochs, save_last_n_epochs, mixed_precision,
            gradient_checkpointing, seed, noise_offset, multires_noise_discount,
            timestep_sampling, discrete_flow_shift,
            cache_latents, cache_text_encoder_outputs, vae_chunk_size, vae_disable_cache,
            num_cpu_threads,
        ]
        basic_inputs = [
            project_name, base_model_dropdown, image_directory, output_directory,
            network_dim, network_alpha, learning_rate, max_train_epochs,
            resolution, repeats, caption_dropout, gpu_dropdown,
        ]
        all_inputs = basic_inputs + adv_inputs

        # ── Training tab events ───────────────────────────────────────────
        configure_btn.click(
            fn=configure_training,
            inputs=all_inputs,
            outputs=[status_box, last_train_cfg, last_dataset_cfg],
        )
        train_btn.click(
            fn=start_training,
            inputs=[custom_config_input, gpu_dropdown, num_cpu_threads, base_model_dropdown],
            outputs=[log_box],
        )

        # ── Queue tab events ──────────────────────────────────────────────
        add_queue_btn.click(
            fn=add_to_queue,
            inputs=all_inputs,
            outputs=[queue_add_status, queue_status_box],
        )
        start_queue_btn.click(
            fn=start_queue,
            inputs=[gpu_dropdown, num_cpu_threads],
            outputs=[queue_log_box, queue_status_box],
        )
        clear_done_btn.click(
            fn=clear_done_jobs,
            inputs=[],
            outputs=[queue_add_status, queue_status_box],
        )
        refresh_queue_btn.click(
            fn=get_queue_status,
            inputs=[],
            outputs=[queue_status_box],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo = build_ui()
    server_name = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    demo.launch(server_name=server_name, server_port=7860, show_error=True)

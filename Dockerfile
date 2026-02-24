# ══════════════════════════════════════════════════════════════════
# Lightweight image — models download at runtime on RunPod
# Image size: ~15GB instead of ~50GB
# ══════════════════════════════════════════════════════════════════
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=0

# ── Upgrade pip ──────────────────────────────────────────────────
RUN pip install --upgrade pip

# ── System Dependencies ──────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    git-lfs \
    wget \
    curl \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# ── Remove distutils blinker ────────────────────────────────────
RUN rm -rf /usr/lib/python3/dist-packages/blinker* \
    && rm -rf /usr/lib/python3/dist-packages/Blinker* \
    && rm -rf /usr/lib/python3.10/dist-packages/blinker* \
    && rm -rf /usr/lib/python3.10/dist-packages/Blinker*

# ── Working Directory ────────────────────────────────────────────
WORKDIR /app

# ── Install Hallo2 ──────────────────────────────────────────────
RUN git clone https://github.com/fudan-generative-vision/hallo2.git /app/hallo2

COPY requirements.txt /app/requirements.txt

# ── Install Dependencies (Hallo2 runtime + Handler) ─────────────
# Filter out Hallo2 dev/UI packages that conflict with RunPod runtime deps.
RUN grep -vE "^(gradio|pylint|isort|pre-commit)==" /app/hallo2/requirements.txt > /tmp/hallo2-runtime-requirements.txt

RUN pip install \
    --timeout 300 \
    --retries 5 \
    -r /tmp/hallo2-runtime-requirements.txt \
    -r /app/requirements.txt

# ── Upgrade PyTorch for newer GPU architectures ─────────────────
RUN pip install -U --timeout 300 --retries 5 \
    --index-url https://download.pytorch.org/whl/cu126 \
    torch torchvision torchaudio

# ── Fix huggingface_hub version ──────────────────────────────────
# Changed to standard upgrade (-U) instead of --force-reinstall
RUN pip install -U --timeout 300 \
    "huggingface_hub>=0.25.0,<1.0"

# ── Copy all handler files ───────────────────────────────────────
COPY download_models.py /app/download_models.py
COPY handler.py /app/handler.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# ── Set Environment ──────────────────────────────────────────────
ENV PYTHONPATH="/app/hallo2"
ENV MODEL_DIR="/runpod-volume/pretrained_models"
ENV FORCE_CPU_ON_UNSUPPORTED_GPU="0"

# ── Entrypoint ───────────────────────────────────────────────────
CMD ["/app/start.sh"]
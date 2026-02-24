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

# ══════════════════════════════════════════════════════════════════
# STEP 1: Install PyTorch with CUDA 12.6 (RTX 4090 / sm_89 support)
# Must come FIRST — bitsandbytes/triton version depends on torch.
# ══════════════════════════════════════════════════════════════════
RUN pip install --timeout 300 --retries 5 \
    --index-url https://download.pytorch.org/whl/cu126 \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

# ══════════════════════════════════════════════════════════════════
# STEP 2: Install bitsandbytes >= 0.45.0 BEFORE other deps.
# Hallo2 pins bitsandbytes==0.43.1 which uses triton.ops — removed
# in triton 3.x (shipped with torch 2.6). 0.45+ removes that dep.
# ══════════════════════════════════════════════════════════════════
RUN pip install --timeout 300 --retries 5 "bitsandbytes>=0.45.0"

# ══════════════════════════════════════════════════════════════════
# STEP 3: Install Hallo2 + Handler dependencies.
# Exclude packages already installed above or that cause conflicts.
# ══════════════════════════════════════════════════════════════════
# xformers is also excluded — its pinned version (0.0.25.post1) requires torch 2.2.x
# and will downgrade our torch 2.6. We install a compatible version below.
RUN grep -vE "^(gradio|pylint|isort|pre-commit|torch|torchvision|torchaudio|bitsandbytes|xformers)([[:space:]]*[<>=!~].*)?$" \
    /app/hallo2/requirements.txt > /tmp/hallo2-runtime-requirements.txt

RUN pip install \
    --timeout 300 \
    --retries 5 \
    -r /tmp/hallo2-runtime-requirements.txt \
    -r /app/requirements.txt

# ── Re-assert core GPU stack (defensive against transitive downgrades) ──
RUN pip install --timeout 300 --retries 5 \
    --index-url https://download.pytorch.org/whl/cu126 \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

# ── Install xformers compatible with torch 2.6 ───────────────────
RUN pip install --timeout 300 --retries 5 \
    --index-url https://download.pytorch.org/whl/cu126 \
    xformers==0.0.29.post3

# ── Re-assert bitsandbytes (in case anything above pulled in old version) ──
RUN pip install --timeout 300 --retries 5 "bitsandbytes>=0.45.0"

# ── Fix huggingface_hub version ──────────────────────────────────
RUN pip install -U --timeout 300 \
    "huggingface_hub>=0.25.0,<1.0"

# ── Fix onnxruntime-gpu for CUDA 12.1 ──────────────────────────
# Hallo2 pins onnxruntime-gpu==1.18.0 which was built against CUDA 11.8.
# Container has CUDA 12.1 (libcublasLt.so.12). Upgrade to 1.19+ for CUDA 12.
RUN pip install --timeout 300 --retries 5 "onnxruntime-gpu>=1.19.0"

# ══════════════════════════════════════════════════════════════════
# STEP 4: Verify stack at build time — visible in docker build log
# ══════════════════════════════════════════════════════════════════
RUN python - <<'PY'
import torch
import diffusers

print(f"PyTorch: {torch.__version__}")
print(f"CUDA arch list: {torch.cuda.get_arch_list()}")
print(f"diffusers: {diffusers.__version__}")

try:
    import bitsandbytes
    print(f"bitsandbytes: {bitsandbytes.__version__}")
except Exception as error:
    print(f"bitsandbytes import warning: {error}")
PY

# ── Copy all handler files ───────────────────────────────────────
COPY download_models.py /app/download_models.py
COPY handler.py /app/handler.py
COPY gfpgan_worker.py /app/gfpgan_worker.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# ── Set Environment ──────────────────────────────────────────────
ENV PYTHONPATH="/app/hallo2"
ENV MODEL_DIR="/runpod-volume/pretrained_models"

# ── Entrypoint ───────────────────────────────────────────────────
CMD ["/app/start.sh"]
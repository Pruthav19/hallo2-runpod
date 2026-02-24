#!/bin/bash
set -e

echo "🚀 Starting Hallo2 RunPod Worker..."

# ══════════════════════════════════════════════════════════════════
# Download models on first boot if not already present
# Uses RunPod's network volume so models persist across restarts
# First boot: ~5-10 min on RunPod's 10Gbps network
# Subsequent boots: instant (models cached on volume)
# ══════════════════════════════════════════════════════════════════

MODEL_DIR="${MODEL_DIR:-/runpod-volume/pretrained_models}"
INSIGHTFACE_MODEL_NAME="${INSIGHTFACE_MODEL_NAME:-buffalo_l}"
INSIGHTFACE_MODEL_DIR="${MODEL_DIR}/insightface/models/${INSIGHTFACE_MODEL_NAME}"

if [ ! -f "${MODEL_DIR}/.download_complete" ] || [ ! -d "${INSIGHTFACE_MODEL_DIR}" ]; then
    echo "📥 First boot — downloading models to ${MODEL_DIR}..."
    echo "   This takes ~5-10 minutes on RunPod's network."
    echo "   Models will be cached on the network volume for future boots."
    if [ -f "${MODEL_DIR}/.download_complete" ] && [ ! -d "${INSIGHTFACE_MODEL_DIR}" ]; then
        echo "   Detected missing InsightFace models; repairing cache..."
    fi
    
    python /app/download_models.py
    
    # Mark download as complete so we skip this on next boot
    touch "${MODEL_DIR}/.download_complete"
    echo "✅ Models downloaded and cached!"
else
    echo "✅ Models already cached at ${MODEL_DIR}, skipping download."
fi

# Set PYTHONPATH
export PYTHONPATH="/app/hallo2:${PYTHONPATH}"

# Start the RunPod serverless handler
echo "🎬 Starting serverless handler..."
python -u /app/handler.py
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
# InsightFace models are stored flat in face_analysis/models/ (name="" layout)
FACE_LANDMARKER="${MODEL_DIR}/face_analysis/models/face_landmarker_v2_with_blendshapes.task"

# ── Critical: symlink so Hallo2's hardcoded relative paths work ──────────────────
# util.py hardcodes: pretrained_models/face_analysis/models/face_landmarker_v2_with_blendshapes.task
# inference_long.py uses: ./pretrained_models/... relative to /app/hallo2
# Symlinking makes all of Hallo2's relative paths resolve to MODEL_DIR.
ln -sfn "${MODEL_DIR}" /app/hallo2/pretrained_models
echo "🔗 Symlinked /app/hallo2/pretrained_models -> ${MODEL_DIR}"

if [ ! -f "${MODEL_DIR}/.download_complete" ] || [ ! -f "${FACE_LANDMARKER}" ]; then
    echo "📥 First boot — downloading models to ${MODEL_DIR}..."
    echo "   This takes ~5-10 minutes on RunPod's network."
    echo "   Models will be cached on the network volume for future boots."
    if [ -f "${MODEL_DIR}/.download_complete" ] && [ ! -f "${FACE_LANDMARKER}" ]; then
        echo "   Detected missing face_landmarker model; repairing cache..."
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
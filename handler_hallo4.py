"""Minimal RunPod handler for Hallo4 H100 smoke testing.

This handler checks that the container can import PyTorch, see the H100 GPU,
see Hallo4 files, and see downloaded model paths. It does not run full inference yet.
Use this first to validate the image build and model cache on RunPod.
"""

from __future__ import annotations

import os
from pathlib import Path

import runpod


HALLO4_DIR = Path(os.environ.get("HALLO4_DIR", "/app/hallo4"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/runpod-volume/pretrained_models_hallo4"))


def handler(job):
    import torch

    checks = {
        "hallo4_dir_exists": HALLO4_DIR.exists(),
        "inference_module_exists": (HALLO4_DIR / "vace" / "vace_wan_inference.py").exists(),
        "model_weight_exists": (MODEL_DIR / "hallo4" / "model_weight.pth").exists(),
        "wan_vae_exists": (MODEL_DIR / "Wan2.1_Encoders" / "Wan2.1_VAE.pth").exists(),
        "audio_separator_exists": (MODEL_DIR / "audio_separator" / "Kim_Vocal_2.onnx").exists(),
        "wav2vec_exists": (MODEL_DIR / "wav2vec" / "wav2vec2-base-960h" / "config.json").exists(),
    }

    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    return {
        "status": "ok" if all(checks.values()) else "missing_files",
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": gpu_name,
        "model_dir": str(MODEL_DIR),
        "hallo4_dir": str(HALLO4_DIR),
        "checks": checks,
        "next_step": "After smoke test passes, replace handler_hallo4.py with full inference handler using python -m vace.vace_wan_inference.",
    }


runpod.serverless.start({"handler": handler})

"""Download Hallo4 pretrained models into the RunPod network volume.

The upstream Hallo4 README expects:

  ./pretrained_models/
  |-- hallo4/model_weight.pth
  |-- Wan2.1_Encoders/...
  |-- audio_separator/Kim_Vocal_2.onnx
  |-- wav2vec/wav2vec2-base-960h/...

This script downloads the Hugging Face repository to MODEL_DIR and then creates
/app/hallo4/pretrained_models as a symlink so upstream relative paths work.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = os.environ.get("HALLO4_HF_REPO", "fudan-generative-ai/hallo4")
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/runpod-volume/pretrained_models_hallo4"))
HALLO4_DIR = Path(os.environ.get("HALLO4_DIR", "/app/hallo4"))


def _ensure_symlink() -> None:
    target = HALLO4_DIR / "pretrained_models"
    if target.is_symlink() or target.exists():
        if target.is_symlink() and target.resolve() == MODEL_DIR.resolve():
            return
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    target.symlink_to(MODEL_DIR, target_is_directory=True)
    print(f"🔗 Symlinked {target} -> {MODEL_DIR}")


def _validate() -> None:
    required = [
        MODEL_DIR / "hallo4" / "model_weight.pth",
        MODEL_DIR / "Wan2.1_Encoders" / "Wan2.1_VAE.pth",
        MODEL_DIR / "Wan2.1_Encoders" / "models_t5_umt5-xxl-enc-bf16.pth",
        MODEL_DIR / "audio_separator" / "Kim_Vocal_2.onnx",
        MODEL_DIR / "wav2vec" / "wav2vec2-base-960h" / "config.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required Hallo4 model files:\n" + "\n".join(missing))


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    done_file = MODEL_DIR / ".download_complete"

    if done_file.exists():
        print(f"✅ Hallo4 models already cached at {MODEL_DIR}")
    else:
        print(f"📥 Downloading Hallo4 models from {REPO_ID} to {MODEL_DIR}")
        snapshot_download(
            repo_id=REPO_ID,
            local_dir=str(MODEL_DIR),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        done_file.touch()
        print("✅ Hallo4 model download complete")

    _ensure_symlink()
    _validate()


if __name__ == "__main__":
    main()

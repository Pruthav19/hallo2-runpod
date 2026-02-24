"""
Download all required pretrained models for Hallo2.
Designed to run on first boot on RunPod (fast network).
Models are saved to a network volume so they persist across restarts.
"""
import os
import subprocess
import sys

MODEL_DIR = os.environ.get("MODEL_DIR", "/runpod-volume/pretrained_models")


def download_hallo2_models():
    """Download Hallo2 pretrained models from HuggingFace."""
    print("📥 Downloading Hallo2 models from HuggingFace...")
    print(f"   Target directory: {MODEL_DIR}")

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="fudan-generative-ai/hallo2",
        local_dir=MODEL_DIR,
    )
    print("✅ Hallo2 models downloaded successfully!")


def download_gfpgan_models():
    """Download GFPGAN v1.4 model for face enhancement."""
    print("📥 Downloading GFPGAN model...")
    gfpgan_dir = os.path.join(MODEL_DIR, "gfpgan")
    os.makedirs(gfpgan_dir, exist_ok=True)

    gfpgan_path = os.path.join(gfpgan_dir, "GFPGANv1.4.pth")
    if os.path.exists(gfpgan_path):
        print("   GFPGAN model already exists, skipping.")
        return

    subprocess.run([
        "wget", "-q", "--timeout=300", "--tries=5",
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
        "-O", gfpgan_path,
    ], check=True)
    print("✅ GFPGAN model downloaded!")


def download_insightface_models():
    """Download InsightFace model pack required by Hallo2 image processor."""
    model_name = os.environ.get("INSIGHTFACE_MODEL_NAME", "buffalo_l")
    insightface_root = os.path.join(MODEL_DIR, "insightface")
    model_dir = os.path.join(insightface_root, "models", model_name)

    if os.path.isdir(model_dir) and os.listdir(model_dir):
        print(f"   InsightFace model '{model_name}' already exists, skipping.")
        return

    print(f"📥 Downloading InsightFace model pack: {model_name}...")
    os.makedirs(insightface_root, exist_ok=True)

    from insightface.utils.storage import ensure_available

    ensure_available("models", model_name, root=insightface_root)
    print("✅ InsightFace model downloaded!")


if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)

    try:
        download_hallo2_models()
        download_gfpgan_models()
        download_insightface_models()
        print("\n🎉 All models downloaded and ready!")
    except Exception as e:
        print(f"\n❌ Model download failed: {e}", file=sys.stderr)
        sys.exit(1)
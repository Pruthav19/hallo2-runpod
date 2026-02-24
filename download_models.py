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
    """
    Download InsightFace buffalo_l models for Hallo2.

    Hallo2's image_processor.py calls:
        FaceAnalysis(name="", root=face_analysis_model_path)
    InsightFace with name="" scans {root}/models/ for .onnx files directly.
    So we download buffalo_l.zip and extract its contents flat into
    {MODEL_DIR}/face_analysis/models/ (not into a buffalo_l subfolder).
    """
    import shutil
    import urllib.request
    import zipfile
    import io

    face_analysis_dir = os.path.join(MODEL_DIR, "face_analysis")
    models_dir = os.path.join(face_analysis_dir, "models")

    # Check if onnx files already present
    if os.path.isdir(models_dir) and any(
        f.endswith(".onnx") for f in os.listdir(models_dir)
    ):
        print("   InsightFace models already present, skipping.")
        return

    os.makedirs(models_dir, exist_ok=True)

    # Download buffalo_l using insightface's own downloader into a temp root,
    # then flatten the onnx files up into models_dir so name="" can find them.
    tmp_root = os.path.join(face_analysis_dir, "_tmp_download")
    os.makedirs(tmp_root, exist_ok=True)

    print("📥 Downloading InsightFace buffalo_l model pack...")
    from insightface.utils.storage import ensure_available
    ensure_available("models", "buffalo_l", root=tmp_root)

    # Copy all files from buffalo_l/ directly into models/ (flat layout)
    buffalo_src = os.path.join(tmp_root, "models", "buffalo_l")
    for fname in os.listdir(buffalo_src):
        shutil.copy2(os.path.join(buffalo_src, fname), models_dir)

    shutil.rmtree(tmp_root)
    print("✅ InsightFace models ready at", models_dir)


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
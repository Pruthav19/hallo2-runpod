"""
RunPod Serverless Handler for Hallo2 Talking Head Video Generation
Accepts: avatar image + (text OR audio) → Returns: video URL
"""

import os
import uuid
import subprocess
import asyncio
import runpod
import boto3
import requests
import yaml
import torch
import logging

# ─── Configuration ───────────────────────────────────────────────
WORKSPACE = "/tmp/workspace"
MODEL_DIR = os.environ.get("MODEL_DIR", "/runpod-volume/pretrained_models")
HALLO2_DIR = "/app/hallo2"
S3_BUCKET = os.environ.get("S3_BUCKET", "your-bucket-name")
S3_REGION = os.environ.get("S3_REGION", "us-east-1")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", None)  # For R2/MinIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── S3 Client ───────────────────────────────────────────────────
def get_s3_client():
    kwargs = {
        "aws_access_key_id": S3_ACCESS_KEY,
        "aws_secret_access_key": S3_SECRET_KEY,
        "region_name": S3_REGION,
    }
    if S3_ENDPOINT:
        kwargs["endpoint_url"] = S3_ENDPOINT
    return boto3.client("s3", **kwargs)


# ─── Utility Functions ───────────────────────────────────────────
def download_file(url, dest_path):
    """Download a file from URL to local path."""
    logger.info(f"Downloading: {url}")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    logger.info(f"Saved to: {dest_path}")
    return dest_path


def upload_to_s3(local_path, s3_key):
    """Upload file to S3 and return public/presigned URL."""
    s3 = get_s3_client()
    s3.upload_file(local_path, S3_BUCKET, s3_key, ExtraArgs={"ContentType": "video/mp4"})

    # Generate a presigned URL valid for 1 hour
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": s3_key},
        ExpiresIn=3600,
    )
    logger.info(f"Uploaded to S3: {s3_key}")
    return url


def generate_tts(text, voice, output_path):
    """Generate speech audio from text using Edge-TTS."""
    import edge_tts

    async def _generate():
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)

    asyncio.run(_generate())
    logger.info(f"TTS generated: {output_path}")
    return output_path


def convert_audio_to_wav(input_path, output_path):
    """Convert any audio format to WAV (required by Hallo2)."""
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000",    # 16kHz sample rate
        "-ac", "1",        # Mono
        "-sample_fmt", "s16",
        output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    logger.info(f"Converted audio to WAV: {output_path}")
    return output_path


def get_audio_duration(audio_path):
    """Get duration of audio file in seconds."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
        capture_output=True, text=True
    )
    return float(result.stdout.strip())


def generate_talking_head(image_path, audio_path, output_path, pose_weight=1.0,
                           face_weight=1.0, lip_weight=1.5):
    """Run Hallo2 inference to generate talking-head video."""
    import glob  # Required to search for the output video
    
    # 1. Create a job-specific YAML config to set the save path dynamically
    base_config_path = os.path.join(HALLO2_DIR, "configs/inference/long.yaml")
    job_dir = os.path.dirname(output_path)
    hallo_out_dir = os.path.join(job_dir, "hallo_out")
    job_config_path = os.path.join(job_dir, "job_config.yaml")
    
    with open(base_config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Hallo2's ImageProcessor calls FaceAnalysis(name="", root=face_analysis_model_path).
    # InsightFace with name="" scans {root}/models/ for .onnx files directly.
    # We must always override this to the absolute path where we placed the models,
    # ignoring whatever relative path is in the base config YAML.
    face_analysis_root = os.path.join(MODEL_DIR, "face_analysis")
    if not isinstance(config.get("face_analysis"), dict):
        config["face_analysis"] = {}
    config["face_analysis"]["model_path"] = face_analysis_root

    config["save_path"] = hallo_out_dir  # Override the default save directory
    
    with open(job_config_path, "w") as f:
        yaml.dump(config, f)

    # 2. Run inference using the new config (and REMOVE the --output argument)
    cmd = [
        "python",
        os.path.join(HALLO2_DIR, "scripts/inference_long.py"),
        "--config", job_config_path,
        "--source_image", image_path,
        "--driving_audio", audio_path,
        "--pose_weight", str(pose_weight),
        "--face_weight", str(face_weight),
        "--lip_weight", str(lip_weight),
    ]

    env = os.environ.copy()
    env["PRETRAINED_MODEL_DIR"] = MODEL_DIR
    env["INSIGHTFACE_HOME"] = os.path.join(MODEL_DIR, "insightface")

    logger.info(f"Running Hallo2 inference: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=HALLO2_DIR, env=env)

    if result.returncode != 0:
        logger.error(f"Hallo2 STDERR: {result.stderr}")
        raise RuntimeError(f"Hallo2 inference failed: {result.stderr}")

    logger.info("Hallo2 inference completed successfully. Locating generated video...")
    
    # 3. Find the generated mp4 and rename it to match what the rest of the script expects
    mp4_files = glob.glob(os.path.join(hallo_out_dir, "**", "*.mp4"), recursive=True)
    
    if not mp4_files:
        raise RuntimeError("Hallo2 finished but no mp4 file was found in the output directory!")
    
    # Grab the first mp4 found (usually named merge_video.mp4)
    generated_video = mp4_files[0]
    os.rename(generated_video, output_path)

    return output_path

def enhance_video(input_path, output_path):
    """Optional: Enhance face quality using GFPGAN frame-by-frame."""
    frames_dir = os.path.join(WORKSPACE, "frames")
    enhanced_dir = os.path.join(WORKSPACE, "enhanced_frames")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(enhanced_dir, exist_ok=True)

    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        os.path.join(frames_dir, "frame_%05d.png")
    ], check=True, capture_output=True)

    subprocess.run([
        "python", "-m", "gfpgan.inference_gfpgan",
        "-i", frames_dir,
        "-o", enhanced_dir,
        "-v", "1.4",
        "-s", "1",
        "--only_center_face",
        "--model_path", os.path.join(MODEL_DIR, "gfpgan", "GFPGANv1.4.pth"),
    ], check=True, capture_output=True)

    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", "25",
        "-i", os.path.join(enhanced_dir, "restored_imgs/frame_%05d.png"),
        "-i", input_path,
        "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        output_path
    ], check=True, capture_output=True)

    return output_path


# ─── Main Handler ────────────────────────────────────────────────
def handler(event):
    """
    RunPod Serverless Handler

    Expected input:
    {
        "input": {
            "avatar_image_url": "https://...",         # REQUIRED
            "text": "Hello, this is a test...",         # OPTIONAL (provide text OR audio_url)
            "audio_url": "https://...",                 # OPTIONAL (provide text OR audio_url)
            "voice": "en-US-JennyNeural",              # OPTIONAL: TTS voice
            "pose_weight": 1.0,                        # OPTIONAL
            "face_weight": 1.0,                        # OPTIONAL
            "lip_weight": 1.5,                         # OPTIONAL
            "enhance": false                           # OPTIONAL
        }
    }
    """
    try:
        input_data = event["input"]
        job_id = str(uuid.uuid4())[:8]

        if not input_data.get("avatar_image_url"):
            return {"error": "avatar_image_url is required"}
        if not input_data.get("text") and not input_data.get("audio_url"):
            return {"error": "Either 'text' or 'audio_url' must be provided"}

        job_dir = os.path.join(WORKSPACE, job_id)
        os.makedirs(job_dir, exist_ok=True)

        # ── Step 1: Download avatar image ──
        image_ext = input_data["avatar_image_url"].split(".")[-1].split("?")[0]
        image_path = os.path.join(job_dir, f"avatar.{image_ext}")
        download_file(input_data["avatar_image_url"], image_path)

        # ── Step 2: Get or generate audio ──
        if input_data.get("audio_url"):
            audio_ext = input_data["audio_url"].split(".")[-1].split("?")[0]
            raw_audio = os.path.join(job_dir, f"audio_raw.{audio_ext}")
            download_file(input_data["audio_url"], raw_audio)
        else:
            voice = input_data.get("voice", "en-US-JennyNeural")
            raw_audio = os.path.join(job_dir, "audio_raw.mp3")
            generate_tts(input_data["text"], voice, raw_audio)

        wav_audio = os.path.join(job_dir, "audio.wav")
        convert_audio_to_wav(raw_audio, wav_audio)

        duration = get_audio_duration(wav_audio)
        logger.info(f"Audio duration: {duration:.1f}s")

        # ── Step 3: Generate talking-head video ──
        raw_video = os.path.join(job_dir, "output_raw.mp4")
        generate_talking_head(
            image_path=image_path,
            audio_path=wav_audio,
            output_path=raw_video,
            pose_weight=input_data.get("pose_weight", 1.0),
            face_weight=input_data.get("face_weight", 1.0),
            lip_weight=input_data.get("lip_weight", 1.5),
        )

        # ── Step 4: Optional enhancement ──
        if input_data.get("enhance", False):
            final_video = os.path.join(job_dir, "output_enhanced.mp4")
            enhance_video(raw_video, final_video)
        else:
            final_video = raw_video

        # ── Step 5: Upload to S3 ──
        s3_key = f"generated_videos/{job_id}.mp4"
        video_url = upload_to_s3(final_video, s3_key)

        subprocess.run(["rm", "-rf", job_dir], capture_output=True)

        return {
            "video_url": video_url,
            "duration_seconds": duration,
            "job_id": job_id,
            "status": "success",
        }

    except Exception as e:
        logger.exception("Handler error")
        return {"error": str(e), "status": "failed"}


# ─── Entry Point ─────────────────────────────────────────────────
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
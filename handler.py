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


def preprocess_avatar_image(image_path, output_path, target_size=512):
    """
    Prepare avatar image for Hallo2 inference:
      1. Correct EXIF orientation
      2. Convert to RGB (strips alpha, fixes CMYK/palette inputs)
      3. Detect face with OpenCV Haar cascade → square-crop centered on face
         (padding = 1.5× face side so neck/hair are included)
      4. Fall back to centre-crop if no face is detected
      5. Resize to target_size × target_size
      6. Save as lossless PNG for maximum quality entering the model
    """
    import cv2
    import numpy as np
    from PIL import Image, ImageOps

    # --- open & fix orientation ---
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)          # honour EXIF rotation
    img = img.convert("RGB")
    w, h = img.size

    # --- face detection ---
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(64, 64),
    )

    if len(faces) > 0:
        # Pick the largest detected face
        fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
        face_size = max(fw, fh)
        cx = fx + fw // 2
        cy = fy + fh // 2

        # Generous padding: face should occupy 50-70% of the final image
        # (official Hallo2 requirement). With half = 0.85 × face_side the face
        # fills ~59% of the 512×512 output after LANCZOS resize.
        half = int(face_size * 0.85)
        x1 = max(cx - half, 0)
        y1 = max(cy - half, 0)
        x2 = min(cx + half, w)
        y2 = min(cy + half, h)
        logger.info(f"Face detected at ({fx},{fy},{fw},{fh}) → crop ({x1},{y1},{x2},{y2})")
    else:
        # Fallback: centre-square crop
        logger.warning("No face detected – falling back to centre-square crop")
        side = min(w, h)
        x1 = (w - side) // 2
        y1 = (h - side) // 2
        x2 = x1 + side
        y2 = y1 + side

    # --- square crop: pad with black if the crop would exceed image bounds ---
    crop_w = x2 - x1
    crop_h = y2 - y1
    side = max(crop_w, crop_h)
    square = Image.new("RGB", (side, side), (0, 0, 0))
    square.paste(img.crop((x1, y1, x2, y2)), (0, 0))

    # --- resize & save ---
    out = square.resize((target_size, target_size), Image.LANCZOS)
    out.save(output_path, format="PNG", optimize=False)
    logger.info(f"Preprocessed avatar saved: {output_path} ({target_size}×{target_size})")
    return output_path


def generate_talking_head(image_path, audio_path, output_path,
                           pose_weight=0.3, face_weight=0.3, lip_weight=0.8,
                           inference_steps=20, cfg_scale=3.5,
                           face_expand_ratio=1.5):
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

    config["inference_steps"] = inference_steps
    config["cfg_scale"] = cfg_scale
    config["face_expand_ratio"] = face_expand_ratio

    # Disable segment cutting for clips under 60 s — use_cut=True creates
    # internal merge seams where face identity can shift between segments.
    audio_duration = get_audio_duration(audio_path)
    if audio_duration < 60:
        config["use_cut"] = False
        logger.info(f"Audio {audio_duration:.1f}s < 60s: use_cut disabled to prevent segment seams")

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

    # ── Post-processing: fade-in + quality re-encode ────────────────────────
    # Hallo2 generates unstable / warped frames in the first ~0.3 s while the
    # face animation warms up.  A short fade-in from the source image hides
    # this completely.  We also re-encode at CRF 15 (high quality) without
    # any blurring filter – the negative-luma unsharp that was here before was
    # softening the image, which hurt rather than helped perceived quality.
    #
    # Filter chain:
    #   fade=in:0:8  → fade from black over first 8 frames (~0.33 s at 25 fps)
    postproc_video = output_path + "_postproc.mp4"
    fps_result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate",
         "-of", "default=noprint_wrappers=1:nokey=1", generated_video],
        capture_output=True, text=True,
    )
    fps_str = fps_result.stdout.strip() or "25"
    fps_num = fps_str.split("/")[0] if "/" in fps_str else fps_str.split(".")[0]
    fade_frames = max(6, round(int(fps_num) * 0.3))  # ~0.3 s worth of frames

    # Filter chain:
    #   1. fade-in from black (hides Hallo2's warm-up warp in first ~0.3 s)
    #   2. 2× lanczos upscale  512×512 → 1024×1024  (huge perceived quality lift)
    #   3. gentle sharpen to counteract upscale softness
    vf_chain = (
        f"fade=type=in:start_frame=0:nb_frames={fade_frames},"
        f"scale=iw*2:ih*2:flags=lanczos,"
        f"unsharp=5:5:0.7:5:5:0.0"
    )
    postproc_result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", generated_video,
            "-vf", vf_chain,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "17", "-preset", "slow",
            "-c:a", "copy",
            postproc_video,
        ],
        capture_output=True, text=True,
    )
    if postproc_result.returncode == 0 and os.path.exists(postproc_video):
        os.rename(postproc_video, output_path)
        logger.info(f"Post-processing applied (fade-in, 2× upscale, sharpen, CRF 17)")
    else:
        logger.warning("Post-processing skipped: " + postproc_result.stderr[-200:])
        os.rename(generated_video, output_path)

    return output_path


def apply_identity_lock(
    pose_weight,
    face_weight,
    lip_weight,
    face_expand_ratio,
    inference_steps,
    cfg_scale,
    enabled=True,
):
    """Clamp motion settings to reduce identity drift between frames.

    Hallo2 quality issues that look like "a different person" are often caused by
    aggressive motion controls and overly wide crops that give the model too much
    freedom to reinterpret facial structure. This helper enforces conservative
    defaults while still allowing users to pass lower values explicitly.
    """
    if not enabled:
        return {
            "pose_weight": pose_weight,
            "face_weight": face_weight,
            "lip_weight": lip_weight,
            "face_expand_ratio": face_expand_ratio,
            "inference_steps": inference_steps,
            "cfg_scale": cfg_scale,
        }

    tuned = {
        "pose_weight": min(pose_weight, 0.22),
        "face_weight": min(face_weight, 0.22),
        "lip_weight": min(lip_weight, 0.75),
        "face_expand_ratio": min(face_expand_ratio, 1.35),
        "inference_steps": max(inference_steps, 40),
        "cfg_scale": min(cfg_scale, 3.2),
    }
    logger.info(
        "Identity lock enabled. Effective settings: "
        f"pose={tuned['pose_weight']}, face={tuned['face_weight']}, "
        f"lip={tuned['lip_weight']}, face_expand_ratio={tuned['face_expand_ratio']}, "
        f"steps={tuned['inference_steps']}, cfg={tuned['cfg_scale']}"
    )
    return tuned

def enhance_video(input_path, output_path):
    """Enhance face quality using GFPGAN (runs in isolated subprocess).

    GFPGAN uses basicsr internally. Hallo2 ships its own bundled basicsr under
    /app/hallo2/basicsr which is on PYTHONPATH.  Importing GFPGANer in the same
    process triggers a double-registration of ResNetArcFace in basicsr's arch
    registry and raises an AssertionError.  Running the enhancement in a child
    process with /app/hallo2 stripped from PYTHONPATH avoids the collision.
    """
    job_dir = os.path.dirname(output_path)
    frames_dir = os.path.join(job_dir, "frames")
    enhanced_dir = os.path.join(job_dir, "enhanced_frames")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(enhanced_dir, exist_ok=True)

    # ── Get source FPS so the rebuilt video matches exactly ──
    fps_result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate",
         "-of", "default=noprint_wrappers=1:nokey=1", input_path],
        capture_output=True, text=True,
    )
    fps_str = fps_result.stdout.strip() or "25"

    # ── Extract frames ──
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path,
         os.path.join(frames_dir, "frame_%05d.png")],
        check=True, capture_output=True,
    )

    # ── Run GFPGAN in a subprocess with Hallo2's basicsr removed from path ──
    model_path = os.path.join(MODEL_DIR, "gfpgan", "GFPGANv1.4.pth")
    worker_path = os.path.join(os.path.dirname(__file__), "gfpgan_worker.py")

    # Strip /app/hallo2 from PYTHONPATH so only the pip-installed basicsr is visible
    clean_env = os.environ.copy()
    pythonpath = clean_env.get("PYTHONPATH", "")
    clean_env["PYTHONPATH"] = ":".join(
        p for p in pythonpath.split(":") if p not in ("/app/hallo2", "")
    )

    logger.info("Running GFPGAN enhancement in isolated subprocess...")
    result = subprocess.run(
        ["python", worker_path, frames_dir, enhanced_dir, model_path],
        capture_output=True, text=True, env=clean_env,
    )
    if result.returncode != 0:
        logger.error(f"GFPGAN worker STDERR: {result.stderr}")
        raise RuntimeError(f"GFPGAN enhancement failed: {result.stderr[-500:]}")
    logger.info(result.stdout.strip())

    # ── Reassemble video with original audio + 2× upscale + sharpen ──
    subprocess.run(
        ["ffmpeg", "-y",
         "-r", fps_str,
         "-i", os.path.join(enhanced_dir, "frame_%05d.png"),
         "-i", input_path,
         "-map", "0:v", "-map", "1:a",
         "-vf", "scale=iw*2:ih*2:flags=lanczos,unsharp=5:5:0.7:5:5:0.0",
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "17", "-preset", "slow",
         "-shortest",
         output_path],
        check=True, capture_output=True,
    )

    return output_path


# ─── Main Handler ────────────────────────────────────────────────
def handler(event):
    """
    RunPod Serverless Handler

    Expected input:
    {
        "input": {
            "avatar_image_url": "https://...",  # REQUIRED
            "text": "Hello...",                 # OPTIONAL – provide text OR audio_url
            "audio_url": "https://...",         # OPTIONAL – provide text OR audio_url
            "voice": "en-US-JennyNeural",       # OPTIONAL TTS voice (default: en-US-JennyNeural)

            # ── Motion weights (0.0 – 1.0, lower = subtler) ──────────────────
            "pose_weight": 0.3,     # head pose movement  (default: 0.3)
            "face_weight": 0.3,     # facial expression   (default: 0.3)
            "lip_weight":  0.8,     # lip sync strength   (default: 0.8)

            # ── Quality / speed ──────────────────────────────────────────────
            "inference_steps": 40,  # diffusion steps; 20=fast, 40=best (default: 40)
            "cfg_scale": 3.5,       # classifier-free guidance scale  (default: 3.5)

            # ── Image pre-processing ─────────────────────────────────────────
            "target_size": 512,     # avatar resize resolution (default: 512)

            # ── Post-processing ───────────────────────────────────────────────
            "enhance": false        # GFPGAN face enhancement (default: false)
            "identity_lock": true   # clamp settings for stable identity
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
        raw_image_path = os.path.join(job_dir, f"avatar_raw.{image_ext}")
        download_file(input_data["avatar_image_url"], raw_image_path)

        # ── Step 1b: Preprocess avatar (face crop, resize, RGB PNG) ──
        image_path = os.path.join(job_dir, "avatar.png")
        preprocess_avatar_image(
            raw_image_path, image_path,
            target_size=input_data.get("target_size", 512),
        )

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

        tuned = apply_identity_lock(
            pose_weight=float(input_data.get("pose_weight", 0.3)),
            face_weight=float(input_data.get("face_weight", 0.3)),
            lip_weight=float(input_data.get("lip_weight", 0.8)),
            inference_steps=int(input_data.get("inference_steps", 40)),
            cfg_scale=float(input_data.get("cfg_scale", 3.5)),
            face_expand_ratio=float(input_data.get("face_expand_ratio", 1.5)),
            enabled=bool(input_data.get("identity_lock", True)),
        )

        generate_talking_head(
            image_path=image_path,
            audio_path=wav_audio,
            output_path=raw_video,
            pose_weight=tuned["pose_weight"],
            face_weight=tuned["face_weight"],
            lip_weight=tuned["lip_weight"],
            inference_steps=tuned["inference_steps"],
            cfg_scale=tuned["cfg_scale"],
            face_expand_ratio=tuned["face_expand_ratio"],
        )

        # ── Step 4: Upload raw video immediately (safety net) ──
        raw_s3_key = f"generated_videos/{job_id}_raw.mp4"
        raw_video_url = upload_to_s3(raw_video, raw_s3_key)
        logger.info(f"Raw video uploaded: {raw_s3_key}")

        # ── Step 5: Optional enhancement ──
        enhanced_video_url = None
        if input_data.get("enhance", False):
            try:
                logger.warning(
                    "GFPGAN enhancement is enabled. For strict identity stability, "
                    "keep enhance=false because per-frame restoration can introduce "
                    "temporal identity drift."
                )
                final_video = os.path.join(job_dir, "output_enhanced.mp4")
                enhance_video(raw_video, final_video)
                enhanced_s3_key = f"generated_videos/{job_id}_enhanced.mp4"
                enhanced_video_url = upload_to_s3(final_video, enhanced_s3_key)
                logger.info(f"Enhanced video uploaded: {enhanced_s3_key}")
            except Exception as enhance_err:
                logger.warning(f"Enhancement failed, returning raw video only: {enhance_err}")

        # ── Step 6: Clean up ──
        subprocess.run(["rm", "-rf", job_dir], capture_output=True)

        result = {
            "video_url": enhanced_video_url or raw_video_url,
            "raw_video_url": raw_video_url,
            "duration_seconds": duration,
            "job_id": job_id,
            "status": "success",
        }
        if enhanced_video_url:
            result["enhanced_video_url"] = enhanced_video_url

        return result

    except Exception as e:
        logger.exception("Handler error")
        return {"error": str(e), "status": "failed"}


# ─── Entry Point ─────────────────────────────────────────────────
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

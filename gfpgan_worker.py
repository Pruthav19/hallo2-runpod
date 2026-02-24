"""
Standalone GFPGAN worker – run as a subprocess with /app/hallo2 removed from
PYTHONPATH so that GFPGAN's own basicsr does not clash with Hallo2's bundled
basicsr registry (which would cause 'ResNetArcFace already registered' errors).

Usage (called by handler.py):
    python /app/gfpgan_worker.py <frames_dir> <enhanced_dir> <model_path>
"""
import sys
import os
import cv2

from gfpgan import GFPGANer

frames_dir, enhanced_dir, model_path = sys.argv[1], sys.argv[2], sys.argv[3]

os.makedirs(enhanced_dir, exist_ok=True)

restorer = GFPGANer(
    model_path=model_path,
    upscale=1,
    arch="clean",
    channel_multiplier=2,
    bg_upsampler=None,
)

frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
for fname in frame_files:
    src = os.path.join(frames_dir, fname)
    img_bgr = cv2.imread(src)
    _, _, restored = restorer.enhance(
        img_bgr,
        has_aligned=False,
        only_center_face=True,
        paste_back=True,
    )
    cv2.imwrite(
        os.path.join(enhanced_dir, fname),
        restored if restored is not None else img_bgr,
    )

print(f"Enhanced {len(frame_files)} frames", flush=True)

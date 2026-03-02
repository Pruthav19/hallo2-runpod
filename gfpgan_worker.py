"""
Standalone GFPGAN + Real-ESRGAN worker – run as a subprocess with /app/hallo2
removed from PYTHONPATH so that GFPGAN's own basicsr does not clash with
Hallo2's bundled basicsr registry.

Pipeline per frame:
  1. Real-ESRGAN 2× neural super-resolution (background + non-face areas)
  2. GFPGAN face restoration at native resolution → paste back at 2×
  3. Blend 35 % restored / 65 % original-upscaled (drift guard: skip if Δ>28)

Output is 1024×1024 when input is 512×512.

Usage (called by handler.py):
    python /app/gfpgan_worker.py <frames_dir> <enhanced_dir> <model_path> [realesrgan_model_path]
"""
import sys
import os
import types

# ── Compatibility shim ────────────────────────────────────────────────────────
# basicsr imports `torchvision.transforms.functional_tensor` removed in 0.16+.
if "torchvision.transforms.functional_tensor" not in sys.modules:
    import torchvision.transforms.functional as _F
    _shim = types.ModuleType("torchvision.transforms.functional_tensor")
    _shim.rgb_to_grayscale = _F.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = _shim
# ─────────────────────────────────────────────────────────────────────────────

import cv2
import numpy as np
import torch
from gfpgan import GFPGANer

frames_dir = sys.argv[1]
enhanced_dir = sys.argv[2]
model_path = sys.argv[3]
realesrgan_model_path = sys.argv[4] if len(sys.argv) > 4 else None

os.makedirs(enhanced_dir, exist_ok=True)

# ── Build Real-ESRGAN background upsampler ───────────────────────────────────
bg_upsampler = None
if realesrgan_model_path and os.path.exists(realesrgan_model_path):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    rrdb_net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                       num_block=23, num_grow_ch=32, scale=2)
    bg_upsampler = RealESRGANer(
        scale=2,
        model_path=realesrgan_model_path,
        dni_weight=None,
        model=rrdb_net,
        tile=400,           # tile-based processing to save VRAM
        tile_pad=10,
        pre_pad=0,
        half=True,          # FP16 for speed
    )
    print(f"Real-ESRGAN 2× upsampler loaded from {realesrgan_model_path}", flush=True)
else:
    print("Real-ESRGAN model not found – falling back to GFPGAN at 1×", flush=True)

# ── Build GFPGAN restorer ────────────────────────────────────────────────────
upscale_factor = 2 if bg_upsampler else 1
restorer = GFPGANer(
    model_path=model_path,
    upscale=upscale_factor,
    arch="clean",
    channel_multiplier=2,
    bg_upsampler=bg_upsampler,
)


def safe_blend(original: np.ndarray, restored: np.ndarray,
               strength: float = 0.35, max_drift: float = 28.0) -> np.ndarray:
    """
    Blend GFPGAN+Real-ESRGAN restoration onto the (upscaled) original with a
    per-frame drift guard.

    strength  = 0.35  → 35 % restored, 65 % original
    max_drift = 28.0  → skip frames where GFPGAN hallucinated too much
    """
    if restored is None:
        return original

    # If GFPGAN upscaled, we need to upscale the original to match for blending
    if restored.shape[:2] != original.shape[:2]:
        original = cv2.resize(original, (restored.shape[1], restored.shape[0]),
                              interpolation=cv2.INTER_LANCZOS4)

    diff = np.mean(np.abs(restored.astype(np.float32) - original.astype(np.float32)))
    if diff > max_drift:
        return original

    return cv2.addWeighted(restored, strength, original, 1.0 - strength, 0)


frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
skipped = 0
for fname in frame_files:
    src = os.path.join(frames_dir, fname)
    img_bgr = cv2.imread(src)

    _, _, restored = restorer.enhance(
        img_bgr,
        has_aligned=False,
        only_center_face=True,
        paste_back=True,
        weight=1.0,
    )

    output = safe_blend(img_bgr, restored, strength=0.35, max_drift=28.0)
    if output is img_bgr:
        skipped += 1
    cv2.imwrite(os.path.join(enhanced_dir, fname), output)

print(f"Enhanced {len(frame_files)} frames ({skipped} skipped due to drift guard) "
      f"[upscale={upscale_factor}×, blend=35%]", flush=True)


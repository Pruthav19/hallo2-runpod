"""
Standalone GFPGAN worker – run as a subprocess with /app/hallo2 removed from
PYTHONPATH so that GFPGAN's own basicsr does not clash with Hallo2's bundled
basicsr registry (which would cause 'ResNetArcFace already registered' errors).

Usage (called by handler.py):
    python /app/gfpgan_worker.py <frames_dir> <enhanced_dir> <model_path>
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

def feathered_blend(original: np.ndarray, restored: np.ndarray,
                    faces: list, feather_px: int = 60,
                    strength: float = 0.15) -> np.ndarray:
    """
    Blend GFPGAN restoration onto the original frame using a feathered mask.

    strength=0.15 means GFPGAN contributes at most 15% at the face centre,
    fading to 0% at the edges via a Gaussian-blurred ellipse.
    This eliminates per-frame face-identity drift while still cleaning noise.
    """
    if restored is None:
        return original
    if not faces:
        return cv2.addWeighted(restored, strength, original, 1.0 - strength, 0)

    h, w = original.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    for face in faces:
        x1, y1, x2, y2 = [int(v) for v in face["bbox"]]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        ax = max((x2 - x1) // 2 + feather_px // 2, 1)
        ay = max((y2 - y1) // 2 + feather_px // 2, 1)
        cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 1.0, -1)

    ksize = feather_px * 2 + 1
    mask = cv2.GaussianBlur(mask, (ksize, ksize), feather_px / 2)
    mask = np.clip(mask * strength, 0.0, 1.0)
    mask3 = mask[:, :, np.newaxis]

    blended = (restored.astype(np.float32) * mask3
               + original.astype(np.float32) * (1.0 - mask3))
    return blended.clip(0, 255).astype(np.uint8)


frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
for fname in frame_files:
    src = os.path.join(frames_dir, fname)
    img_bgr = cv2.imread(src)

    # weight=1.0: full GFPGAN restoration; blend strength is controlled by
    # feathered_blend(strength=0.15) — max 15% contribution at face centre.
    faces, _, restored = restorer.enhance(
        img_bgr,
        has_aligned=False,
        only_center_face=True,
        paste_back=True,
        weight=1.0,
    )

    output = feathered_blend(img_bgr, restored, faces if faces else [])
    cv2.imwrite(os.path.join(enhanced_dir, fname), output)

print(f"Enhanced {len(frame_files)} frames", flush=True)


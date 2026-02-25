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

def safe_blend(original: np.ndarray, restored: np.ndarray,
               strength: float = 0.10, max_drift: float = 20.0) -> np.ndarray:
    """
    Blend GFPGAN restoration onto the original frame at low strength, with a
    per-frame drift guard.

    strength=0.10: GFPGAN contributes 10%, original 90%.

    max_drift=20.0: if the mean-absolute-difference between the GFPGAN output
    and the original exceeds this value (on a 0–255 scale), the frame is
    considered "drifted" and the original is returned untouched.  This stops
    the handful of frames where GFPGAN hallucinates noticeably different facial
    features from ever appearing in the final video.
    """
    if restored is None:
        return original

    # Drift guard — skip enhancement when GFPGAN changed the face too much
    diff = np.mean(np.abs(restored.astype(np.float32) - original.astype(np.float32)))
    if diff > max_drift:
        return original

    return cv2.addWeighted(restored, strength, original, 1.0 - strength, 0)


frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
skipped = 0
for fname in frame_files:
    src = os.path.join(frames_dir, fname)
    img_bgr = cv2.imread(src)

    # GFPGANer.enhance() returns (cropped_faces, restored_faces, restored_img)
    # cropped_faces/restored_faces are numpy arrays, not bbox dicts.
    _, _, restored = restorer.enhance(
        img_bgr,
        has_aligned=False,
        only_center_face=True,
        paste_back=True,
        weight=1.0,
    )

    output = safe_blend(img_bgr, restored, strength=0.10, max_drift=20.0)
    if output is img_bgr:
        skipped += 1
    cv2.imwrite(os.path.join(enhanced_dir, fname), output)

print(f"Enhanced {len(frame_files)} frames ({skipped} skipped due to drift guard)", flush=True)


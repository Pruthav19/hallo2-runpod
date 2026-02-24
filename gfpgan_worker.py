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
                    strength: float = 0.15) -> np.ndarray:
    """
    Blend GFPGAN restoration onto the original frame at low strength.

    strength=0.15: GFPGAN contributes only 15%, original contributes 85%.
    This preserves temporal consistency (no per-frame identity drift) while
    still softening compression artefacts from Hallo2's output.
    """
    if restored is None:
        return original
    return cv2.addWeighted(restored, strength, original, 1.0 - strength, 0)


frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
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

    output = feathered_blend(img_bgr, restored, strength=0.15)
    cv2.imwrite(os.path.join(enhanced_dir, fname), output)

print(f"Enhanced {len(frame_files)} frames", flush=True)


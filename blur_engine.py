"""
blur_engine.py
─────────────────────────────────────────────────────────────────────────────
All blurring strategies live here.  Each function accepts a full BGR frame
and a list of face dicts (from FaceDetector.detect) and returns the blurred
frame.  The original frame is never mutated.

Blur modes
──────────
pixel   – classic mosaic / pixelation (default, most recognisable as "censored")
gaussian– heavy Gaussian blur (smooth, softer look)
box     – fast box/average blur (slightly cheaper on CPU)

Shape modes
───────────
rect    – rectangular bounding box (fastest)
oval    – elliptical mask blended over the face (more natural)

Usage
─────
    from blur_engine import apply_blur
    out = apply_blur(frame, faces, mode="pixel", block_size=12, shape="oval")
"""

import cv2
import numpy as np


# ──────────────────────────────────────────────────── internal helpers ── #

def _pixel_blur(roi: np.ndarray, block_size: int = 12) -> np.ndarray:
    """Shrink then zoom back → mosaic pixelation."""
    h, w = roi.shape[:2]
    bs = max(1, block_size)
    small_w = max(1, w // bs)
    small_h = max(1, h // bs)
    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def _gaussian_blur(roi: np.ndarray, strength: int = 51) -> np.ndarray:
    """Gaussian blur.  strength must be odd."""
    k = strength if strength % 2 == 1 else strength + 1
    k = max(3, k)
    return cv2.GaussianBlur(roi, (k, k), 0)


def _box_blur(roi: np.ndarray, strength: int = 25) -> np.ndarray:
    """Simple box / average blur — cheapest option."""
    k = strength if strength % 2 == 1 else strength + 1
    k = max(3, k)
    return cv2.blur(roi, (k, k))


# Dispatch table
_BLUR_FN = {
    "pixel": _pixel_blur,
    "gaussian": _gaussian_blur,
    "box": _box_blur,
}


# ────────────────────────────────────────────────────── oval masking ── #

def _apply_oval_mask(
    frame: np.ndarray,
    blurred_roi: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> np.ndarray:
    """
    Blend blurred_roi into frame using a soft elliptical mask.
    Pixels outside the ellipse stay untouched; inside are blurred.
    A feathered edge avoids hard outlines.
    """
    h_roi = y2 - y1
    w_roi = x2 - x1

    # Build ellipse mask (0=transparent, 255=opaque)
    mask = np.zeros((h_roi, w_roi), dtype=np.uint8)
    cx, cy = w_roi // 2, h_roi // 2
    cv2.ellipse(
        mask,
        (cx, cy),
        (max(1, cx - 2), max(1, cy - 2)),
        0, 0, 360,
        255,
        -1,
    )
    # Feather / soften the mask edges
    feather_k = max(3, min(w_roi, h_roi) // 8)
    if feather_k % 2 == 0:
        feather_k += 1
    mask = cv2.GaussianBlur(mask, (feather_k, feather_k), 0)

    # Alpha blend
    alpha = mask.astype(np.float32) / 255.0
    alpha_3 = np.stack([alpha] * 3, axis=-1)

    orig_roi = frame[y1:y2, x1:x2].astype(np.float32)
    blur_roi = blurred_roi.astype(np.float32)
    blended = (blur_roi * alpha_3 + orig_roi * (1 - alpha_3)).astype(np.uint8)

    out = frame.copy()
    out[y1:y2, x1:x2] = blended
    return out


# ────────────────────────────────────────────────────── public API ──── #

def apply_blur(
    frame: np.ndarray,
    faces: list[dict],
    mode: str = "pixel",
    block_size: int = 15,
    strength: int = 55,
    shape: str = "oval",
) -> np.ndarray:
    """
    Apply blur to every detected face in `faces`.

    Parameters
    ----------
    frame      : BGR numpy array (from OpenCV)
    faces      : list of face dicts from FaceDetector.detect()
    mode       : "pixel" | "gaussian" | "box"
    block_size : mosaic tile size in pixels (pixel mode only)
    strength   : kernel size for gaussian/box modes
    shape      : "oval" (elliptical mask) | "rect" (rectangle, fastest)

    Returns
    -------
    New BGR frame with faces blurred.  Original is not modified.
    """
    if not faces:
        return frame

    blur_fn = _BLUR_FN.get(mode, _pixel_blur)
    out = frame.copy()

    for face in faces:
        x1, y1, x2, y2 = face["x1"], face["y1"], face["x2"], face["y2"]

        # Guard: ensure valid ROI
        if x1 >= x2 or y1 >= y2:
            continue

        roi = out[y1:y2, x1:x2]

        # Build blurred region
        if mode == "pixel":
            blurred = blur_fn(roi, block_size=block_size)
        else:
            blurred = blur_fn(roi, strength=strength)

        if shape == "oval":
            out = _apply_oval_mask(out, blurred, x1, y1, x2, y2)
        else:
            out[y1:y2, x1:x2] = blurred

    return out

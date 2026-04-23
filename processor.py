"""
processor.py
─────────────────────────────────────────────────────────────────────────────
Unified processing pipeline.  Handles three input modes:

  webcam  – live capture from a camera index (default 0)
  video   – read from a file, write blurred output to a new file
  image   – single image in → single image out

Pressing  Q  or  ESC  closes the live window.
Pressing  S  in live mode saves the current frame as a PNG.

Overlay info rendered on-screen (can be toggled):
  • FPS counter
  • Face count
  • Blur mode & shape
  • Confidence threshold
"""

import time
import os
import cv2
import numpy as np
from tqdm import tqdm

from face_detector import FaceDetector
from blur_engine import apply_blur


# ─────────────────────────────────────────────────────── helpers ─── #

def _draw_overlay(
    frame: np.ndarray,
    fps: float,
    face_count: int,
    mode: str,
    shape: str,
    show: bool = True,
) -> np.ndarray:
    if not show:
        return frame
    out = frame.copy()
    lines = [
        f"FPS: {fps:5.1f}",
        f"Faces: {face_count}",
        f"Mode: {mode} / {shape}",
    ]
    y = 28
    for line in lines:
        # shadow
        cv2.putText(out, line, (11, y + 1), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 2, cv2.LINE_AA)
        # text
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 120), 2, cv2.LINE_AA)
        y += 28
    return out


def _draw_face_boxes(
    frame: np.ndarray,
    faces: list[dict],
    show: bool = False,
) -> np.ndarray:
    """Optional: draw a thin box around each detected face for debugging."""
    if not show:
        return frame
    out = frame.copy()
    for f in faces:
        cv2.rectangle(out, (f["x1"], f["y1"]), (f["x2"], f["y2"]),
                      (0, 200, 255), 1)
        label = f"{f['confidence']:.2f}"
        cv2.putText(out, label, (f["x1"], f["y1"] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
    return out


# ─────────────────────────────────────────────── Webcam / live ──── #

def run_webcam(
    camera_index: int = 0,
    model_selection: int = 0,
    min_confidence: float = 0.4,
    padding: float = 0.20,
    blur_mode: str = "pixel",
    block_size: int = 15,
    strength: int = 55,
    shape: str = "oval",
    show_overlay: bool = True,
    show_boxes: bool = False,
    width: int = 1280,
    height: int = 720,
):
    """
    Open webcam, detect & blur faces in real time.

    Hotkeys
    -------
    Q / ESC  – quit
    S        – save screenshot
    O        – toggle overlay
    B        – toggle face debug boxes
    """
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # reduce buffer lag

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    print(f"[webcam] Press Q/ESC to quit  |  S to screenshot  |  O overlay  |  B boxes")

    fps_smooth = 0.0
    alpha = 0.1          # EMA smoothing for FPS display
    save_idx = 0

    with FaceDetector(model_selection, min_confidence, padding) as detector:
        while True:
            t0 = time.perf_counter()

            ok, frame = cap.read()
            if not ok:
                print("[webcam] Frame read failed – retrying…")
                continue

            faces = detector.detect(frame)
            out = apply_blur(frame, faces, mode=blur_mode,
                             block_size=block_size, strength=strength,
                             shape=shape)

            # optional debug boxes on the blurred output
            out = _draw_face_boxes(out, faces, show=show_boxes)
            out = _draw_overlay(out, fps_smooth, len(faces),
                                blur_mode, shape, show=show_overlay)

            cv2.imshow("Face Blur  [Q=quit]", out)

            elapsed = time.perf_counter() - t0
            instant_fps = 1.0 / elapsed if elapsed > 0 else 0
            fps_smooth = alpha * instant_fps + (1 - alpha) * fps_smooth

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):          # Q or ESC
                break
            elif key == ord("s"):
                fname = f"screenshot_{save_idx:04d}.png"
                cv2.imwrite(fname, out)
                print(f"[webcam] Saved {fname}")
                save_idx += 1
            elif key == ord("o"):
                show_overlay = not show_overlay
            elif key == ord("b"):
                show_boxes = not show_boxes

    cap.release()
    cv2.destroyAllWindows()
    print("[webcam] Done.")


# ─────────────────────────────────────────────── Video file ───── #

def run_video(
    input_path: str,
    output_path: str | None = None,
    model_selection: int = 0,
    min_confidence: float = 0.4,
    padding: float = 0.20,
    blur_mode: str = "pixel",
    block_size: int = 15,
    strength: int = 55,
    shape: str = "oval",
    preview: bool = False,
):
    """
    Process a video file, write blurred version to output_path.
    If output_path is None, auto-generates name: <input>_blurred.mp4
    """
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_blurred{ext}"

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    fps_in  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps_in, (w, h))

    print(f"[video] {input_path}  →  {output_path}")
    print(f"        {w}×{h} @ {fps_in:.1f} fps  |  {total} frames")

    with FaceDetector(model_selection, min_confidence, padding) as detector:
        with tqdm(total=total, unit="frame") as bar:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                faces = detector.detect(frame)
                out = apply_blur(frame, faces, mode=blur_mode,
                                 block_size=block_size, strength=strength,
                                 shape=shape)
                writer.write(out)
                if preview:
                    cv2.imshow("Processing…", out)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                bar.update(1)

    cap.release()
    writer.release()
    if preview:
        cv2.destroyAllWindows()
    print(f"[video] Saved → {output_path}")


# ─────────────────────────────────────────────── Still image ──── #

def run_image(
    input_path: str,
    output_path: str | None = None,
    model_selection: int = 0,
    min_confidence: float = 0.4,
    padding: float = 0.20,
    blur_mode: str = "pixel",
    block_size: int = 15,
    strength: int = 55,
    shape: str = "oval",
    show: bool = True,
):
    """
    Blur faces in a single image.
    """
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_blurred{ext}"

    frame = cv2.imread(input_path)
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")

    with FaceDetector(model_selection, min_confidence, padding) as detector:
        faces = detector.detect(frame)

    out = apply_blur(frame, faces, mode=blur_mode,
                     block_size=block_size, strength=strength, shape=shape)
    cv2.imwrite(output_path, out)
    print(f"[image] {len(faces)} face(s) blurred → {output_path}")

    if show:
        cv2.imshow("Result  [any key to close]", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

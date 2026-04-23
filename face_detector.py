"""
face_detector.py
─────────────────────────────────────────────────────────────────────────────
Ultra-low-latency face detector built on Google MediaPipe's BlazeFace model.

Key design choices
──────────────────
• MediaPipe Short-Range model  → fastest for webcam / close-up footage
• MediaPipe Full-Range model   → better for wide shots / surveillance footage
• min_detection_confidence     → lower = catch more partial faces (but more FP)
• model_selection 0 (short)    is tuned for faces 2 m away; model 1 (full)
  handles up to 5 m.
"""

import mediapipe as mp
import numpy as np


class FaceDetector:
    """
    Wraps MediaPipe Face Detection.

    Parameters
    ----------
    model_selection : int
        0 = short-range (~2 m, fastest)
        1 = full-range  (~5 m, slightly slower but catches distant/partial)
    min_confidence : float
        Detection threshold.  0.3–0.5 catches most partial/occluded faces.
        Lower → more detections, more false positives.
    padding : float
        Fractional padding added around each detected face box.
        0.15 = expand box by 15 % on each side (helps blur include forehead/chin).
    """

    def __init__(
        self,
        model_selection: int = 0,
        min_confidence: float = 0.4,
        padding: float = 0.20,
    ):
        self.padding = padding
        self._mp_fd = mp.solutions.face_detection
        self.detector = self._mp_fd.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_confidence,
        )

    # ------------------------------------------------------------------ #
    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        """
        Detect faces in a BGR OpenCV frame.

        Returns
        -------
        list of dicts with keys:
            x1, y1, x2, y2   – absolute pixel coordinates (clamped to frame)
            confidence        – detection score 0–1
        """
        h, w = frame_bgr.shape[:2]
        # MediaPipe works in RGB
        rgb = frame_bgr[:, :, ::-1]
        results = self.detector.process(rgb)

        faces = []
        if not results.detections:
            return faces

        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            score = det.score[0]

            # Raw normalized coords
            rx, ry = bbox.xmin, bbox.ymin
            rw, rh = bbox.width, bbox.height

            # Add padding
            pad_x = rw * self.padding
            pad_y = rh * self.padding
            rx -= pad_x
            ry -= pad_y
            rw += 2 * pad_x
            rh += 2 * pad_y

            # Convert to absolute + clamp
            x1 = max(0, int(rx * w))
            y1 = max(0, int(ry * h))
            x2 = min(w, int((rx + rw) * w))
            y2 = min(h, int((ry + rh) * h))

            if x2 > x1 and y2 > y1:
                faces.append(
                    {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": score}
                )

        return faces

    # ------------------------------------------------------------------ #
    def close(self):
        self.detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

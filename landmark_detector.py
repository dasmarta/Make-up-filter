import cv2
import mediapipe as mp
import numpy as np
import json
import time
import urllib.request
import os
from dataclasses import dataclass, field, asdict
from typing import Optional
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode

LIPS       = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318,
              402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270,
              409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]

LEFT_CHEEK = [117, 118, 101, 205,187,123]
RIGHT_CHEEK = [347,330,425,411,376, 352 ]

LEFT_EYEBROW  = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")

def _ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    print("[INFO] Preuzimam face_landmarker.task")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[OK]   Model preuzet.")
    except Exception as exc:
        raise RuntimeError(
            f"\n[GRESKA] Ne mogu automatski preuzeti model:\n  {exc}\n\n"
            "Rucno preuzmi model ovdje:\n"
            f"  {MODEL_URL}\n"
            f"i spremi ga u isti direktorij kao ovu skriptu:\n"
            f"  {MODEL_PATH}\n"
        ) from exc

@dataclass
class FaceRegion:
    points:           list  
    pixel_points:     list   
    landmark_indices: list  

@dataclass
class FaceLandmarks:
    timestamp:    float
    frame_width:  int
    frame_height: int
    face_detected: bool

    lips:             Optional[FaceRegion] = None   
    left_cheek:       Optional[FaceRegion] = None
    right_cheek:      Optional[FaceRegion] = None
    left_eyebrow:     Optional[FaceRegion] = None   
    right_eyebrow:    Optional[FaceRegion] = None 

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

class FaceLandmarkDetector:

    def __init__(self, num_faces: int = 1, det_conf: float = 0.6, track_conf: float = 0.5):
        _ensure_model()
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_faces=num_faces,
            min_face_detection_confidence=det_conf,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=track_conf,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(opts)
        self._ts_ms = 0

    def _extract(self, lm_list, indices: list, w: int, h: int) -> FaceRegion:
        norm, pxl = [], []
        for i in indices:
            p = lm_list[i]
            norm.append([round(p.x, 5), round(p.y, 5)])
            pxl.append([int(p.x * w), int(p.y * h)])
        return FaceRegion(points=norm, pixel_points=pxl, landmark_indices=indices)

    def process_frame(self, bgr_frame: np.ndarray) -> FaceLandmarks:
        h, w = bgr_frame.shape[:2]
        result = FaceLandmarks(
            timestamp=time.time(), frame_width=w, frame_height=h, face_detected=False
        )

        self._ts_ms += 33

        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        det = self._landmarker.detect_for_video(mp_img, self._ts_ms)

        if not det.face_landmarks:
            return result

        lm = det.face_landmarks[0]
        result.face_detected = True

        result.lips            = self._extract(lm, LIPS,        w, h)

        result.left_cheek  = self._extract(lm, LEFT_CHEEK,  w, h)
        result.right_cheek = self._extract(lm, RIGHT_CHEEK, w, h)

        result.left_eyebrow    = self._extract(lm, LEFT_EYEBROW,  w, h)
        result.right_eyebrow   = self._extract(lm, RIGHT_EYEBROW, w, h)

        return result

    def draw_debug(self, frame: np.ndarray, lm: FaceLandmarks) -> np.ndarray: # SAMO ZA TEST ZA DETEKCIJE
        if not lm.face_detected:
            cv2.putText(frame, "Nema lica", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 60, 220), 2)
            return frame

        overlay = frame.copy()

        def fill(pts, color):
            if pts:
                cv2.fillPoly(overlay, [np.array(pts, np.int32)], color)

        fill(lm.lips.pixel_points, (0, 0, 210))

        fill(lm.left_cheek.pixel_points,  (147, 20, 255))
        fill(lm.right_cheek.pixel_points, (147, 20, 255))

        fill(lm.left_eyebrow.pixel_points,  (33, 36, 51))
        fill(lm.right_eyebrow.pixel_points, (33, 36, 51))

        frame = cv2.addWeighted(overlay, 0.40, frame, 0.60, 0)

        cv2.putText(frame, "Landmark detekcija", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        legend = [
            ("Usne", (0, 0, 210)),
            ("Obrazi", (147, 20, 255)),
            ("Obrve", (33, 36, 51)),
        ]
        for i, (label, color) in enumerate(legend):
            y = 55 + i * 22
            cv2.rectangle(frame, (10, y - 12), (24, y + 2), color, -1)
            cv2.putText(frame, label, (30, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1)

        return frame

    def release(self):
        self._landmarker.close()
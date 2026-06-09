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

    def draw_debug(self, frame: np.ndarray, lm: FaceLandmarks) -> np.ndarray: # za probu
        
        if not lm.face_detected:
            cv2.putText(frame, "Lice nije detektirano", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 60, 220), 2)
            return frame

  
        mattes = self.create_alpha_matte(lm)
        display = frame.copy()

        
        lip_color   = np.array([120, 130, 210], dtype=np.uint8)
        cheek_color  = np.array([200, 180, 255], dtype=np.uint8)
        eyebrow_color   = np.array([51, 36, 33], dtype=np.uint8)  
        regions = [("lips", lip_color), ("cheeks", cheek_color), ("eyebrows", eyebrow_color)]

        for region_name, color in regions:
            alpha = mattes[region_name].astype(float) / 255.0
            alpha = np.expand_dims(alpha, axis=2)
            display = (alpha * color + (1.0 - alpha) * display).astype(np.uint8)

        
        
        cv2.putText(display, "Alpha Matte + Konture", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        legend = [
            ("Usne", (0, 0, 210)),
            ("Obrazi", (147, 20, 255)),
            ("Obrve", (33, 36, 51)),
        ]
        for i, (label, color) in enumerate(legend):
            y = 55 + i * 22
            cv2.rectangle(display, (10, y - 12), (24, y + 2), color, -1)
            cv2.putText(display, label, (30, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1)

        return display


    def create_masks(self, lm: FaceLandmarks):
        h = lm.frame_height
        w = lm.frame_width

        lips_mask = np.zeros((h, w), dtype=np.uint8)
        cheeks_mask = np.zeros((h, w), dtype=np.uint8)
        eyebrows_mask = np.zeros((h, w), dtype=np.uint8)

        if not lm.face_detected:
            return lips_mask, cheeks_mask, eyebrows_mask

  
        cv2.fillPoly(
            lips_mask,
            [np.array(lm.lips.pixel_points, np.int32)],
            255
        )


        cv2.fillPoly(
            cheeks_mask,
            [np.array(lm.left_cheek.pixel_points, np.int32)],
            255
        )

        cv2.fillPoly(
            cheeks_mask,
            [np.array(lm.right_cheek.pixel_points, np.int32)],
            255
        )

        cv2.fillPoly(
            eyebrows_mask,
            [np.array(lm.left_eyebrow.pixel_points, np.int32)],
            255
        )

        cv2.fillPoly(
            eyebrows_mask,
            [np.array(lm.right_eyebrow.pixel_points, np.int32)],
            255
        )

        return lips_mask, cheeks_mask, eyebrows_mask
    

    def get_contours(self, lm: FaceLandmarks):
        if not lm.face_detected:
            return None

        lips_mask, cheeks_mask, eyebrows_mask = self.create_masks(lm)

        lips_contours, _ = cv2.findContours(
            lips_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cheeks_contours, _ = cv2.findContours(
            cheeks_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        eyebrows_contours, _ = cv2.findContours(
            eyebrows_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        return {
            "lips": lips_contours,
            "cheeks": cheeks_contours,
            "eyebrows": eyebrows_contours
        }
    

    def create_alpha_matte(self, lm: FaceLandmarks, blur_radius: int = 55):  # povećanjem blur_radius dobijemo mekše rubove maski, ali i gubitak detalja
    
        lips_mask, cheeks_mask, eyebrows_mask = self.create_masks(lm)

        return {
            "lips":    cv2.GaussianBlur(lips_mask, (blur_radius, blur_radius), 0),
            "cheeks":   cv2.GaussianBlur(cheeks_mask, (blur_radius, blur_radius), 0),
            "eyebrows": cv2.GaussianBlur(eyebrows_mask, (blur_radius, blur_radius), 0),
           
        }
    

    def evaluate_masks(self, lm: FaceLandmarks):
        
        if not lm.face_detected:
            return {"status": "Nema lica", "metrics": {"lips": 0, "cheeks": 0, "eyebrows": 0}}

        mattes = self.create_alpha_matte(lm, blur_radius=35)
        
        
        lips_score = float(np.mean(mattes["lips"]))
        cheeks_score = float(np.mean(mattes["cheeks"]))
        eyebrows_score = float(np.mean(mattes["eyebrows"]))

        return {
            "status": "Uspjesno",
            "metrics": {
                "lips": round(lips_score, 2),
                "cheeks": round(cheeks_score, 2),
                "eyebrows": round(eyebrows_score, 2)
            }
        }

    def release(self):
        self._landmarker.close()
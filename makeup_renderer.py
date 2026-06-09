import cv2
import numpy as np
from dataclasses import dataclass
from landmark_detector import FaceRegion, FaceLandmarks




def visualize_masks(frame: np.ndarray, mattes: dict) -> np.ndarray:
    display = frame.astype(np.float32)
    colors = {
        "lips":     np.array([0,   0,   200], dtype=np.float32),
        "cheeks":   np.array([160, 90,  230], dtype=np.float32),
        "eyebrows": np.array([30,  35,  55 ], dtype=np.float32),
    }
    for name, color in colors.items():
        matte = mattes.get(name)
        if matte is None:
            continue
        a = (matte.astype(np.float32) / 255.0)[:, :, np.newaxis]
        display = color * a + display * (1.0 - a)
    return np.clip(display, 0, 255).astype(np.uint8)



LIPS_PRESETS = [
    ("Roza",   (150, 85,  210)),
    ("Crvena", (0,   0,   185)),
    ("Nude",   (75,  105, 155)),
]

BLUSH_PRESETS = [
    ("Crvena",     (40,  45,  190)),
    ("Roza",       (150, 80,  225)),
    ("Breskvasta", (55,  115, 235)),  
]

EYEBROW_PRESETS = [
    ("Crna",      (10,  10,  15)),
    ("Smeda",     (20,  25,  50)),
    ("Svjetlija", (60,  90,  145)),
]


@dataclass
class MakeupParams:
    lips_color:         tuple = LIPS_PRESETS[0][1]
    lips_intensity:     float = 0.75
    blush_color:        tuple = BLUSH_PRESETS[1][1]   
    blush_intensity:    float = 0.35
    eyebrow_color:      tuple = EYEBROW_PRESETS[1][1] 
    eyebrow_intensity:  float = 0.35



def _apply_lips(frame: np.ndarray, region: FaceRegion,
                color_bgr: tuple, intensity: float) -> np.ndarray:
    pts = np.array(region.pixel_points, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)

    mask_roi = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    cv2.fillPoly(mask_roi, [pts - np.array([x1, y1])], 255)
    cv2.GaussianBlur(mask_roi, (7, 7), 0, dst=mask_roi)
    alpha = mask_roi.astype(np.float32) / 255.0 * intensity

    roi = frame[y1:y2, x1:x2]
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
    target_hsv = cv2.cvtColor(np.uint8([[list(color_bgr)]]), cv2.COLOR_BGR2HSV)[0, 0]

    colored_hsv = roi_hsv.copy()
    colored_hsv[:, :, 0] = float(target_hsv[0])
    colored_hsv[:, :, 1] = np.clip(roi_hsv[:, :, 1] * 0.3 + float(target_hsv[1]) * 0.7, 0, 255)

    roi_colored = cv2.cvtColor(colored_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    result = frame.copy()
    a = alpha[:, :, np.newaxis]
    result[y1:y2, x1:x2] = np.clip(
        roi_colored.astype(np.float32) * a + roi.astype(np.float32) * (1.0 - a),
        0, 255
    ).astype(np.uint8)
    return result


def _apply_blush(frame: np.ndarray,
                 left: FaceRegion | None, right: FaceRegion | None,
                 color_bgr: tuple, intensity: float) -> np.ndarray:
    overlay = frame.copy()
    for region in (left, right):
        if region is None:
            continue
        cv2.fillPoly(overlay, [np.array(region.pixel_points, dtype=np.int32)], color_bgr)
    cv2.GaussianBlur(overlay, (61, 61), sigmaX=0, dst=overlay)
    return cv2.addWeighted(overlay, intensity, frame, 1.0 - intensity, 0)


def _apply_eyebrows(frame: np.ndarray,
                    left: FaceRegion | None, right: FaceRegion | None,
                    color_bgr: tuple, intensity: float) -> np.ndarray:
    overlay = frame.copy()
    for region in (left, right):
        if region is None:
            continue
        cv2.fillPoly(overlay, [np.array(region.pixel_points, dtype=np.int32)], color_bgr)
    cv2.GaussianBlur(overlay, (5, 5), sigmaX=0, dst=overlay)
    return cv2.addWeighted(overlay, intensity, frame, 1.0 - intensity, 0)


def apply_makeup(frame: np.ndarray, landmarks: FaceLandmarks,
                 params: MakeupParams) -> np.ndarray:
    if not landmarks.face_detected:
        return frame

    result = frame

    if landmarks.lips is not None:
        result = _apply_lips(result, landmarks.lips, params.lips_color, params.lips_intensity)

    result = _apply_blush(result,
                          landmarks.left_cheek, landmarks.right_cheek,
                          params.blush_color, params.blush_intensity)

    result = _apply_eyebrows(result,
                             landmarks.left_eyebrow, landmarks.right_eyebrow,
                             params.eyebrow_color, params.eyebrow_intensity)

    return result

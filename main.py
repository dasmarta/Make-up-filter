import cv2
import time
from landmark_detector import FaceLandmarkDetector
from makeup_renderer import (apply_makeup, MakeupParams,
                             LIPS_PRESETS, BLUSH_PRESETS, EYEBROW_PRESETS)

def main():
    detector = FaceLandmarkDetector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera nije pronadjena!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    print("Tipke: Q=izlaz  A=šminka on/off  S=screenshot")
    print("       1=boja usana  2=boja rumenila  3=boja obrva")

    show_makeup   = False
    makeup_params = MakeupParams()
    lips_idx, blush_idx, eyebrow_idx = 0, 1, 1
    last_lm = None
    fps, t0, n = 0, time.time(), 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        lm = detector.process_frame(frame)
        last_lm = lm

        display = apply_makeup(frame, lm, makeup_params) if show_makeup else frame.copy()

        n += 1
        if time.time() - t0 >= 1.0:
            fps = n / (time.time() - t0)
            n, t0 = 0, time.time()

        status = "LICE DETEKTIRANO" if lm.face_detected else "NEMA LICA"
        cv2.putText(display, status, (10, display.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 220, 100) if lm.face_detected else (0, 60, 220), 2)
        cv2.putText(display, f"FPS: {fps:.1f}", (display.shape[1] - 110, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if show_makeup:
            labels = [
                f"1-Usne:    {LIPS_PRESETS[lips_idx][0]}",
                f"2-Rumenilo:{BLUSH_PRESETS[blush_idx][0]}",
                f"3-Obrve:   {EYEBROW_PRESETS[eyebrow_idx][0]}",
            ]
            for i, txt in enumerate(labels):
                cv2.putText(display, txt, (10, 30 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv2.putText(display, txt, (10, 30 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)

        cv2.imshow("Make-up filter [Q=izlaz]", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('a'):
            show_makeup = not show_makeup
        elif key == ord('s'):
            if last_lm and last_lm.face_detected:
                fn = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(fn, display)
                print(f"\nScreenshot: {fn}")
            else:
                print("\nNema lica za screenshot.")
        elif key == ord('1'):
            lips_idx = (lips_idx + 1) % len(LIPS_PRESETS)
            makeup_params.lips_color = LIPS_PRESETS[lips_idx][1]
        elif key == ord('2'):
            blush_idx = (blush_idx + 1) % len(BLUSH_PRESETS)
            makeup_params.blush_color = BLUSH_PRESETS[blush_idx][1]
        elif key == ord('3'):
            eyebrow_idx = (eyebrow_idx + 1) % len(EYEBROW_PRESETS)
            makeup_params.eyebrow_color = EYEBROW_PRESETS[eyebrow_idx][1]

    cap.release()
    detector.release()
    cv2.destroyAllWindows()
    print("\nKraj.")

if __name__ == "__main__":
    main()

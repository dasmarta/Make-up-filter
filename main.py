import cv2
import time
from landmark_detector import FaceLandmarkDetector

def main():
    detector = FaceLandmarkDetector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera nije pronadjena!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    print("Tipke: Q=izlaz, S=spremi JSON, D=debug")

    show_debug = True
    last_lm    = None
    fps, t0, n = 0, time.time(), 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        lm = detector.process_frame(frame)
        last_lm = lm # pamćenje

        display = detector.draw_debug(frame, lm) if show_debug else frame.copy()

        n += 1
        if time.time() - t0 >= 1.0:
            fps = n / (time.time() - t0)
            n, t0 = 0, time.time()

        status = "LICE DETEKTIRANO" if lm.face_detected else "NEMA LICA"
        color  = (0, 220, 100) if lm.face_detected else (0, 60, 220)
        cv2.putText(display, status, (10, display.shape[0]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(display, f"FPS: {fps:.1f}", (display.shape[1]-110, 30),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("Osoba [Q=izlaz]", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            if last_lm and last_lm.face_detected:
                fn = f"landmarks_{int(time.time())}.json"
                with open(fn, 'w') as f:
                    f.write(last_lm.to_json())
                print(f"SAVED {fn}")
            else:
                print("Nema lica za snimanje.")
        elif key == ord('d'):
            show_debug = not show_debug

    cap.release()
    detector.release()
    cv2.destroyAllWindows()
    print("Kraj.")

if __name__ == "__main__":
    main()
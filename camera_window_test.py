import cv2
from pathlib import Path


def open_camera(source: int) -> cv2.VideoCapture:
    # Try DirectShow first on Windows for more stable webcam behavior.
    candidates = [
        ("DSHOW", cv2.CAP_DSHOW),
        ("MSMF", cv2.CAP_MSMF),
        ("ANY", cv2.CAP_ANY),
    ]
    for name, backend in candidates:
        cap = cv2.VideoCapture(source, backend)
        if cap.isOpened():
            print(f"[INFO] Camera opened with backend={name}")
            return cap
        cap.release()
    return cv2.VideoCapture(source)


def main() -> None:
    print("[INFO] Starting camera window test...")
    print("[INFO] A window named 'Test Window' should appear.")
    print("[INFO] Press ESC to close.")

    cap = open_camera(0)
    if not cap.isOpened():
        print("[ERROR] Camera failed to open (VideoCapture(0)).")
        print("[HINT] Close other apps using camera (Zoom/Teams/Browser) and retry.")
        return

    cv2.namedWindow("Test Window", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Test Window", 960, 540)
    cv2.moveWindow("Test Window", 40, 40)
    try:
        cv2.setWindowProperty("Test Window", cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

    snapshot_dir = Path("debug_snapshots")
    snapshot_dir.mkdir(exist_ok=True)

    frame_idx = 0
    while True:
        frame_idx += 1
        ret, frame = cap.read()
        if frame_idx % 30 == 0:
            print(f"Frame received: {ret} | frame={frame_idx}")

        if not ret:
            print("[ERROR] Camera read failed.")
            break

        cv2.putText(
            frame,
            "CAMERA ACTIVE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Test Window", frame)

        # Save periodic snapshots to confirm camera output even if the window is hidden.
        if frame_idx % 120 == 0:
            out_path = snapshot_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            print(f"[SNAPSHOT] Saved {out_path}")

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("[INFO] ESC pressed. Exiting.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[DONE] Camera released and windows closed.")


if __name__ == "__main__":
    main()

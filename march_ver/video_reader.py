# =============================================================================
# video_reader.py
# =============================================================================
# Thin wrapper around cv2.VideoCapture.
#
# Responsibilities:
#   - Open the video file and raise a clear error if it can't be opened.
#   - Expose frame dimensions, FPS, and total frame count.
#   - Provide a generator so the main loop can do:
#       for frame in reader.frames():
#           ...
#   - Provide a seek() method so the self-test can jump to any frame.
# =============================================================================

import cv2


class VideoReader:
    """Wraps cv2.VideoCapture with convenience properties and iteration."""

    def __init__(self, path: str):
        self.path = path
        self.cap  = cv2.VideoCapture(path)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"VideoReader: cannot open '{path}'.\n"
                "Check that the path is correct and the file is not corrupted."
            )

    # ------------------------------------------------------------------
    # Properties — read once from the capture object
    # ------------------------------------------------------------------

    @property
    def width(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def fps(self) -> float:
        return self.cap.get(cv2.CAP_PROP_FPS)

    @property
    def frame_count(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def duration_seconds(self) -> float:
        if self.fps > 0:
            return self.frame_count / self.fps
        return 0.0

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def read(self):
        """Read one frame.  Returns (True, frame) or (False, None)."""
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        return True, frame

    def frames(self):
        """Generator — yields every frame until the video ends."""
        while True:
            ret, frame = self.read()
            if not ret:
                break
            yield frame

    def seek(self, frame_index: int):
        """Jump to a specific frame index (0-based)."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    def release(self):
        self.cap.release()

    # ------------------------------------------------------------------
    # String representation for quick debugging
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"VideoReader('{self.path}'  "
            f"{self.width}×{self.height}  "
            f"{self.fps:.1f} fps  "
            f"{self.frame_count} frames  "
            f"{self.duration_seconds:.1f} s)"
        )


# =============================================================================
# SELF-TEST
# Run:  python video_reader.py
#
# What to check:
#   1. No error on open — confirms path is correct.
#   2. Width, height, fps, frame_count print sensible values.
#   3. Three sample frames displayed — confirms the video is readable.
#      Press any key to advance through the three frames.
#      Press Q or ESC to quit early.
# =============================================================================
if __name__ == "__main__":
    import sys
    from config import VIDEO_PATH

    print("\n" + "="*52)
    print("  video_reader.py — self-test")
    print("="*52)

    # ── Step 1: open ──────────────────────────────────────────────────
    try:
        reader = VideoReader(VIDEO_PATH)
    except RuntimeError as e:
        print(f"\n  ERROR: {e}")
        sys.exit(1)

    print(f"\n  Opened: {reader}")

    # ── Step 2: print metadata ────────────────────────────────────────
    print(f"\n  Width         : {reader.width} px")
    print(f"  Height        : {reader.height} px")
    print(f"  FPS           : {reader.fps:.2f}")
    print(f"  Total frames  : {reader.frame_count}")
    print(f"  Duration      : {reader.duration_seconds:.1f} s")

    # Sanity checks
    checks = [
        (reader.width  > 0,  "Width must be > 0"),
        (reader.height > 0,  "Height must be > 0"),
        (reader.fps    > 0,  "FPS must be > 0"),
        (reader.frame_count > 0, "Frame count must be > 0"),
    ]
    print("\n  [ Sanity Checks ]")
    all_ok = True
    for passed, msg in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {msg}")
        if not passed:
            all_ok = False

    if not all_ok:
        print("\n  One or more checks FAILED.")
        reader.release()
        sys.exit(1)

    # ── Step 3: display 3 sample frames ──────────────────────────────
    # Sample from start, middle, and near end of the video.
    sample_indices = [
        0,
        reader.frame_count // 2,
        max(0, reader.frame_count - 30),
    ]
    labels = ["Frame 0 (start)", "Frame (middle)", "Frame (near end)"]

    print(f"\n  Displaying 3 sample frames.")
    print("  Press any key to advance.  Q or ESC to quit.\n")

    for idx, label in zip(sample_indices, labels):
        reader.seek(idx)
        ok, frame = reader.read()
        if not ok:
            print(f"  WARNING: could not read frame {idx}")
            continue

        # Stamp the label onto the frame so you know which one it is
        display = frame.copy()
        cv2.putText(
            display, f"{label}  (frame {idx})",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )

        # Resize to fit on screen if necessary
        h, w = display.shape[:2]
        max_w, max_h = 1280, 720
        if w > max_w or h > max_h:
            scale   = min(max_w / w, max_h / h)
            display = cv2.resize(display, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_AREA)

        cv2.imshow("video_reader — self-test", display)
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord("q")):   # ESC or Q
            print("  Quit early.")
            break

    cv2.destroyAllWindows()
    reader.release()

    print("  Self-test complete.\n")
    sys.exit(0)
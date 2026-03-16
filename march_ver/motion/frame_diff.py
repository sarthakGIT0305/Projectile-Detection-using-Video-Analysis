# =============================================================================
# motion/frame_diff.py
# =============================================================================
# Three-frame temporal difference for detecting moving objects.
#
# WHY 3-FRAME DIFFERENCE?
# -----------------------
# Simple 2-frame difference (frame[t] − frame[t−1]) produces thick "ghost"
# edges because both the old AND new position of a moving object light up.
# The 3-frame method ANDs two consecutive diffs:
#
#     diff1 = |frame[t] − frame[t − gap]|
#     diff2 = |frame[t − gap] − frame[t − 2*gap]|
#     mask  = threshold(diff1) AND threshold(diff2)
#
# Only pixels that changed in BOTH intervals survive — the moving object
# itself.  Static edges and single-frame sensor noise are eliminated.
#
# FRAME_DIFF_GAP controls the temporal spacing:
#   Gap = 1 → consecutive frames (picks up slow motion — clothing sway, etc.)
#   Gap = 2 → skip 1 frame (slow sway cancels; fast ball remains)
#   Gap = 3 → only very fast objects
#
# PARAMETERS (set in config.py):
#   FRAME_DIFF_GAP       — temporal spacing between compared frames
#   FRAME_DIFF_THRESHOLD — intensity difference required to count as motion
# =============================================================================

import cv2
import numpy as np
from collections import deque


class FrameDiffer:
    """Maintains a rolling buffer of grayscale frames and produces a binary
    motion mask using 3-frame differencing."""

    def __init__(self, gap: int = 2, threshold: int = 20):
        self.gap       = gap
        self.threshold = threshold
        # We need frames at indices: t, t-gap, t-2*gap
        # So buffer length = 2 * gap + 1
        self._buffer = deque(maxlen=2 * gap + 1)

    def update(self, gray: np.ndarray) -> np.ndarray | None:
        """Feed the next grayscale frame.

        Returns
        -------
        np.ndarray or None
            Binary mask (0/255) if enough frames have been buffered,
            otherwise None.
        """
        self._buffer.append(gray.copy())

        if len(self._buffer) < 2 * self.gap + 1:
            return None

        cur  = self._buffer[-1]
        mid  = self._buffer[-(self.gap + 1)]
        old  = self._buffer[0]

        diff1 = cv2.absdiff(cur, mid)
        diff2 = cv2.absdiff(mid, old)

        _, mask1 = cv2.threshold(diff1, self.threshold, 255, cv2.THRESH_BINARY)
        _, mask2 = cv2.threshold(diff2, self.threshold, 255, cv2.THRESH_BINARY)

        return cv2.bitwise_and(mask1, mask2)

    def reset(self):
        """Clear the frame buffer (e.g. after a seek)."""
        self._buffer.clear()


# =============================================================================
# SELF-TEST
# Run:  python motion/frame_diff.py
#
# What you will see — THREE panels side by side:
#   Left   — CLAHE-enhanced grayscale input
#   Middle — |diff1| raw absolute difference (bright = motion)
#   Right  — binary mask after 3-frame AND + threshold
#
# Interactive controls:
#   SPACE                   — play / pause
#   LEFT / RIGHT (or A / D) — step one frame (while paused)
#   + / -                   — raise / lower threshold
#   G                       — cycle gap: 1 → 2 → 3 → 1
#   Q / ESC                 — quit
#
# What to look for:
#   The ball should appear as a white blob in the right panel while most
#   background motion (swaying clothes, wind) is suppressed.
# =============================================================================
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from config        import (VIDEO_PATH, ENABLE_CLAHE, ENABLE_ROI,
                               CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID,
                               ROI_PERCENT, ROI_ANCHOR_X, ROI_ANCHOR_Y,
                               FRAME_DIFF_GAP, FRAME_DIFF_THRESHOLD,
                               DISPLAY_MAX_W, DISPLAY_MAX_H)
    from video_reader  import VideoReader
    from preprocessing.clahe_gray import to_gray
    from preprocessing.roi        import apply_roi

    print("\n" + "="*52)
    print("  motion/frame_diff.py — self-test")
    print("="*52)
    print(f"\n  FRAME_DIFF_GAP       = {FRAME_DIFF_GAP}")
    print(f"  FRAME_DIFF_THRESHOLD = {FRAME_DIFF_THRESHOLD}")
    print()
    print("  Controls:")
    print("    SPACE           — play / pause")
    print("    LEFT/RIGHT      — step one frame (paused)")
    print("    + / -           — raise / lower threshold")
    print("    G               — cycle gap (1/2/3)")
    print("    Q / ESC         — quit\n")

    try:
        reader = VideoReader(VIDEO_PATH)
    except RuntimeError as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    total    = reader.frame_count
    gap      = FRAME_DIFF_GAP
    thresh   = FRAME_DIFF_THRESHOLD
    paused   = False
    differ   = FrameDiffer(gap=gap, threshold=thresh)

    for frame_no, frame in enumerate(reader.frames()):
        # Preprocessing
        if ENABLE_ROI:
            frame, _ = apply_roi(frame, ROI_PERCENT, ROI_ANCHOR_X, ROI_ANCHOR_Y)
        gray = to_gray(frame, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID, ENABLE_CLAHE)

        mask = differ.update(gray)

        if mask is None:
            continue

        # Build display
        left   = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        right  = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Show raw |diff| for middle panel
        if len(differ._buffer) >= differ.gap + 1:
            raw_diff = cv2.absdiff(gray, differ._buffer[-(differ.gap + 1)])
            middle   = cv2.cvtColor(raw_diff, cv2.COLOR_GRAY2BGR)
        else:
            middle = left.copy()

        h = left.shape[0]
        for panel, label in [
            (left,   "CLAHE gray"),
            (middle, f"|diff|  gap={gap}"),
            (right,  f"Mask  thresh={thresh}"),
        ]:
            cv2.putText(panel, label,
                        (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (0, 255, 180), 1)

        combined = np.hstack([left, middle, right])
        ch, cw = combined.shape[:2]
        if cw > DISPLAY_MAX_W or ch > DISPLAY_MAX_H:
            scale    = min(DISPLAY_MAX_W / cw, DISPLAY_MAX_H / ch)
            combined = cv2.resize(combined, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_AREA)

        cv2.putText(combined,
                    f"Frame {frame_no}/{total-1}   gap={gap}   thresh={thresh}",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 100), 1)

        cv2.imshow("frame_diff.py — self-test", combined)
        wait = 0 if paused else 30
        key  = cv2.waitKey(wait) & 0xFF

        if key in (27, ord("q")):
            break
        elif key == ord(" "):
            paused = not paused
            print(f"  {'Paused' if paused else 'Playing'}")
        elif key == 83 or key == ord("d"):
            paused = True
        elif key == 81 or key == ord("a"):
            paused = True
        elif key in (ord("+"), ord("=")):
            thresh = min(thresh + 2, 100)
            differ.threshold = thresh
            print(f"  threshold → {thresh}")
        elif key == ord("-"):
            thresh = max(thresh - 2, 1)
            differ.threshold = thresh
            print(f"  threshold → {thresh}")
        elif key == ord("g"):
            gap = (gap % 3) + 1
            differ = FrameDiffer(gap=gap, threshold=thresh)
            reader.seek(max(0, frame_no - 2 * gap))
            print(f"  gap → {gap}  (buffer reset)")

    cv2.destroyAllWindows()
    reader.release()
    print(f"\n  Final settings:  gap={gap}  threshold={thresh}")
    print("  Update config.py if you changed values.\n")
    print("  Self-test complete.\n")

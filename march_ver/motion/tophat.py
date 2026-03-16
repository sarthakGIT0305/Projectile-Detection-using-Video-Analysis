# =============================================================================
# motion/tophat.py
# =============================================================================
# Applies a morphological top-hat (or black-hat, or both) spatial filter to
# a grayscale frame.
#
# WHY TOP-HAT?
# ------------
# A white top-hat extracts bright features SMALLER than the structuring
# element (kernel).  Since the ball is only ~4–8 px wide at 50 m, a 15–19 px
# kernel lets the ball through while suppressing everything larger — walls,
# railings, clothing.  It is the single most important spatial filter for
# isolating tiny bright objects on a complex background.
#
# A black top-hat does the inverse — it extracts small DARK features against
# a bright background.  When TOPHAT_MODE = "both", we run both and take the
# pixel-wise maximum, so the filter works regardless of whether the ball is
# lighter or darker than its local surroundings.
#
# PARAMETERS (set in config.py):
#   TOPHAT_MODE        — "white", "black", or "both"
#   TOPHAT_KERNEL_SIZE — (w, h) of the elliptical structuring element.
#                        Must be LARGER than the ball, SMALLER than background.
#   TOPHAT_THRESHOLD   — after the top-hat, pixels above this value become
#                        foreground (255).  Raise to reduce noise, lower to
#                        catch a faint ball.
# =============================================================================

import cv2
import numpy as np


def apply_tophat(gray: np.ndarray,
                 kernel_size: tuple = (17, 17),
                 threshold: int    = 12,
                 mode: str        = "both") -> np.ndarray:
    """Apply morphological top-hat filtering and threshold the result.

    Parameters
    ----------
    gray        : np.ndarray   8-bit single-channel grayscale image.
    mode        : str          "white", "black", or "both".
    kernel_size : tuple        Size of the elliptical structuring element.
    threshold   : int          Binary threshold applied to the filtered image.

    Returns
    -------
    np.ndarray   8-bit binary mask (0 or 255).
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)

    if mode == "white":
        filtered = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    elif mode == "black":
        filtered = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    elif mode == "both":
        white = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        black = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        filtered = cv2.max(white, black)

    else:
        raise ValueError(f"tophat: unknown mode '{mode}'. "
                         f"Expected 'white', 'black', or 'both'.")

    _, mask = cv2.threshold(filtered, threshold, 255, cv2.THRESH_BINARY)
    return mask


# =============================================================================
# SELF-TEST
# Run:  python motion/tophat.py
#
# What you will see — THREE panels side by side:
#   Left   — CLAHE-enhanced grayscale input
#   Middle — raw top-hat response (bright = small features detected)
#   Right  — binary mask after thresholding
#
# Interactive controls:
#   LEFT / RIGHT arrows (or A / D) — step through frames
#   T key                           — cycle mode: white → black → both
#   + / - keys                      — raise / lower threshold
#   Q or ESC                        — quit
#
# What to look for:
#   The ball should appear as a bright spot in the middle panel and a white
#   blob in the right panel.  Large background structures should be dark /
#   absent.  If the ball disappears, lower the threshold or shrink the kernel
#   in config.py.
# =============================================================================
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from config        import (VIDEO_PATH, ENABLE_CLAHE, ENABLE_ROI,
                               CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID,
                               ROI_PERCENT, ROI_ANCHOR_X, ROI_ANCHOR_Y,
                               TOPHAT_MODE, TOPHAT_KERNEL_SIZE,
                               TOPHAT_THRESHOLD,
                               DISPLAY_MAX_W, DISPLAY_MAX_H)
    from video_reader  import VideoReader
    from preprocessing.clahe_gray import to_gray
    from preprocessing.roi        import apply_roi

    print("\n" + "="*52)
    print("  motion/tophat.py — self-test")
    print("="*52)
    print(f"\n  TOPHAT_MODE        = {TOPHAT_MODE}")
    print(f"  TOPHAT_KERNEL_SIZE = {TOPHAT_KERNEL_SIZE}")
    print(f"  TOPHAT_THRESHOLD   = {TOPHAT_THRESHOLD}")
    print()
    print("  Controls:")
    print("    LEFT/RIGHT (or A/D) — step frames")
    print("    T                   — cycle mode (white/black/both)")
    print("    + / -               — raise / lower threshold")
    print("    Q / ESC             — quit\n")

    try:
        reader = VideoReader(VIDEO_PATH)
    except RuntimeError as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    total     = reader.frame_count
    idx       = 0
    step      = max(1, total // 50)
    mode      = TOPHAT_MODE
    thresh    = TOPHAT_THRESHOLD
    modes     = ["white", "black", "both"]

    while True:
        reader.seek(idx)
        ok, frame = reader.read()
        if not ok:
            break

        # Preprocessing
        if ENABLE_ROI:
            frame, _ = apply_roi(frame, ROI_PERCENT, ROI_ANCHOR_X, ROI_ANCHOR_Y)
        gray = to_gray(frame, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID, ENABLE_CLAHE)

        # Top-hat
        mask = apply_tophat(gray, mode=mode,
                            kernel_size=TOPHAT_KERNEL_SIZE,
                            threshold=thresh)

        # Build 3-panel display
        left   = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # Raw top-hat response (before threshold) for visualisation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, TOPHAT_KERNEL_SIZE)
        if mode == "white":
            raw = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        elif mode == "black":
            raw = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        else:
            w = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
            b = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            raw = cv2.max(w, b)
        middle = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        right  = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Panel labels
        h = left.shape[0]
        for panel, label in [
            (left,   "CLAHE gray input"),
            (middle, f"Top-hat raw  mode={mode}"),
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
                    f"Frame {idx}/{total-1}   mode={mode}   thresh={thresh}",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 100), 1)

        cv2.imshow("tophat.py — self-test", combined)
        key = cv2.waitKey(0) & 0xFF

        if key in (27, ord("q")):
            break
        elif key == 83 or key == ord("d"):
            idx = min(idx + step, total - 1)
        elif key == 81 or key == ord("a"):
            idx = max(idx - step, 0)
        elif key == ord("t"):
            mode = modes[(modes.index(mode) + 1) % len(modes)]
            print(f"  mode → {mode}")
        elif key in (ord("+"), ord("=")):
            thresh = min(thresh + 2, 100)
            print(f"  threshold → {thresh}")
        elif key == ord("-"):
            thresh = max(thresh - 2, 1)
            print(f"  threshold → {thresh}")

    cv2.destroyAllWindows()
    reader.release()
    print(f"\n  Final settings:  mode={mode}  threshold={thresh}")
    print("  Update config.py if you changed values.\n")
    print("  Self-test complete.\n")

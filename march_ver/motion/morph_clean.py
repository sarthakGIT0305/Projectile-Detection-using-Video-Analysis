# =============================================================================
# motion/morph_clean.py
# =============================================================================
# Morphological denoising of the combined motion mask.
#
# WHY MORPHOLOGICAL CLEANING?
# ----------------------------
# After combining the three motion masks, the result may still contain:
#   - Isolated single-pixel noise (salt-and-pepper)
#   - Thin connecting "bridges" between blobs
#   - Small holes inside the ball blob
#
# A gentle morphological OPEN (erode → dilate) with a TINY kernel removes
# isolated noise pixels without destroying the ball's small blob.
#
# CRITICAL: The kernel MUST be small (2×2 or 3×3).
# A larger kernel (e.g. 5×5) will erase a 4–8 px ball entirely.
#
# An optional CLOSE (dilate → erode) after the OPEN fills small holes inside
# the ball blob, making it more solid for downstream contour detection.
#
# PARAMETERS (set in config.py):
#   MORPH_KERNEL_SIZE — (w, h) of the rectangular kernel.
#                       (2, 2) = minimal noise removal, safest for tiny objects
#                       (3, 3) = slightly more aggressive
# =============================================================================

import cv2
import numpy as np


def morph_clean(mask: np.ndarray,
                kernel_size: tuple = (2, 2)) -> np.ndarray:
    """Remove salt-and-pepper noise from a binary mask.

    Parameters
    ----------
    mask        : np.ndarray   8-bit binary mask (0 or 255).
    kernel_size : tuple        Size of the morphological kernel.

    Returns
    -------
    np.ndarray   Cleaned 8-bit binary mask.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    # Open: erode then dilate — kills isolated noise dots
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Close: dilate then erode — fills tiny holes inside blobs
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    return cleaned


# Alias for main.py compatibility
clean_mask = morph_clean


# =============================================================================
# SELF-TEST
# Run:  python motion/morph_clean.py
#
# What you will see — THREE panels side by side:
#   Left   — combined mask BEFORE morphological cleaning
#   Middle — cleaned mask AFTER open + close
#   Right  — difference (pixels removed shown in red, filled in green)
#
# Interactive controls:
#   SPACE           — play / pause
#   + / -           — increase / decrease kernel size
#   Q / ESC         — quit
#
# What to look for:
#   Isolated noise pixels should disappear (red in the diff panel).
#   The ball blob should remain intact — if it disappears, the kernel is
#   too large.  Reduce MORPH_KERNEL_SIZE in config.py.
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
                               FRAME_DIFF_GAP, FRAME_DIFF_THRESHOLD,
                               MOG2_HISTORY, MOG2_VAR_THRESHOLD,
                               MOG2_DETECT_SHADOWS,
                               MASK_COMBINE_MODE, MORPH_KERNEL_SIZE,
                               DISPLAY_MAX_W, DISPLAY_MAX_H)
    from video_reader          import VideoReader
    from preprocessing.clahe_gray import to_gray
    from preprocessing.roi        import apply_roi
    from tophat       import apply_tophat
    from frame_diff   import FrameDiffer
    from bg_sub       import BgSubtractor
    from mask_combine import combine_masks

    print("\n" + "="*52)
    print("  motion/morph_clean.py — self-test")
    print("="*52)
    print(f"\n  MORPH_KERNEL_SIZE = {MORPH_KERNEL_SIZE}")
    print()
    print("  Controls:")
    print("    SPACE   — play / pause")
    print("    + / -   — grow / shrink kernel")
    print("    Q / ESC — quit\n")

    try:
        reader = VideoReader(VIDEO_PATH)
    except RuntimeError as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    total    = reader.frame_count
    paused   = False
    ks       = MORPH_KERNEL_SIZE[0]   # keep square
    differ   = FrameDiffer(gap=FRAME_DIFF_GAP, threshold=FRAME_DIFF_THRESHOLD)
    bg_sub   = BgSubtractor(history=MOG2_HISTORY,
                            var_threshold=MOG2_VAR_THRESHOLD,
                            detect_shadows=MOG2_DETECT_SHADOWS)

    for frame_no, frame in enumerate(reader.frames()):
        # Preprocessing
        if ENABLE_ROI:
            frame, _ = apply_roi(frame, ROI_PERCENT, ROI_ANCHOR_X, ROI_ANCHOR_Y)
        gray = to_gray(frame, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID, ENABLE_CLAHE)

        # Individual masks
        th_mask   = apply_tophat(gray, mode=TOPHAT_MODE,
                                 kernel_size=TOPHAT_KERNEL_SIZE,
                                 threshold=TOPHAT_THRESHOLD)
        fd_mask   = differ.update(gray)
        mog2_mask = bg_sub.update(gray)

        if fd_mask is None:
            continue

        # Combine → clean
        raw_combined = combine_masks(th_mask, fd_mask, mog2_mask,
                                     mode=MASK_COMBINE_MODE)
        cleaned      = morph_clean(raw_combined, kernel_size=(ks, ks))

        # Diff visualisation: red = removed, green = filled
        diff_vis = np.zeros((*raw_combined.shape[:2], 3), dtype=np.uint8)
        removed  = cv2.bitwise_and(raw_combined,
                                    cv2.bitwise_not(cleaned))
        filled   = cv2.bitwise_and(cleaned,
                                    cv2.bitwise_not(raw_combined))
        diff_vis[:, :, 2] = removed    # red channel
        diff_vis[:, :, 1] = filled     # green channel

        # Build display
        left   = cv2.cvtColor(raw_combined, cv2.COLOR_GRAY2BGR)
        middle = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        right  = diff_vis

        h = left.shape[0]
        for panel, label in [
            (left,   "Before cleaning"),
            (middle, f"After  kernel=({ks},{ks})"),
            (right,  "Diff (red=removed, grn=filled)"),
        ]:
            cv2.putText(panel, label,
                        (4, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (0, 255, 180), 1)

        display = np.hstack([left, middle, right])
        dh, dw = display.shape[:2]
        if dw > DISPLAY_MAX_W or dh > DISPLAY_MAX_H:
            scale   = min(DISPLAY_MAX_W / dw, DISPLAY_MAX_H / dh)
            display = cv2.resize(display, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_AREA)

        cv2.putText(display,
                    f"Frame {frame_no}/{total-1}   kernel=({ks},{ks})",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 100), 1)

        cv2.imshow("morph_clean.py — self-test", display)
        wait = 0 if paused else 30
        key  = cv2.waitKey(wait) & 0xFF

        if key in (27, ord("q")):
            break
        elif key == ord(" "):
            paused = not paused
            print(f"  {'Paused' if paused else 'Playing'}")
        elif key in (ord("+"), ord("=")):
            ks = min(ks + 1, 7)
            print(f"  kernel → ({ks},{ks})")
        elif key == ord("-"):
            ks = max(ks - 1, 1)
            print(f"  kernel → ({ks},{ks})")

    cv2.destroyAllWindows()
    reader.release()
    print(f"\n  Final kernel: ({ks},{ks})")
    print("  Update MORPH_KERNEL_SIZE in config.py if you changed it.\n")
    print("  Self-test complete.\n")

# =============================================================================
# motion/mask_combine.py
# =============================================================================
# Merges the three motion masks (top-hat, frame-diff, MOG2) into one.
#
# WHY COMBINE?
# ------------
# Each detector has different strengths:
#   - Top-hat   isolates small bright/dark features (spatial — shape-based)
#   - Frame diff detects fast motion (temporal — movement-based)
#   - MOG2      separates foreground from learned background (statistical)
#
# No single detector is reliable alone.  By combining them we can require
# that a pixel looks like a small object AND has moved AND doesn't belong to
# the learned background.  This dramatically reduces false positives.
#
# COMBINATION MODES (set in config.py → MASK_COMBINE_MODE):
#   "or"              — any detector fires → foreground  (most sensitive)
#   "and"             — ALL detectors must agree          (cleanest, may miss)
#   "tophat_and_any"  — top-hat AND (frame_diff OR MOG2) (recommended start)
#   "motion_primary"  — (frame_diff OR MOG2) AND top-hat boost
#                       motion-first approach; top-hat acts as spatial filter
# =============================================================================

import cv2
import numpy as np


def combine_masks(tophat_mask: np.ndarray | None,
                  fdiff_mask:  np.ndarray | None,
                  mog2_mask:   np.ndarray | None,
                  mode: str = "tophat_and_any") -> np.ndarray:
    """Merge up to three binary masks into one final motion mask.

    Parameters
    ----------
    tophat_mask : np.ndarray or None   Binary mask from top-hat filter.
    fdiff_mask  : np.ndarray or None   Binary mask from 3-frame difference.
    mog2_mask   : np.ndarray or None   Binary mask from MOG2 bg subtraction.
    mode        : str                  Combination strategy (see module doc).

    Returns
    -------
    np.ndarray   8-bit binary mask (0 or 255).
    """
    # Collect the masks that are actually present
    masks = []
    if tophat_mask is not None:
        masks.append(tophat_mask)
    if fdiff_mask is not None:
        masks.append(fdiff_mask)
    if mog2_mask is not None:
        masks.append(mog2_mask)

    if not masks:
        raise ValueError("combine_masks: at least one mask must be provided.")

    # If only one mask, return it directly
    if len(masks) == 1:
        return masks[0].copy()

    # Create zero mask with same shape for missing inputs
    h, w  = masks[0].shape[:2]
    zero  = np.zeros((h, w), dtype=np.uint8)
    th    = tophat_mask if tophat_mask is not None else zero
    fd    = fdiff_mask  if fdiff_mask  is not None else zero
    mg    = mog2_mask   if mog2_mask   is not None else zero

    if mode == "or":
        result = cv2.bitwise_or(th, fd)
        result = cv2.bitwise_or(result, mg)

    elif mode == "and":
        result = cv2.bitwise_and(th, fd)
        result = cv2.bitwise_and(result, mg)

    elif mode == "tophat_and_any":
        temporal = cv2.bitwise_or(fd, mg)
        result   = cv2.bitwise_and(th, temporal)

    elif mode == "motion_primary":
        temporal = cv2.bitwise_or(fd, mg)
        result   = cv2.bitwise_and(temporal, th)

    else:
        raise ValueError(
            f"combine_masks: unknown mode '{mode}'. "
            f"Expected 'or', 'and', 'tophat_and_any', or 'motion_primary'."
        )

    return result


# =============================================================================
# SELF-TEST
# Run:  python motion/mask_combine.py
#
# What you will see — FIVE panels:
#   1. CLAHE gray input
#   2. Top-hat mask
#   3. Frame-diff mask
#   4. MOG2 mask
#   5. Combined mask (using current mode)
#
# Interactive controls:
#   SPACE           — play / pause
#   M               — cycle combination mode
#   Q / ESC         — quit
#
# What to look for:
#   The combined mask (rightmost) should contain only the ball and minimal
#   noise.  Compare it against the individual masks to see which detector
#   contributes noise and which contributes signal.
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
                               MASK_COMBINE_MODE,
                               DISPLAY_MAX_W, DISPLAY_MAX_H)
    from video_reader          import VideoReader
    from preprocessing.clahe_gray import to_gray
    from preprocessing.roi        import apply_roi
    from tophat     import apply_tophat
    from frame_diff import FrameDiffer
    from bg_sub     import BgSubtractor

    print("\n" + "="*52)
    print("  motion/mask_combine.py — self-test")
    print("="*52)
    print(f"\n  MASK_COMBINE_MODE = {MASK_COMBINE_MODE}")
    print()
    print("  Controls:")
    print("    SPACE   — play / pause")
    print("    M       — cycle combination mode")
    print("    Q / ESC — quit\n")

    try:
        reader = VideoReader(VIDEO_PATH)
    except RuntimeError as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    total    = reader.frame_count
    paused   = False
    mode     = MASK_COMBINE_MODE
    modes    = ["or", "and", "tophat_and_any", "motion_primary"]

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

        # Combine
        combined_mask = combine_masks(th_mask, fd_mask, mog2_mask, mode=mode)

        # Build 5-panel display
        panels = [
            (cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),          "CLAHE gray"),
            (cv2.cvtColor(th_mask, cv2.COLOR_GRAY2BGR),       "Top-hat"),
            (cv2.cvtColor(fd_mask, cv2.COLOR_GRAY2BGR),       "Frame diff"),
            (cv2.cvtColor(mog2_mask, cv2.COLOR_GRAY2BGR),     "MOG2"),
            (cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR), f"Combined ({mode})"),
        ]

        h = panels[0][0].shape[0]
        for panel, label in panels:
            cv2.putText(panel, label,
                        (4, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (0, 255, 180), 1)

        display = np.hstack([p for p, _ in panels])
        dh, dw = display.shape[:2]
        if dw > DISPLAY_MAX_W or dh > DISPLAY_MAX_H:
            scale   = min(DISPLAY_MAX_W / dw, DISPLAY_MAX_H / dh)
            display = cv2.resize(display, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_AREA)

        cv2.putText(display,
                    f"Frame {frame_no}/{total-1}   mode={mode}",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 100), 1)

        cv2.imshow("mask_combine.py — self-test", display)
        wait = 0 if paused else 30
        key  = cv2.waitKey(wait) & 0xFF

        if key in (27, ord("q")):
            break
        elif key == ord(" "):
            paused = not paused
            print(f"  {'Paused' if paused else 'Playing'}")
        elif key == ord("m"):
            mode = modes[(modes.index(mode) + 1) % len(modes)]
            print(f"  mode → {mode}")

    cv2.destroyAllWindows()
    reader.release()
    print(f"\n  Final mode: {mode}")
    print("  Update MASK_COMBINE_MODE in config.py if you changed it.\n")
    print("  Self-test complete.\n")

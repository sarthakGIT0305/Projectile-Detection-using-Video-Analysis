
# =============================================================================
# preprocessing/clahe_gray.py
# =============================================================================
# Converts a BGR frame to grayscale with optional CLAHE contrast enhancement.
#
# WHY CLAHE?
# -----------
# At 50 m a tennis ball is only 3–8 pixels wide.  Its pixel values may differ
# from the surrounding background by as little as 8–20 intensity counts out
# of 255.  Standard grayscale conversion keeps those values small and the ball
# is easily swamped by noise.
#
# CLAHE (Contrast Limited Adaptive Histogram Equalization) works on small
# local tiles of the image independently, boosting the contrast of subtle
# intensity differences within each tile.  A faint ball that was barely 10
# counts brighter than its background can become 40–60 counts brighter after
# CLAHE — making it far more detectable by both frame-differencing and the
# top-hat filter downstream.
#
# We apply CLAHE to the L (lightness) channel of LAB colour space, not to
# the raw grayscale image.  This matters because:
#   - LAB separates luminance from colour, so the enhancement does not shift
#     hues or saturate colours unnaturally.
#   - After enhancing L we convert back to BGR → gray, giving a grayscale
#     image whose contrast has been boosted without colour artefacts.
#
# PARAMETERS (set in config.py):
#   CLAHE_CLIP_LIMIT  — how aggressively contrast is amplified per tile.
#                       1.0 = mild,  3.0 = moderate (default),  5.0 = strong.
#                       Too high → noise is amplified alongside the signal.
#   CLAHE_TILE_GRID   — size of each local tile (e.g. (8,8) means the frame
#                       is divided into an 8×8 grid of tiles).
#                       Smaller tiles = more local adaptation.
#                       Very small tiles (2×2) can over-enhance noise patches.
# =============================================================================

import cv2
import numpy as np


def to_gray(frame: np.ndarray,
            clahe_clip: float = 3.0,
            clahe_tile: tuple  = (8, 8),
            use_clahe: bool    = True) -> np.ndarray:
    """Convert BGR frame to grayscale, optionally with CLAHE enhancement.

    Parameters
    ----------
    frame      : np.ndarray   Input BGR image.
    clahe_clip : float        CLAHE clip limit.
    clahe_tile : tuple        CLAHE tile grid size.
    use_clahe  : bool         If False, returns plain grayscale (no CLAHE).
                              Controlled by ENABLE_CLAHE in config.py.

    Returns
    -------
    np.ndarray   8-bit single-channel grayscale image.
    """
    if not use_clahe:
        # Plain conversion — fast path when CLAHE is disabled
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── CLAHE on L channel of LAB ─────────────────────────────────────
    lab                  = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a, b      = cv2.split(lab)

    clahe                = cv2.createCLAHE(clipLimit=clahe_clip,
                                            tileGridSize=clahe_tile)
    l_enhanced           = clahe.apply(l_channel)

    lab_enhanced         = cv2.merge((l_enhanced, a, b))
    bgr_enhanced         = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    gray                 = cv2.cvtColor(bgr_enhanced, cv2.COLOR_BGR2GRAY)
    return gray


# =============================================================================
# SELF-TEST
# Run:  python preprocessing/clahe_gray.py
#
# What you will see — THREE panels side by side:
#   Left   — original BGR frame (colour reference)
#   Middle — plain grayscale (no CLAHE)
#   Right  — CLAHE-enhanced grayscale
#
# What to look for:
#   Compare middle vs right in regions where the ball would appear.
#   The right panel should show higher local contrast — textures and edges
#   that were flat / washed-out in the middle panel become more distinct.
#
#   IMPORTANT: you are NOT looking for a dramatic overall brightness change.
#   CLAHE is subtle.  Look for:
#     • Faint edges becoming crisper
#     • Slightly dark objects in bright backgrounds becoming more visible
#     • Previously flat / grey areas showing more texture detail
#
# Interactive controls:
#   LEFT / RIGHT arrows (or A / D) — step through frames
#   C key                           — toggle CLAHE on/off to compare live
#   + / - keys                      — increase / decrease CLAHE clip limit
#   Q or ESC                        — quit
#
# How to tune:
#   If CLAHE is amplifying noise patches (speckles appearing in flat areas),
#   lower CLAHE_CLIP_LIMIT in config.py (try 1.5 or 2.0).
#   If the enhancement looks too subtle, raise it (try 4.0 or 5.0).
# =============================================================================
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from config       import (VIDEO_PATH, ENABLE_CLAHE,
                               CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID,
                               DISPLAY_MAX_W, DISPLAY_MAX_H)
    from video_reader import VideoReader

    print("\n" + "="*52)
    print("  preprocessing/clahe_gray.py — self-test")
    print("="*52)
    print(f"\n  ENABLE_CLAHE     = {ENABLE_CLAHE}")
    print(f"  CLAHE_CLIP_LIMIT = {CLAHE_CLIP_LIMIT}")
    print(f"  CLAHE_TILE_GRID  = {CLAHE_TILE_GRID}")
    print()
    print("  Controls:")
    print("    LEFT/RIGHT (or A/D) — step frames")
    print("    C                   — toggle CLAHE on/off")
    print("    + / -               — raise / lower clip limit")
    print("    Q / ESC             — quit\n")

    try:
        reader = VideoReader(VIDEO_PATH)
    except RuntimeError as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    total     = reader.frame_count
    idx       = 0
    step      = max(1, total // 50)
    clip      = CLAHE_CLIP_LIMIT
    tile      = CLAHE_TILE_GRID
    show_clahe = ENABLE_CLAHE

    def _build_display(frame, clip_val, tile_val, clahe_on):
        """Build the three-panel comparison image."""
        # Panel 1: original colour (convert to 3-ch if needed)
        left = frame.copy()

        # Panel 2: plain grayscale → back to BGR so hstack works
        plain_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        middle     = cv2.cvtColor(plain_gray, cv2.COLOR_GRAY2BGR)

        # Panel 3: CLAHE-enhanced grayscale
        if clahe_on:
            enhanced_gray = to_gray(frame, clahe_clip=clip_val,
                                    clahe_tile=tile_val, use_clahe=True)
        else:
            enhanced_gray = plain_gray   # same as middle when disabled
        right = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

        # Add panel labels at the bottom
        h = left.shape[0]
        for panel, label in [
            (left,   "Original (colour)"),
            (middle, "Plain grayscale"),
            (right,  f"CLAHE  clip={clip_val:.1f}  {'ON' if clahe_on else 'OFF'}"),
        ]:
            cv2.putText(panel, label,
                        (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (0, 255, 180), 1)

        return np.hstack([left, middle, right])

    while True:
        reader.seek(idx)
        ok, frame = reader.read()
        if not ok:
            break

        combined = _build_display(frame, clip, tile, show_clahe)

        # Resize to fit display
        ch, cw = combined.shape[:2]
        if cw > DISPLAY_MAX_W or ch > DISPLAY_MAX_H:
            scale    = min(DISPLAY_MAX_W / cw, DISPLAY_MAX_H / ch)
            combined = cv2.resize(combined, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_AREA)

        # Frame / settings overlay
        cv2.putText(combined,
                    f"Frame {idx}/{total-1}   clip={clip:.1f}   "
                    f"CLAHE={'ON' if show_clahe else 'OFF'}",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 100), 1)

        cv2.imshow("clahe_gray.py — self-test", combined)
        key = cv2.waitKey(0) & 0xFF

        if key in (27, ord("q")):
            break
        elif key == 83 or key == ord("d"):          # RIGHT / D
            idx = min(idx + step, total - 1)
        elif key == 81 or key == ord("a"):          # LEFT / A
            idx = max(idx - step, 0)
        elif key == ord("c"):                        # toggle CLAHE
            show_clahe = not show_clahe
            state = "ON" if show_clahe else "OFF"
            print(f"  CLAHE toggled {state}")
        elif key in (ord("+"), ord("=")):            # raise clip
            clip = min(clip + 0.5, 8.0)
            print(f"  clip limit → {clip:.1f}")
        elif key == ord("-"):                        # lower clip
            clip = max(clip - 0.5, 0.5)
            print(f"  clip limit → {clip:.1f}")

    cv2.destroyAllWindows()
    reader.release()
    print(f"\n  Final settings:  CLAHE_CLIP_LIMIT = {clip:.1f}")
    print("  Update config.py if you changed the clip limit.\n")
    print("  Self-test complete.\n")
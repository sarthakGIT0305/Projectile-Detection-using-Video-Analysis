# =============================================================================
# preprocessing/roi.py
# =============================================================================
# Crops the frame to a centered rectangle that is ROI_PERCENT of the full
# frame size.
#
# Why ROI?
#   If you know the ball will only fly through a specific zone of the frame
#   (e.g. the upper-left quarter), you can restrict processing to that zone.
#   This reduces false positives from irrelevant areas and speeds up all
#   downstream stages because every stage works on a smaller image.
#
# When to use ROI_PERCENT = 1.0 (full frame):
#   When you don't yet know where the ball will appear — leave it at 1.0
#   until you've watched the video and can identify the flight zone.
#
# The function returns both the cropped frame AND the crop offset (x, y)
# so that downstream code can map detections back to original frame coords.
# =============================================================================

import cv2
import numpy as np
X = 0.467
Y = 0.5

def apply_roi(frame: np.ndarray, percent: float,
              anchor_x: float = X, anchor_y: float = Y):
    """
    anchor_x, anchor_y are fractions (0.0 – 1.0) of the full frame.
    They define where the CENTER of the ROI box sits.

    Examples:
      anchor_x=0.5, anchor_y=0.5  →  centered             (default)
      anchor_x=0.0, anchor_y=0.0  →  top-left corner
      anchor_x=1.0, anchor_y=0.0  →  top-right corner
      anchor_x=0.3, anchor_y=0.2  →  upper-left zone
    """
    percent  = max(0.01, min(1.0, percent))
    anchor_x = max(0.0,  min(1.0, anchor_x))
    anchor_y = max(0.0,  min(1.0, anchor_y))

    h, w  = frame.shape[:2]
    new_w = int(w * percent)
    new_h = int(h * percent)

    # Center of the box in pixels
    cx = int(w * anchor_x)
    cy = int(h * anchor_y)

    # Top-left corner, clamped so the box never goes outside the frame
    x = max(0, min(cx - new_w // 2, w - new_w))
    y = max(0, min(cy - new_h // 2, h - new_h))

    cropped = frame[y : y + new_h, x : x + new_w]
    offset  = (x, y)
    return cropped, offset

def draw_roi_on_frame(frame, percent,
                      anchor_x=X, anchor_y=Y):
    percent  = max(0.01, min(1.0, percent))
    anchor_x = max(0.0,  min(1.0, anchor_x))
    anchor_y = max(0.0,  min(1.0, anchor_y))

    h, w  = frame.shape[:2]
    new_w = int(w * percent)
    new_h = int(h * percent)
    cx    = int(w * anchor_x)
    cy    = int(h * anchor_y)
    x     = max(0, min(cx - new_w // 2, w - new_w))
    y     = max(0, min(cy - new_h // 2, h - new_h))

    vis  = frame.copy()
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y : y + new_h, x : x + new_w] = 255
    dimmed = (vis * 0.35).astype(np.uint8)
    vis[mask == 0] = dimmed[mask == 0]
    cv2.rectangle(vis, (x, y), (x + new_w, y + new_h), (0, 255, 0), 2)
    cv2.putText(vis,
                f"ROI {percent*100:.0f}%  anchor=({anchor_x:.2f},{anchor_y:.2f})",
                (x + 6, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    return vis


# =============================================================================
# SELF-TEST
# Run:  python preprocessing/roi.py
#
# What you will see:
#   A window with TWO panels side by side:
#     Left  — original frame with ROI rectangle and dimmed surroundings
#     Right — the cropped ROI output (what the rest of the pipeline sees)
#
# Press LEFT / RIGHT arrow keys to step through frames.
# Press Q or ESC to quit.
#
# What to check:
#   1. The green rectangle on the left covers the area where the ball flies.
#   2. The right panel shows only that area — nothing important is cut off.
#   3. If ROI_PERCENT = 1.0 both panels look identical (expected).
#
# How to tune:
#   Change ROI_PERCENT in config.py and re-run to see the new crop.
# =============================================================================
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from config       import VIDEO_PATH, ROI_PERCENT, DISPLAY_MAX_W, DISPLAY_MAX_H
    from video_reader import VideoReader

    print("\n" + "="*52)
    print("  preprocessing/roi.py — self-test")
    print("="*52)
    print(f"\n  ROI_PERCENT = {ROI_PERCENT}  ({ROI_PERCENT*100:.0f}% of frame kept)")
    print("  LEFT / RIGHT arrow = step frames  |  Q / ESC = quit\n")

    try:
        reader = VideoReader(VIDEO_PATH)
    except RuntimeError as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    print(f"  Video: {reader.width}x{reader.height}  {reader.fps:.1f} fps  "
          f"{reader.frame_count} frames")

    total   = reader.frame_count
    idx     = 0
    step    = max(1, total // 50)   # arrow key jumps ~2% through video

    while True:
        reader.seek(idx)
        ok, frame = reader.read()
        if not ok:
            break

        # ── Build left panel: full frame + ROI rectangle overlay ──────
        left = draw_roi_on_frame(frame, ROI_PERCENT)

        # ── Build right panel: cropped result ─────────────────────────
        cropped, (ox, oy) = apply_roi(frame, ROI_PERCENT)

        # Pad right panel to match left panel height for side-by-side display
        h_full = left.shape[0]
        h_crop, w_crop = cropped.shape[:2]
        pad_top  = (h_full - h_crop) // 2
        pad_bot  = h_full - h_crop - pad_top
        right    = cv2.copyMakeBorder(
            cropped, pad_top, pad_bot, 0, 0,
            cv2.BORDER_CONSTANT, value=(30, 30, 30)
        )

        # Add panel labels
        cv2.putText(left,  "Original + ROI box",
                    (8, left.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(right, f"Cropped output  offset=({ox},{oy})",
                    (8, right.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # ── Combine side by side ──────────────────────────────────────
        combined = np.hstack([left, right])

        # Resize to fit display
        ch, cw   = combined.shape[:2]
        max_w    = DISPLAY_MAX_W
        max_h    = DISPLAY_MAX_H
        if cw > max_w or ch > max_h:
            scale    = min(max_w / cw, max_h / ch)
            combined = cv2.resize(combined, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_AREA)

        # Frame index overlay
        cv2.putText(combined, f"Frame {idx}/{total-1}",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 100), 1)

        cv2.imshow("roi.py — self-test  (arrows to step, Q to quit)", combined)
        key = cv2.waitKey(0) & 0xFF

        if key in (27, ord("q")):
            break
        elif key == 83 or key == ord("d"):   # RIGHT arrow or D
            idx = min(idx + step, total - 1)
        elif key == 81 or key == ord("a"):   # LEFT arrow or A
            idx = max(idx - step, 0)

    cv2.destroyAllWindows()
    reader.release()
    print("  Self-test complete.\n")
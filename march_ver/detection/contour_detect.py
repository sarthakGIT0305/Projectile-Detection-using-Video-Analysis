# =============================================================================
# detection/contour_detect.py
# =============================================================================
# Finds the boundary of every white blob in the cleaned binary mask.
#
# HOW CONTOUR DETECTION WORKS
# ----------------------------
# cv2.findContours traces the outer boundary of each connected white region
# in a binary image and returns a list of point sequences — one per blob.
# Each contour is a numpy array of (x,y) points forming a closed polygon
# around one blob.
#
# These contours are then passed to blob_filter.py which measures each one's
# area, circularity, and solidity to decide if it could be the ball.
#
# RETR_EXTERNAL vs RETR_LIST
# ---------------------------
# RETR_EXTERNAL — only returns the outermost contour of each blob.
#   If a blob has a hole inside it, the hole boundary is ignored.
#   This is what we want — we only care about the outer shape.
#
# RETR_LIST — returns ALL contours including inner hole boundaries.
#   Produces duplicate work for hollow blobs.  Not useful here.
#
# CHAIN_APPROX_SIMPLE
# -------------------
# Compresses horizontal, vertical, and diagonal segments so only their
# endpoints are stored.  A rectangle's contour needs only 4 points instead
# of hundreds.  Saves memory and speeds up downstream geometry calculations.
#
# MINIMUM CONTOUR AREA PRE-FILTER
# --------------------------------
# OpenCV may return thousands of 1-pixel contours from compression noise
# that survived all previous stages.  A fast pre-filter of area > 1 px²
# drops these immediately before any geometry is computed, keeping the
# blob_filter workload small.
# =============================================================================

import cv2
import numpy as np
from typing import List


def detect_contours(mask: np.ndarray,
                    min_area_prefilter: float = 1.5) -> List[np.ndarray]:
    """Find external contours in a binary mask.

    Parameters
    ----------
    mask               : np.ndarray   Binary mask (0 or 255).
    min_area_prefilter : float        Drop contours with area below this
                                      value immediately — cheap pre-screen
                                      before full blob_filter geometry.
                                      Keep at 1.5 to drop single-pixel hits.

    Returns
    -------
    list of np.ndarray   Each element is one contour (array of points).
    """
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if min_area_prefilter > 0:
        contours = [c for c in contours
                    if cv2.contourArea(c) >= min_area_prefilter]

    return list(contours)


# =============================================================================
# SELF-TEST
# Run:  python detection/contour_detect.py
#
# TWO-PANEL window (plays the video):
#   Left  — cleaned binary mask with all detected contours drawn in green
#   Right — original frame (ROI) with bounding boxes drawn for every contour
#           that passed the area pre-filter, regardless of shape
#
# What to look for:
#   Left panel: every white blob should have a green outline.
#   Right panel: green boxes should appear over anything the pipeline flagged.
#                At this stage there will be many false positives — that is
#                expected and normal.  blob_filter.py (next stage) removes them.
#
# Counter in top-left shows how many contours were found this frame.
# If it shows 0 constantly → earlier stages are too strict, go back and tune.
# If it shows 500+ constantly → earlier stages are too loose, tune thresholds.
# A healthy range while the ball is in flight: 5–50 contours per frame.
#
# Controls:
#   SPACE           — pause / resume
#   LEFT/RIGHT(A/D) — step frames (paused)
#   Q / ESC         — quit
# =============================================================================
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from config import (
        VIDEO_PATH,
        ENABLE_CLAHE, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID,
        ENABLE_TOPHAT, TOPHAT_KERNEL_SIZE, TOPHAT_THRESHOLD, TOPHAT_MODE,
        ENABLE_FRAME_DIFF, FRAME_DIFF_GAP, FRAME_DIFF_THRESHOLD,
        ENABLE_BG_SUB, MOG2_HISTORY, MOG2_VAR_THRESHOLD, MOG2_DETECT_SHADOWS,
        ENABLE_MORPH_CLEAN, MORPH_KERNEL_SIZE,
        MASK_COMBINE_MODE,
        ROI_PERCENT, ROI_ANCHOR_X, ROI_ANCHOR_Y,
        DISPLAY_MAX_W, DISPLAY_MAX_H,
    )
    from video_reader              import VideoReader
    from preprocessing.clahe_gray import to_gray
    from preprocessing.roi        import apply_roi
    from motion.tophat             import apply_tophat
    from motion.frame_diff         import FrameDiffer
    from motion.bg_sub             import BackgroundSubtractor
    from motion.mask_combine       import combine_masks
    from motion.morph_clean        import clean_mask

    print("\n" + "="*52)
    print("  detection/contour_detect.py — self-test")
    print("="*52)
    print("\n  Left panel : mask + contour outlines")
    print("  Right panel: frame + bounding boxes (pre blob_filter)")
    print("  Counter shows raw contour count per frame.")
    print("  Healthy range during ball flight: 5–50\n")
    print("  SPACE=pause  A/D=step  Q=quit\n")

    try:
        reader = VideoReader(VIDEO_PATH)
    except RuntimeError as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    differ  = FrameDiffer(FRAME_DIFF_GAP, FRAME_DIFF_THRESHOLD)
    bgs     = BackgroundSubtractor(MOG2_HISTORY, MOG2_VAR_THRESHOLD,
                                    MOG2_DETECT_SHADOWS)
    paused  = False
    idx     = 0

    MAX_CACHE = 3000
    print(f"  Pre-reading up to {MAX_CACHE} frames...")
    frames_cache = []
    reader.seek(0)
    for frame in reader.frames():
        frames_cache.append(frame)
        if len(frames_cache) >= MAX_CACHE:
            break
    cached_total = len(frames_cache)
    print(f"  Cached {cached_total} frames.\n")

    while idx < cached_total:
        frame        = frames_cache[idx]
        roi_frame, _ = apply_roi(frame, ROI_PERCENT, ROI_ANCHOR_X, ROI_ANCHOR_Y)
        gray         = to_gray(roi_frame, CLAHE_CLIP_LIMIT,
                                CLAHE_TILE_GRID, ENABLE_CLAHE)

        th_mask = apply_tophat(gray, TOPHAT_KERNEL_SIZE,
                                TOPHAT_THRESHOLD, TOPHAT_MODE) \
                  if ENABLE_TOPHAT else None
        df_mask = differ.update(gray)     if ENABLE_FRAME_DIFF else None
        mg_mask = bgs.apply(roi_frame)    if ENABLE_BG_SUB     else None

        combined = combine_masks(th_mask, df_mask, mg_mask, MASK_COMBINE_MODE)

        cleaned  = clean_mask(combined, MORPH_KERNEL_SIZE) \
                   if ENABLE_MORPH_CLEAN else combined

        contours = detect_contours(cleaned)
        n        = len(contours)

        # ── Left panel: mask + contour outlines ──────────────────────
        mask_vis = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(mask_vis, contours, -1, (0, 255, 0), 1)

        # ── Right panel: ROI frame + bounding boxes ───────────────────
        frame_vis = roi_frame.copy()
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame_vis, (x, y), (x+w, y+h), (0, 255, 0), 1)

        h_img = mask_vis.shape[0]
        for panel, label in [
            (mask_vis,  f"Mask + contours  ({n} found)"),
            (frame_vis, "Frame + bounding boxes  (pre blob_filter)"),
        ]:
            cv2.putText(panel, label, (6, h_img - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,255,180), 1)

        combined_disp = np.hstack([mask_vis, frame_vis])
        ch, cw = combined_disp.shape[:2]
        if cw > DISPLAY_MAX_W or ch > DISPLAY_MAX_H:
            scale         = min(DISPLAY_MAX_W / cw, DISPLAY_MAX_H / ch)
            combined_disp = cv2.resize(combined_disp, None,
                                        fx=scale, fy=scale,
                                        interpolation=cv2.INTER_AREA)

        status = "PAUSED" if paused else "PLAYING"
        cv2.putText(combined_disp,
                    f"Frame {idx}/{cached_total-1}  contours={n}  [{status}]",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,100), 1)

        cv2.imshow("contour_detect.py — self-test", combined_disp)
        wait = 0 if paused else 30
        key  = cv2.waitKey(wait) & 0xFF

        if key in (27, ord("q")):
            break
        elif key == ord(" "):
            paused = not paused
        elif key in (83, ord("d")):
            idx = min(idx + 1, cached_total - 1)
        elif key in (81, ord("a")):
            idx = max(idx - 1, 0)
            differ.reset()
        else:
            if not paused:
                idx += 1

    cv2.destroyAllWindows()
    reader.release()
    print("  Self-test complete.\n")
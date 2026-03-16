# =============================================================================
# detection/blob_filter.py
# =============================================================================
# Applies geometric tests to each contour and keeps only those that
# could plausibly be a tennis ball.
#
# THE FOUR FILTERS — in order of computational cost (cheapest first)
# -------------------------------------------------------------------
#
# 1. AREA  (cheapest — one function call)
#    cv2.contourArea() returns the number of pixels² enclosed by the contour.
#    At 50 m the ball subtends ~4–16 px².  Single-pixel noise is 1 px².
#    Large objects like people are thousands of px².
#    This single filter removes ~90% of contours immediately.
#
# 2. ASPECT RATIO  (cheap — one boundingRect call)
#    Width / height of the bounding rectangle.  A tennis ball is roughly
#    circular so the ratio stays near 1.0.  Even with motion blur it rarely
#    exceeds 3:1.  Elongated blobs (railing shadows, cable edges) are rejected.
#
# 3. CIRCULARITY  (moderate — needs perimeter)
#    Score = 4π × area / perimeter²
#    Perfect circle = 1.0.  The formula penalises jagged or complex outlines.
#    Ball: ~0.6–1.0 (motion blur makes it slightly oval but still smooth).
#    Irregular blobs (tree edges, fabric folds): ~0.1–0.4.
#
# 4. SOLIDITY  (most expensive — needs convex hull)
#    Score = contour_area / convex_hull_area
#    A convex shape (ball) scores ~0.85–1.0.
#    A concave shape (crumpled cloth, forked branch) scores much lower
#    because its convex hull is much bigger than its actual area.
#    This catches irregular blobs that happened to pass the circularity test.
#
# WHY THIS ORDER?
# ---------------
# Area is O(1) and eliminates most contours.  Aspect ratio costs one
# boundingRect call.  Circularity needs arcLength.  Solidity needs convexHull
# which is the most expensive.  By gating each step only the tiny fraction
# of contours that pass all previous tests ever reach solidity.
#
# WHAT IS RETURNED
# ----------------
# A list of detection dicts, one per surviving blob:
# {
#   "bbox"        : (x, y, w, h)     bounding rectangle
#   "center"      : (cx, cy)         centroid
#   "area"        : float            contour area in px²
#   "circularity" : float            0.0 – 1.0
#   "solidity"    : float            0.0 – 1.0
#   "aspect_ratio": float            max/min side of bounding rect
# }
# =============================================================================

import cv2
import math
import numpy as np
from typing import List, Dict


def filter_blobs(contours,
                 min_area       : float = 4.0,
                 max_area       : float = 500.0,
                 min_circularity: float = 0.25,
                 min_solidity   : float = 0.45,
                 max_aspect_ratio: float = 4.0) -> List[Dict]:
    """Filter contours by geometric properties and return detection dicts.

    Parameters
    ----------
    contours         : list of np.ndarray  Raw contour list from findContours.
    min_area         : float   Minimum blob area in px².
    max_area         : float   Maximum blob area in px².
    min_circularity  : float   Minimum circularity score (0–1).
    min_solidity     : float   Minimum solidity score (0–1).
    max_aspect_ratio : float   Maximum bounding-rect width/height ratio.

    Returns
    -------
    list of dict    One dict per surviving blob (see module docstring).
    """
    detections: List[Dict] = []

    for cnt in contours:

        # ── 1. Area ───────────────────────────────────────────────────
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        # ── 2. Aspect ratio ───────────────────────────────────────────
        x, y, w, h    = cv2.boundingRect(cnt)
        long_side     = max(w, h)
        short_side    = min(w, h)
        aspect_ratio  = long_side / short_side if short_side > 0 else 999.0
        if aspect_ratio > max_aspect_ratio:
            continue

        # ── 3. Circularity ────────────────────────────────────────────
        perimeter = cv2.arcLength(cnt, True)
        if perimeter < 1e-6:
            continue
        circularity = (4.0 * math.pi * area) / (perimeter ** 2)
        if circularity < min_circularity:
            continue

        # ── 4. Solidity ───────────────────────────────────────────────
        hull      = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity  = area / hull_area if hull_area > 0 else 0.0
        if solidity < min_solidity:
            continue

        # ── Passed all filters — build detection dict ─────────────────
        cx = x + w // 2
        cy = y + h // 2
        detections.append({
            "bbox"         : (x, y, w, h),
            "center"       : (cx, cy),
            "area"         : round(area, 2),
            "circularity"  : round(circularity, 3),
            "solidity"     : round(solidity, 3),
            "aspect_ratio" : round(aspect_ratio, 2),
        })

    return detections


# =============================================================================
# SELF-TEST
# Run:  python detection/blob_filter.py
#
# THREE-PANEL window (plays the video):
#   Left   — all contours found (before blob_filter), drawn in grey
#   Middle — only blobs that PASSED all filters, drawn in green
#             with their area, circularity, solidity printed next to each
#   Right  — original ROI frame with passing bounding boxes overlaid
#
# What to look for:
#   Left vs Middle: most contours should disappear in the middle panel.
#   Middle panel: only compact, roughly circular blobs should remain.
#   Right panel: boxes should be over actual moving objects, not noise.
#
#   When the ball is in frame: ONE box should appear near it.
#   If there are still many boxes: tighten one filter at a time (see controls).
#   If the ball is never boxed: loosen one filter at a time.
#
# Controls:
#   SPACE           — pause / resume
#   LEFT/RIGHT(A/D) — step frames (paused)
#   1 / Q           — raise / lower MIN_BLOB_AREA
#   2 / W           — raise / lower MIN_CIRCULARITY
#   3 / E           — raise / lower MIN_SOLIDITY
#   4 / R           — raise / lower MAX_ASPECT_RATIO
#   P               — print current filter values to terminal
#   ESC             — quit
#
# The terminal prints the blob stats for each surviving detection so you
# can see exactly what is passing and why.
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
        MIN_BLOB_AREA, MAX_BLOB_AREA,
        MIN_CIRCULARITY, MIN_SOLIDITY, MAX_ASPECT_RATIO,
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
    from detection.contour_detect  import detect_contours

    print("\n" + "="*52)
    print("  detection/blob_filter.py — self-test")
    print("="*52)
    print("\n  Controls:")
    print("    SPACE / A / D   — pause / step")
    print("    1/Q  2/W  3/E  4/R — tune area / circ / solid / aspect")
    print("    P               — print current filter values")
    print("    ESC             — quit\n")

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

    # Live-tunable filter values
    f_min_area   = MIN_BLOB_AREA
    f_max_area   = MAX_BLOB_AREA
    f_circ       = MIN_CIRCULARITY
    f_solid      = MIN_SOLIDITY
    f_aspect     = MAX_ASPECT_RATIO

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

    def _print_filters():
        print(f"\n  Current filter values:")
        print(f"    MIN_BLOB_AREA    = {f_min_area:.1f}")
        print(f"    MAX_BLOB_AREA    = {f_max_area:.1f}")
        print(f"    MIN_CIRCULARITY  = {f_circ:.2f}")
        print(f"    MIN_SOLIDITY     = {f_solid:.2f}")
        print(f"    MAX_ASPECT_RATIO = {f_aspect:.1f}")
        print("  Copy into config.py\n")

    while idx < cached_total:
        frame        = frames_cache[idx]
        roi_frame, _ = apply_roi(frame, ROI_PERCENT, ROI_ANCHOR_X, ROI_ANCHOR_Y)
        gray         = to_gray(roi_frame, CLAHE_CLIP_LIMIT,
                                CLAHE_TILE_GRID, ENABLE_CLAHE)

        th_mask  = apply_tophat(gray, TOPHAT_KERNEL_SIZE,
                                 TOPHAT_THRESHOLD, TOPHAT_MODE) \
                   if ENABLE_TOPHAT else None
        df_mask  = differ.update(gray)  if ENABLE_FRAME_DIFF else None
        mg_mask  = bgs.apply(roi_frame) if ENABLE_BG_SUB      else None
        combined = combine_masks(th_mask, df_mask, mg_mask, MASK_COMBINE_MODE)
        cleaned  = clean_mask(combined, MORPH_KERNEL_SIZE) \
                   if ENABLE_MORPH_CLEAN else combined

        all_contours = detect_contours(cleaned)
        detections   = filter_blobs(
            all_contours, f_min_area, f_max_area,
            f_circ, f_solid, f_aspect
        )

        h_img, w_img = roi_frame.shape[:2]

        # ── Left: all contours (grey) ─────────────────────────────────
        left = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(left, all_contours, -1, (120, 120, 120), 1)

        # ── Middle: passing blobs with stats ──────────────────────────
        middle = np.zeros((h_img, w_img, 3), dtype=np.uint8)
        for det in detections:
            x, y, w, h = det["bbox"]
            cv2.rectangle(middle, (x, y), (x+w, y+h), (0, 255, 0), 1)
            label = (f"a={det['area']:.0f} "
                     f"c={det['circularity']:.2f} "
                     f"s={det['solidity']:.2f}")
            cv2.putText(middle, label,
                        (x, max(y - 3, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 180), 1)

        # ── Right: ROI frame + passing boxes ──────────────────────────
        right = roi_frame.copy()
        for det in detections:
            x, y, w, h = det["bbox"]
            cv2.rectangle(right, (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv2.circle(right, det["center"], 3, (255, 0, 0), -1)

        n_all  = len(all_contours)
        n_pass = len(detections)
        for panel, label in [
            (left,   f"All contours ({n_all})"),
            (middle, f"Passed filters ({n_pass})"),
            (right,  "Frame output"),
        ]:
            cv2.putText(panel, label, (5, h_img - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        disp = np.hstack([left, middle, right])
        ch, cw = disp.shape[:2]
        if cw > DISPLAY_MAX_W or ch > DISPLAY_MAX_H:
            scale = min(DISPLAY_MAX_W / cw, DISPLAY_MAX_H / ch)
            disp  = cv2.resize(disp, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)

        status = "PAUSED" if paused else "PLAYING"
        cv2.putText(disp,
                    f"Frame {idx}/{cached_total-1}  "
                    f"all={n_all}  passed={n_pass}  [{status}]",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,100), 1)

        cv2.imshow("blob_filter.py — self-test", disp)
        wait = 0 if paused else 30
        key  = cv2.waitKey(wait) & 0xFF

        if key == 27:
            break
        elif key == ord(" "):
            paused = not paused
        elif key in (83, ord("d")):
            idx = min(idx + 1, cached_total - 1)
        elif key in (81, ord("a")):
            idx = max(idx - 1, 0);  differ.reset()
        # Area
        elif key == ord("1"):
            f_min_area = min(f_min_area + 1, f_max_area - 1)
            print(f"  MIN_BLOB_AREA → {f_min_area:.1f}")
        elif key == ord("q"):
            f_min_area = max(f_min_area - 1, 1)
            print(f"  MIN_BLOB_AREA → {f_min_area:.1f}")
        # Circularity
        elif key == ord("2"):
            f_circ = min(f_circ + 0.05, 0.95)
            print(f"  MIN_CIRCULARITY → {f_circ:.2f}")
        elif key == ord("w"):
            f_circ = max(f_circ - 0.05, 0.05)
            print(f"  MIN_CIRCULARITY → {f_circ:.2f}")
        # Solidity
        elif key == ord("3"):
            f_solid = min(f_solid + 0.05, 0.95)
            print(f"  MIN_SOLIDITY → {f_solid:.2f}")
        elif key == ord("e"):
            f_solid = max(f_solid - 0.05, 0.05)
            print(f"  MIN_SOLIDITY → {f_solid:.2f}")
        # Aspect ratio
        elif key == ord("4"):
            f_aspect = min(f_aspect + 0.5, 10.0)
            print(f"  MAX_ASPECT_RATIO → {f_aspect:.1f}")
        elif key == ord("r"):
            f_aspect = max(f_aspect - 0.5, 1.0)
            print(f"  MAX_ASPECT_RATIO → {f_aspect:.1f}")
        elif key == ord("p"):
            _print_filters()
        else:
            if not paused:
                idx += 1

    cv2.destroyAllWindows()
    reader.release()
    _print_filters()
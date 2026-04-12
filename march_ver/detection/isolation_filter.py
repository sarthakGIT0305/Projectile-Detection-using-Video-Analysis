# =============================================================================
# detection/isolation_filter.py
# =============================================================================
# Rejects blobs that are clustered together — noise sources like leaves,
# grass, and textured backgrounds produce many nearby blobs, while a real
# tennis ball in flight is ALWAYS a single isolated blob.
#
# HOW IT WORKS
# ------------
# The key physical observation is:
#
#     A tennis ball in flight is always a SINGLE isolated blob.
#     It exists alone in relatively empty space.
#     It is NEVER surrounded by other blobs at close range.
#
#     Noise sources — leaves, grass, textured backgrounds — always produce
#     MULTIPLE blobs clustered tightly together in the same local region.
#
# Therefore: if N or more blobs exist within a radius R of each other,
# they are all noise and should ALL be rejected together.
#
# This is the exact opposite of the usual "find clusters" task —
# here, being IN a cluster is the rejection criterion, not the selection
# criterion.
#
# ALGORITHM
# ---------
# 1. For every pair of detections, compute Euclidean distance between
#    their center points.
# 2. For each detection, count how many other detections are within
#    `radius` pixels (its neighbour count).
# 3. Build a rejection set: any detection with neighbour count (including
#    itself) >= min_cluster_size is rejected.  ALL of its neighbours are
#    also rejected — rejection is contagious.
# 4. Return all detections NOT in the rejection set.
#
# PARAMETERS (set in config.py):
#   ENABLE_ISOLATION_FILTER  — master on/off switch
#   ISOLATION_RADIUS         — pixel radius for neighbour queries
#   ISOLATION_MIN_CLUSTER    — minimum blobs in a neighbourhood to reject
# =============================================================================

import math
from typing import Dict, List, Tuple, Optional


def reject_clustered_blobs(
    detections: List[Dict],
    radius: float = 35.0,
    min_cluster_size: int = 3,
    return_debug: bool = False,
):
    """Remove blobs that are clustered together — they are noise.

    A real ball in flight is always isolated.  Leaves, grass, and textured
    backgrounds produce many blobs clustered in the same region.  This
    function rejects every blob that belongs to a local group of
    ``min_cluster_size`` or more detections within ``radius`` pixels.

    Parameters
    ----------
    detections       : list of dict   Output from ``filter_blobs()``.
                       Each dict must have a ``"center"`` key → (cx, cy).
    radius           : float          Max pixel distance to be neighbours.
    min_cluster_size : int            Minimum blobs in a local group to
                                      trigger rejection (must be >= 2).
    return_debug     : bool           If True, return extra debug info.

    Returns
    -------
    If ``return_debug`` is False (default):
        list of dict — surviving (isolated) detections.

    If ``return_debug`` is True:
        tuple of (surviving_list, rejected_list, neighbour_counts_dict)
        where neighbour_counts_dict maps detection index → neighbour count
        (including itself).

    Raises
    ------
    ValueError
        If ``min_cluster_size`` < 2 (this would reject everything).

    Edge Cases
    ----------
    - Empty input          → empty output
    - Single detection     → always survives
    - radius == 0          → no neighbours possible, all survive
    - All in one cluster   → empty output
    """
    # ── Validate ──────────────────────────────────────────────────────────
    if min_cluster_size < 2:
        raise ValueError(
            f"min_cluster_size must be >= 2, got {min_cluster_size}. "
            f"A value of 1 would reject every detection unconditionally."
        )

    n = len(detections)

    # Trivial cases
    if n == 0:
        return ([], [], {}) if return_debug else []

    if n == 1:
        if return_debug:
            return (list(detections), [], {0: 1})
        return list(detections)

    # ── Build neighbour lists ─────────────────────────────────────────────
    # For typical counts (5–100 blobs) O(n²) is perfectly fine.
    # For large counts (>200), use scipy.spatial.cKDTree if available.
    centres = [d["center"] for d in detections]
    radius_sq = radius * radius

    # neighbours[i] = set of indices within radius of detection i
    neighbours: List[set] = [set() for _ in range(n)]

    if n > 200:
        # Fast path: try scipy cKDTree
        try:
            from scipy.spatial import cKDTree
            import numpy as np

            pts = np.array(centres, dtype=float)
            tree = cKDTree(pts)
            pairs = tree.query_pairs(r=radius, output_type="set")
            for i, j in pairs:
                neighbours[i].add(j)
                neighbours[j].add(i)
        except ImportError:
            # Fallback to O(n²) if scipy is not available
            _pairwise_neighbours(centres, radius_sq, neighbours)
    else:
        _pairwise_neighbours(centres, radius_sq, neighbours)

    # ── Count neighbours (including self) ─────────────────────────────────
    # neighbour_count[i] = len(neighbours[i]) + 1  (counting itself)
    neighbour_counts = {}
    for i in range(n):
        neighbour_counts[i] = len(neighbours[i]) + 1  # +1 for itself

    # ── Build rejection set ───────────────────────────────────────────────
    # Rejection is contagious: if blob i has >= min_cluster_size neighbours,
    # reject blob i AND all its neighbours.
    rejection_set: set = set()

    for i in range(n):
        if neighbour_counts[i] >= min_cluster_size:
            rejection_set.add(i)
            rejection_set.update(neighbours[i])

    # ── Partition into surviving / rejected ────────────────────────────────
    surviving = []
    rejected = []
    for i in range(n):
        if i in rejection_set:
            rejected.append(detections[i])
        else:
            surviving.append(detections[i])

    if return_debug:
        return surviving, rejected, neighbour_counts
    return surviving


def _pairwise_neighbours(
    centres: List[Tuple[int, int]],
    radius_sq: float,
    neighbours: List[set],
) -> None:
    """O(n²) pairwise distance check.  Modifies *neighbours* in-place."""
    n = len(centres)
    for i in range(n):
        cx_i, cy_i = centres[i]
        for j in range(i + 1, n):
            cx_j, cy_j = centres[j]
            dx = cx_i - cx_j
            dy = cy_i - cy_j
            if dx * dx + dy * dy <= radius_sq:
                neighbours[i].add(j)
                neighbours[j].add(i)


# =============================================================================
# SELF-TEST — THREE-PANEL PIPELINE PREVIEW
# Run:  python detection/isolation_filter.py
#
# THREE-PANEL window playing the video:
#
#   Left   — all blobs that passed filter_blobs() (WHITE circles)
#   Middle — cluster membership:
#              RED   = will be rejected (in a cluster)
#              GREEN = will survive (isolated)
#              Faint circle of radius=ISOLATION_RADIUS around each red blob
#   Right  — only surviving isolated blobs on the original ROI frame
#             with green bounding boxes
#
# Controls:
#   SPACE           — pause / resume
#   LEFT/RIGHT(A/D) — step frames when paused
#   R / F           — raise / lower ISOLATION_RADIUS (+/- 5px)
#   C / V           — raise / lower ISOLATION_MIN_CLUSTER (+/- 1)
#   P               — print current values to terminal
#   Q / ESC         — quit and print final recommended config values
# =============================================================================
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    import cv2
    import numpy as np

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
        ISOLATION_RADIUS, ISOLATION_MIN_CLUSTER,
        ROI_PERCENT, ROI_ANCHOR_X, ROI_ANCHOR_Y,
        DISPLAY_MAX_W, DISPLAY_MAX_H,
    )
    from video_reader              import VideoReader
    from preprocessing.clahe_gray  import to_gray
    from preprocessing.roi         import apply_roi
    from motion.tophat             import apply_tophat
    from motion.frame_diff         import FrameDiffer
    from motion.bg_sub             import BackgroundSubtractor
    from motion.mask_combine       import combine_masks
    from motion.morph_clean        import clean_mask
    from detection.contour_detect  import detect_contours
    from detection.blob_filter     import filter_blobs

    print("\n" + "=" * 52)
    print("  detection/isolation_filter.py — self-test")
    print("=" * 52)
    print("\n  Controls:")
    print("    SPACE / A / D   — pause / step")
    print("    R / F           — raise / lower ISOLATION_RADIUS (+/-5)")
    print("    C / V           — raise / lower ISOLATION_MIN_CLUSTER (+/-1)")
    print("    P               — print current values")
    print("    Q / ESC         — quit\n")

    try:
        reader = VideoReader(VIDEO_PATH)
    except RuntimeError as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    differ = FrameDiffer(FRAME_DIFF_GAP, FRAME_DIFF_THRESHOLD)
    bgs    = BackgroundSubtractor(MOG2_HISTORY, MOG2_VAR_THRESHOLD,
                                   MOG2_DETECT_SHADOWS)
    paused = False

    # Live-tunable values
    iso_radius  = ISOLATION_RADIUS
    iso_cluster = ISOLATION_MIN_CLUSTER

    # Pre-cache
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

    idx = 0

    def _print_values():
        print(f"\n  Current isolation filter values:")
        print(f"    ISOLATION_RADIUS      = {iso_radius}")
        print(f"    ISOLATION_MIN_CLUSTER = {iso_cluster}")
        print("  Copy into config.py\n")

    while idx < cached_total:
        frame        = frames_cache[idx]
        roi_frame, _ = apply_roi(frame, ROI_PERCENT, ROI_ANCHOR_X, ROI_ANCHOR_Y)
        gray         = to_gray(roi_frame, CLAHE_CLIP_LIMIT,
                                CLAHE_TILE_GRID, ENABLE_CLAHE)

        th_mask = apply_tophat(gray, TOPHAT_KERNEL_SIZE,
                                TOPHAT_THRESHOLD, TOPHAT_MODE) \
                  if ENABLE_TOPHAT else None
        df_mask = differ.update(gray)   if ENABLE_FRAME_DIFF else None
        mg_mask = bgs.apply(roi_frame)  if ENABLE_BG_SUB     else None
        combined = combine_masks(th_mask, df_mask, mg_mask, MASK_COMBINE_MODE)
        cleaned  = clean_mask(combined, MORPH_KERNEL_SIZE) \
                   if ENABLE_MORPH_CLEAN else combined

        contours   = detect_contours(cleaned)
        detections = filter_blobs(
            contours, MIN_BLOB_AREA, MAX_BLOB_AREA,
            MIN_CIRCULARITY, MIN_SOLIDITY, MAX_ASPECT_RATIO,
        )

        # Run isolation filter in debug mode
        surviving, rejected, ncounts = reject_clustered_blobs(
            detections, radius=iso_radius,
            min_cluster_size=iso_cluster, return_debug=True,
        )

        h_img, w_img = roi_frame.shape[:2]

        # ── Left: all blobs from filter_blobs (white circles) ──────────
        left = np.zeros((h_img, w_img, 3), dtype=np.uint8)
        for det in detections:
            cx, cy = det["center"]
            cv2.circle(left, (cx, cy), 4, (255, 255, 255), -1)

        # ── Middle: cluster membership ─────────────────────────────────
        middle = np.zeros((h_img, w_img, 3), dtype=np.uint8)
        # Draw rejection radii first (faint)
        for det in rejected:
            cx, cy = det["center"]
            cv2.circle(middle, (cx, cy), int(iso_radius),
                       (40, 40, 80), 1)
        # Draw rejected blobs (red)
        for det in rejected:
            cx, cy = det["center"]
            cv2.circle(middle, (cx, cy), 4, (0, 0, 255), -1)
        # Draw surviving blobs (green)
        for det in surviving:
            cx, cy = det["center"]
            cv2.circle(middle, (cx, cy), 5, (0, 255, 0), -1)

        # ── Right: surviving blobs on ROI frame ────────────────────────
        right = roi_frame.copy()
        for det in surviving:
            x, y, w, h = det["bbox"]
            cv2.rectangle(right, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.circle(right, det["center"], 3, (255, 0, 0), -1)

        # Labels
        n_all  = len(detections)
        n_rej  = len(rejected)
        n_surv = len(surviving)
        for panel, label in [
            (left,   f"filter_blobs ({n_all})"),
            (middle, f"Rejected {n_rej} | Surviving {n_surv}"),
            (right,  f"Isolated blobs ({n_surv})"),
        ]:
            cv2.putText(panel, label, (5, h_img - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        disp = np.hstack([left, middle, right])
        ch, cw = disp.shape[:2]
        if cw > DISPLAY_MAX_W * 2 or ch > DISPLAY_MAX_H:
            scale = min((DISPLAY_MAX_W * 2) / cw, DISPLAY_MAX_H / ch)
            disp = cv2.resize(disp, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)

        status = "PAUSED" if paused else "PLAYING"
        cv2.putText(disp,
                    f"Frame {idx}/{cached_total-1}  R={iso_radius}  "
                    f"C={iso_cluster}  [{status}]",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (255, 255, 100), 1)

        cv2.imshow("isolation_filter.py — self-test", disp)
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
        elif key == ord("r"):
            iso_radius += 5
            print(f"  ISOLATION_RADIUS → {iso_radius}")
        elif key == ord("f"):
            iso_radius = max(5, iso_radius - 5)
            print(f"  ISOLATION_RADIUS → {iso_radius}")
        elif key == ord("c"):
            iso_cluster += 1
            print(f"  ISOLATION_MIN_CLUSTER → {iso_cluster}")
        elif key == ord("v"):
            iso_cluster = max(2, iso_cluster - 1)
            print(f"  ISOLATION_MIN_CLUSTER → {iso_cluster}")
        elif key == ord("p"):
            _print_values()
            if paused:
                print(f"  Frame {idx}: filter_blobs kept {n_all}  |  "
                      f"isolation removed {n_rej}  |  surviving {n_surv}")
        else:
            if not paused:
                idx += 1

    cv2.destroyAllWindows()
    reader.release()
    _print_values()
    print("  Self-test complete.\n")

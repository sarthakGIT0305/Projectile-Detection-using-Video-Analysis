# =============================================================================
# tracking/trail_store.py
# =============================================================================
# Records the recent position history of each tracked object so we can:
#   1. Draw a rolling arc trail behind the ball
#   2. Feed the trajectory validator with historical positions
#   3. Visually distinguish observed vs. predicted positions
#
# HOW IT WORKS
# -------------
# For each track ID, TrailStore keeps a deque of the last TRAIL_LENGTH
# centroid positions.  Each entry is a (cx, cy, observed) tuple where
# ``observed`` is True if the position came from a real detection, or False
# if it was a Kalman prediction (the track was in "missing" mode).
#
# The trail drawing changes colour for predicted segments so the user can
# see exactly where the tracker was guessing vs. where it had real data.
#
# PARAMETERS (set in config.py):
#   TRAIL_LENGTH — number of past positions kept per track
#   COLOR_TRAIL  — BGR drawing colour for the observed trail
# =============================================================================

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque


class TrailStore:
    """Accumulates per-track position history for trail drawing."""

    def __init__(self, max_length: int = 40):
        self.max_length = max_length
        self._trails: Dict[int, deque] = {}

    def update(self, tracks_or_id, cx=None, cy=None, observed=True):
        """Record positions — accepts EITHER a list of track dicts OR individual args.

        Calling conventions:
          update(tracks_list)               — from main.py
          update(track_id, cx, cy, observed) — from self-test / manual code
        """
        if isinstance(tracks_or_id, list):
            # main.py style: list of track dicts
            active_ids = set()
            for t in tracks_or_id:
                tid = t["id"]
                tcx, tcy = t["center"]
                obs = t.get("missing", 0) == 0
                active_ids.add(tid)
                self.update_single(tid, tcx, tcy, obs)
            self.prune(active_ids)
        else:
            # Original style: individual args
            self.update_single(tracks_or_id, cx, cy, observed)

    def update_single(self, track_id: int, cx: int, cy: int, observed: bool = True):
        """Record a new position for *track_id*.

        Parameters
        ----------
        track_id  : int    Unique track identifier from KalmanTracker.
        cx, cy    : int    Centroid position.
        observed  : bool   True if this came from a real detection,
                           False if it was a Kalman prediction.
        """
        if track_id not in self._trails:
            self._trails[track_id] = deque(maxlen=self.max_length)
        self._trails[track_id].append((cx, cy, observed))

    def get(self, track_id: int) -> Optional[deque]:
        """Return the trail deque for *track_id*, or None."""
        return self._trails.get(track_id)

    def get_points(self, track_id: int) -> List[Tuple[int, int]]:
        """Return just the (cx, cy) points for *track_id*."""
        trail = self._trails.get(track_id)
        if trail is None:
            return []
        return [(p[0], p[1]) for p in trail]

    def get_observed_points(self, track_id: int) -> List[Tuple[int, int]]:
        """Return only observed (non-predicted) points for *track_id*."""
        trail = self._trails.get(track_id)
        if trail is None:
            return []
        return [(p[0], p[1]) for p in trail if p[2]]

    def remove(self, track_id: int):
        """Remove trail for a track that no longer exists."""
        self._trails.pop(track_id, None)

    def prune(self, active_ids: set):
        """Remove trails for tracks that are no longer active."""
        dead = [tid for tid in self._trails if tid not in active_ids]
        for tid in dead:
            del self._trails[tid]

    def draw_all_trails(self, frame: np.ndarray,
                        colors: List[Tuple[int, int, int]] = None,
                        thickness: int = 2,
                        fade: bool = True) -> np.ndarray:
        """Draw all trails with per-track colours from a palette.

        Parameters
        ----------
        frame    : np.ndarray  BGR image to draw on (modified in-place).
        colors   : list        Colour palette; cycled by track ID.
        thickness: int         Line thickness.
        fade     : bool        Fade older segments.

        Returns
        -------
        np.ndarray   The same frame.
        """
        if colors is None:
            colors = [(0, 255, 255)]  # default yellow

        for tid, trail in self._trails.items():
            base_color = colors[tid % len(colors)]
            dim_color  = tuple(int(c * 0.5) for c in base_color)
            pts = list(trail)
            n = len(pts)
            if n < 2:
                continue
            for i in range(1, n):
                p1 = (pts[i - 1][0], pts[i - 1][1])
                p2 = (pts[i][0], pts[i][1])
                is_observed = pts[i][2]
                if fade:
                    alpha = 0.3 + 0.7 * (i / (n - 1))
                    c_base = base_color if is_observed else dim_color
                    c = tuple(int(ch * alpha) for ch in c_base)
                else:
                    c = base_color if is_observed else dim_color
                cv2.line(frame, p1, p2, c, thickness)
        return frame

    def draw_trails(self, frame: np.ndarray,
                    color_observed: Tuple[int, int, int] = (0, 255, 255),
                    color_predicted: Tuple[int, int, int] = (0, 120, 180),
                    thickness: int = 2,
                    fade: bool = True) -> np.ndarray:
        """Draw all trails onto *frame* (single-colour mode).

        Parameters
        ----------
        frame            : np.ndarray  BGR image to draw on (modified in-place).
        color_observed   : tuple       BGR colour for observed segments.
        color_predicted  : tuple       BGR colour (dimmer) for predicted segments.
        thickness        : int         Line thickness.
        fade             : bool        If True, older segments are progressively
                                       dimmer (alpha-like fade effect).

        Returns
        -------
        np.ndarray   The same frame (for chaining).
        """
        for tid, trail in self._trails.items():
            pts = list(trail)
            n = len(pts)
            if n < 2:
                continue

            for i in range(1, n):
                p1 = (pts[i - 1][0], pts[i - 1][1])
                p2 = (pts[i][0], pts[i][1])
                is_observed = pts[i][2]

                if fade:
                    alpha = 0.3 + 0.7 * (i / (n - 1))
                    if is_observed:
                        c = tuple(int(ch * alpha) for ch in color_observed)
                    else:
                        c = tuple(int(ch * alpha) for ch in color_predicted)
                else:
                    c = color_observed if is_observed else color_predicted

                cv2.line(frame, p1, p2, c, thickness)

        return frame

    def reset(self):
        """Clear all trails."""
        self._trails.clear()


# =============================================================================
# SELF-TEST — FULL PIPELINE PREVIEW
# Run:  python tracking/trail_store.py
#
# This is the MOST IMPORTANT test in the project.  It wires together every
# stage from video_reader through tracking and draws the final annotated
# output — this is what main.py will look like.
#
# ONE WINDOW with:
#   • Green bounding box + track ID for each active track
#   • Yellow arc trail behind the ball (observed segments)
#   • Dimmer orange arc for predicted segments (ball was occluded)
#   • Blue centroid dot on current position
#   • "PRED(n)" label if the tracker is predicting
#
# THE ARC TRAIL IS THE VISUAL PROOF
# -----------------------------------
# | Trail appearance           | What it means                                |
# |----------------------------|----------------------------------------------|
# | Smooth continuous arc      | Everything is working correctly              |
# | Arc has gaps               | KALMAN_MAX_MISSING needs to be higher        |
# | Arc has sharp zigzags      | Blob filter too loose — multiple blobs       |
# | No arc at all              | Detection failing — go back to blob_filter   |
# | Dimmer colour segments     | Kalman prediction fired — correct behaviour  |
#
# Controls:
#   SPACE           — pause / resume
#   LEFT/RIGHT(A/D) — step frames (paused)
#   T               — toggle trail on/off
#   R               — reset tracker + trails
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
        MIN_BLOB_AREA, MAX_BLOB_AREA,
        MIN_CIRCULARITY, MIN_SOLIDITY, MAX_ASPECT_RATIO,
        KALMAN_MAX_DISTANCE, KALMAN_MAX_MISSING,
        TRAIL_LENGTH,
        ROI_PERCENT, ROI_ANCHOR_X, ROI_ANCHOR_Y,
        DISPLAY_MAX_W, DISPLAY_MAX_H,
        COLOR_BBOX, COLOR_CENTER, COLOR_ID, COLOR_TRAIL,
        FONT_FACE, FONT_SCALE, FONT_THICKNESS,
    )
    from video_reader              import VideoReader
    from preprocessing.clahe_gray  import to_gray
    from preprocessing.roi         import apply_roi
    from motion.tophat             import apply_tophat
    from motion.frame_diff         import FrameDiffer
    from motion.bg_sub             import BgSubtractor
    from motion.mask_combine       import combine_masks
    from motion.morph_clean        import morph_clean
    from detection.contour_detect  import detect_contours
    from detection.blob_filter     import filter_blobs
    from tracking.kalman_tracker   import KalmanTracker

    print("\n" + "="*52)
    print("  tracking/trail_store.py — FULL PIPELINE PREVIEW")
    print("="*52)
    print(f"\n  TRAIL_LENGTH         = {TRAIL_LENGTH}")
    print(f"  KALMAN_MAX_DISTANCE  = {KALMAN_MAX_DISTANCE}")
    print(f"  KALMAN_MAX_MISSING   = {KALMAN_MAX_MISSING}")
    print()
    print("  Controls:")
    print("    SPACE / A / D   — pause / step")
    print("    T               — toggle trail drawing")
    print("    R               — reset tracker + trails")
    print("    Q / ESC         — quit\n")

    try:
        reader = VideoReader(VIDEO_PATH)
    except RuntimeError as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    differ   = FrameDiffer(gap=FRAME_DIFF_GAP, threshold=FRAME_DIFF_THRESHOLD)
    bg_sub   = BgSubtractor(history=MOG2_HISTORY,
                            var_threshold=MOG2_VAR_THRESHOLD,
                            detect_shadows=MOG2_DETECT_SHADOWS)
    tracker  = KalmanTracker(max_distance=KALMAN_MAX_DISTANCE,
                             max_missing=KALMAN_MAX_MISSING)
    trails   = TrailStore(max_length=TRAIL_LENGTH)
    paused   = False
    show_trail = True

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

    while idx < cached_total:
        frame        = frames_cache[idx]
        roi_frame, _ = apply_roi(frame, ROI_PERCENT, ROI_ANCHOR_X, ROI_ANCHOR_Y)
        gray         = to_gray(roi_frame, CLAHE_CLIP_LIMIT,
                               CLAHE_TILE_GRID, ENABLE_CLAHE)

        # ── Motion masks ─────────────────────────────────────────────
        th_mask  = apply_tophat(gray, mode=TOPHAT_MODE,
                                kernel_size=TOPHAT_KERNEL_SIZE,
                                threshold=TOPHAT_THRESHOLD) \
                   if ENABLE_TOPHAT else None
        df_mask  = differ.update(gray)  if ENABLE_FRAME_DIFF else None
        mg_mask  = bg_sub.update(gray)  if ENABLE_BG_SUB     else None
        combined = combine_masks(th_mask, df_mask, mg_mask, MASK_COMBINE_MODE)
        cleaned  = morph_clean(combined, kernel_size=MORPH_KERNEL_SIZE) \
                   if ENABLE_MORPH_CLEAN else combined

        # ── Detection ────────────────────────────────────────────────
        contours   = detect_contours(cleaned)
        detections = filter_blobs(contours, MIN_BLOB_AREA, MAX_BLOB_AREA,
                                  MIN_CIRCULARITY, MIN_SOLIDITY,
                                  MAX_ASPECT_RATIO)

        # ── Tracking ─────────────────────────────────────────────────
        tracks = tracker.update(detections)

        # ── Trail recording ──────────────────────────────────────────
        active_ids = set()
        for t in tracks:
            active_ids.add(t["id"])
            px, py = t["center"]
            trails.update_single(t["id"], px, py, observed=t["missing"] == 0)
        trails.prune(active_ids)

        # ── Drawing ──────────────────────────────────────────────────
        display = roi_frame.copy()

        # Trail arcs
        if show_trail:
            trails.draw_trails(display,
                              color_observed=COLOR_TRAIL,
                              color_predicted=(0, 120, 180),
                              thickness=2, fade=True)

        # Track annotations
        for t in tracks:
            px, py = t["center"]

            # Bounding box from last detection
            if t["det"] is not None:
                bx, by, bw, bh = t["det"]["bbox"]
                box_color = COLOR_BBOX if t["missing"] == 0 else (0, 100, 255)
                cv2.rectangle(display, (bx, by), (bx+bw, by+bh), box_color, 1)

            # Centroid
            cv2.circle(display, (px, py), 3, COLOR_CENTER, -1)

            # ID label
            label = f"ID:{t['id']}"
            if t["predicted"]:
                label += f" PRED({t['missing']})"
            cv2.putText(display, label,
                        (px + 6, py - 6), FONT_FACE, FONT_SCALE,
                        COLOR_ID, FONT_THICKNESS)

        # Resize
        dh, dw = display.shape[:2]
        if dw > DISPLAY_MAX_W or dh > DISPLAY_MAX_H:
            scale   = min(DISPLAY_MAX_W / dw, DISPLAY_MAX_H / dh)
            display = cv2.resize(display, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_AREA)

        status = "PAUSED" if paused else "PLAYING"
        trail_label = "ON" if show_trail else "OFF"
        cv2.putText(display,
                    f"Frame {idx}/{cached_total-1}  tracks={len(tracks)}  "
                    f"trail={trail_label}  [{status}]",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 100), 1)

        cv2.imshow("trail_store.py — FULL PIPELINE PREVIEW", display)
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
        elif key == ord("t"):
            show_trail = not show_trail
            print(f"  Trail {'ON' if show_trail else 'OFF'}")
        elif key == ord("r"):
            tracker.reset()
            trails.reset()
            print("  Tracker + trails reset.")
        else:
            if not paused:
                idx += 1

    cv2.destroyAllWindows()
    reader.release()
    print("\n  Self-test complete.\n")

# =============================================================================
# detection/trajectory_fit.py
# =============================================================================
# Validates tracked objects by fitting a parabola to their position history
# and rejecting any track whose path does not match projectile motion.
#
# THE PHYSICS BEHIND THIS
# ------------------------
# Any object thrown or launched under gravity (ignoring air resistance) follows:
#
#   y = ax² + bx + c    (parabola in image coordinates)
#
# where x is the horizontal pixel position and y is the vertical pixel position.
#
# This is a second-degree polynomial.  Given at least 3 points we can fit one,
# but with only 3 points the fit is always perfect regardless of the actual path.
# We need TRAJECTORY_MIN_POINTS (default 6) so that genuine noise tracks —
# which don't follow a parabola — show a high residual error.
#
# HOW THE FIT WORKS
# ------------------
# numpy.polyfit(x_coords, y_coords, deg=2) finds the coefficients [a, b, c]
# that minimise the sum of squared vertical residuals.
#
# We then compute the mean absolute residual:
#   residual = mean(|y_actual - y_predicted|)   in pixels
#
# If residual > TRAJECTORY_MAX_RESIDUAL the track is not parabolic and is
# almost certainly noise.  We reject it.
#
# WHY THIS IS THE STRONGEST FILTER
# ----------------------------------
# Every other filter works per-frame on individual blobs.  This filter works
# ACROSS TIME using physics.  In a fixed-camera outdoor scene, the tennis ball
# is the only object that:
#   1. Appears for multiple consecutive frames
#   2. Moves consistently in one direction
#   3. Decelerates vertically (gravity)
#   4. Follows a parabola to within a few pixels
#
# A bird flies straight.  A leaf drifts randomly.  A compression artefact
# appears for one frame.  A person walks — a straight line, not a parabola.
# None of them pass this test.  The ball always does.
#
# DIRECTION DETECTION
# --------------------
# We also detect the ball's travel direction (left-to-right or right-to-left)
# and flight phase (ascending, descending, or complete arc) from the fitted
# coefficients and the sign of 'a' (positive a = opens downward in image
# coordinates where y increases downward).
#
# LIMITATIONS
# ------------
# 1. Needs TRAJECTORY_MIN_POINTS frames before it can fire.  For 6 frames
#    at 30 fps that is 0.2 seconds of latency before a track is confirmed.
# 2. If the camera is moving, the background motion shifts all detected
#    positions and the parabola fit breaks down.  Only use with a fixed camera.
# 3. Very short throws that stay in frame for fewer than 6 frames may not
#    accumulate enough points.  Lower TRAJECTORY_MIN_POINTS to 4 or 5 in
#    that case (but expect slightly more false positives).
# =============================================================================

import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import deque


class TrajectoryValidator:
    """Fits a parabola to a rolling history of centroid positions.

    One instance per tracked object.  Call update() every frame with the
    latest centroid.  Call is_valid() to check if the track passes.

    Usage
    -----
    val = TrajectoryValidator(min_points=6, max_residual=8.0)
    for cx, cy in centroids:
        val.update(cx, cy)
        if val.is_valid():
            print("Ball confirmed!", val.get_info())
    """

    def __init__(self, min_points: int = 6, max_residual: float = 8.0):
        self.min_points   = min_points
        self.max_residual = max_residual
        self._history     = deque(maxlen=max(min_points * 3, 20))
        self._residual    = None
        self._coeffs      = None   # [a, b, c] from polyfit

    def update(self, cx: int, cy: int):
        """Add a new centroid observation."""
        self._history.append((cx, cy))
        self._fit()

    def _fit(self):
        """Fit parabola if enough points are available."""
        if len(self._history) < self.min_points:
            self._residual = None
            self._coeffs   = None
            return

        xs = np.array([p[0] for p in self._history], dtype=np.float64)
        ys = np.array([p[1] for p in self._history], dtype=np.float64)

        # Degenerate case: all x values identical (ball moving straight up)
        if np.ptp(xs) < 2:
            self._residual = None
            self._coeffs   = None
            return

        try:
            coeffs    = np.polyfit(xs, ys, deg=2)
            y_pred    = np.polyval(coeffs, xs)
            residuals = np.abs(ys - y_pred)
            self._residual = float(np.mean(residuals))
            self._coeffs   = coeffs
        except np.linalg.LinAlgError:
            self._residual = None
            self._coeffs   = None

    def is_valid(self) -> bool:
        """True if track has enough points AND fits a parabola well."""
        if self._residual is None:
            return False
        return self._residual <= self.max_residual

    def is_ready(self) -> bool:
        """True once enough points have been collected to attempt a fit."""
        return len(self._history) >= self.min_points

    def get_residual(self) -> Optional[float]:
        return self._residual

    def get_info(self) -> Dict:
        """Return a dict of diagnostics for display / logging."""
        info = {
            "n_points"  : len(self._history),
            "residual"  : self._residual,
            "is_valid"  : self.is_valid(),
            "is_ready"  : self.is_ready(),
            "direction" : None,
            "phase"     : None,
        }
        if self._coeffs is not None and len(self._history) >= 2:
            xs = [p[0] for p in self._history]
            # Direction: ball moving left or right
            info["direction"] = "left_to_right" if xs[-1] > xs[0] \
                                 else "right_to_left"
            # Phase: ascending (y decreasing), descending (y increasing),
            # or full arc (both present).
            # In image coords y increases downward, so ascending = y decreasing.
            ys     = [p[1] for p in self._history]
            y_diff = ys[-1] - ys[0]
            if abs(y_diff) < 5:
                info["phase"] = "arc_peak"
            elif y_diff < 0:
                info["phase"] = "ascending"
            else:
                info["phase"] = "descending"
        return info

    def reset(self):
        self._history.clear()
        self._residual = None
        self._coeffs   = None

    def get_fitted_points(self, n: int = 50) -> Optional[List[Tuple[int,int]]]:
        """Return points along the fitted parabola for drawing.

        Returns None if no fit is available.
        """
        if self._coeffs is None or len(self._history) < 2:
            return None
        xs     = np.array([p[0] for p in self._history])
        x_min, x_max = xs.min(), xs.max()
        if x_max <= x_min:
            return None
        x_range = np.linspace(x_min, x_max, n)
        y_range = np.polyval(self._coeffs, x_range)
        return [(int(x), int(y)) for x, y in zip(x_range, y_range)]


# =============================================================================
# SELF-TEST
# Run:  python detection/trajectory_fit.py
#
# This test simulates a single tracked ball by running the full pipeline
# and feeding detections into ONE TrajectoryValidator instance.
# (The multi-track version lives in tracking/kalman_tracker.py —
#  here we validate the parabola fitting logic in isolation.)
#
# TWO-PANEL window (plays the video):
#   Left  — ROI frame with:
#             • green box around the BEST detection each frame
#               (smallest blob that passed blob_filter)
#             • blue fitted parabola arc drawn once enough points collected
#             • red cross if track is REJECTED (residual too high)
#             • green arc if track is VALID
#   Right — residual plot showing how the parabola fit quality evolves
#           over the last 60 frames (lower = better fit)
#
# What to look for:
#   Once a ball is thrown the residual should drop as the arc forms.
#   It should stay low (below the red threshold line) throughout the flight.
#   After the ball lands the residual should rise as the track degrades.
#
# Controls:
#   SPACE           — pause / resume
#   LEFT/RIGHT(A/D) — step frames (paused)
#   T / G           — raise / lower max_residual threshold
#   N               — reset / clear the validator history
#   Q / ESC         — quit
# =============================================================================
if __name__ == "__main__":
    import sys
    import os
    import cv2
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
        TRAJECTORY_MIN_POINTS, TRAJECTORY_MAX_RESIDUAL,
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
    from detection.blob_filter     import filter_blobs

    print("\n" + "="*52)
    print("  detection/trajectory_fit.py — self-test")
    print("="*52)
    print(f"\n  TRAJECTORY_MIN_POINTS   = {TRAJECTORY_MIN_POINTS}")
    print(f"  TRAJECTORY_MAX_RESIDUAL = {TRAJECTORY_MAX_RESIDUAL}")
    print()
    print("  Controls:")
    print("    SPACE / A / D   — pause / step")
    print("    T / G           — raise / lower max residual")
    print("    N               — reset validator")
    print("    Q / ESC         — quit\n")

    try:
        reader = VideoReader(VIDEO_PATH)
    except RuntimeError as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    differ    = FrameDiffer(FRAME_DIFF_GAP, FRAME_DIFF_THRESHOLD)
    bgs       = BackgroundSubtractor(MOG2_HISTORY, MOG2_VAR_THRESHOLD,
                                      MOG2_DETECT_SHADOWS)
    validator = TrajectoryValidator(TRAJECTORY_MIN_POINTS,
                                     TRAJECTORY_MAX_RESIDUAL)
    max_res   = TRAJECTORY_MAX_RESIDUAL
    paused    = False
    idx       = 0

    # Rolling residual history for the plot
    PLOT_W    = 300
    PLOT_H    = 200
    res_history = deque(maxlen=60)

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

    def _draw_residual_plot(history, threshold, w=PLOT_W, h=PLOT_H):
        """Draw a simple line plot of residual over time."""
        plot = np.zeros((h, w, 3), dtype=np.uint8)
        # Threshold line
        max_plot_val = max(threshold * 2, 20.0)
        ty = int(h - (threshold / max_plot_val) * (h - 20) - 10)
        cv2.line(plot, (0, ty), (w, ty), (0, 0, 200), 1)
        cv2.putText(plot, f"threshold={threshold:.1f}",
                    (4, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100,100,255), 1)

        if len(history) < 2:
            cv2.putText(plot, "collecting points...",
                        (10, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100,100,100), 1)
            return plot

        pts = list(history)
        n   = len(pts)
        for i in range(1, n):
            if pts[i-1] is None or pts[i] is None:
                continue
            x1 = int((i-1) / (n-1) * (w-1))
            x2 = int(i     / (n-1) * (w-1))
            v1 = min(pts[i-1], max_plot_val)
            v2 = min(pts[i],   max_plot_val)
            y1 = int(h - (v1 / max_plot_val) * (h-20) - 10)
            y2 = int(h - (v2 / max_plot_val) * (h-20) - 10)
            color = (0, 255, 0) if pts[i] <= threshold else (0, 100, 255)
            cv2.line(plot, (x1, y1), (x2, y2), color, 1)

        # Current value label
        last_val = [v for v in history if v is not None]
        if last_val:
            cv2.putText(plot, f"residual={last_val[-1]:.1f}",
                        (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1)
        cv2.putText(plot, "residual over time",
                    (4, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120,120,120), 1)
        return plot

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

        all_cnts    = detect_contours(cleaned)
        detections  = filter_blobs(all_cnts, MIN_BLOB_AREA, MAX_BLOB_AREA,
                                    MIN_CIRCULARITY, MIN_SOLIDITY, MAX_ASPECT_RATIO)

        # Pick the most ball-like detection (smallest area = most compact)
        best = None
        if detections:
            best = min(detections, key=lambda d: d["area"])
            validator.update(*best["center"])
        res_history.append(validator.get_residual())

        # ── Build left panel ──────────────────────────────────────────
        left = roi_frame.copy()

        # Draw fitted arc
        arc_pts = validator.get_fitted_points()
        if arc_pts and len(arc_pts) > 1:
            color = (0, 255, 0) if validator.is_valid() else (0, 80, 255)
            for i in range(1, len(arc_pts)):
                p1, p2 = arc_pts[i-1], arc_pts[i]
                h_img, w_img = left.shape[:2]
                if (0 <= p1[0] < w_img and 0 <= p1[1] < h_img and
                    0 <= p2[0] < w_img and 0 <= p2[1] < h_img):
                    cv2.line(left, p1, p2, color, 1)

        # Draw best detection
        if best:
            x, y, w, h = best["bbox"]
            bx, by     = best["center"]
            box_color  = (0,255,0) if validator.is_valid() else (0,80,255)
            cv2.rectangle(left, (x,y), (x+w,y+h), box_color, 1)
            cv2.circle(left, (bx,by), 3, (255,0,0), -1)

        # Status label
        info  = validator.get_info()
        state = ("VALID" if info["is_valid"] else
                 "WAITING" if not info["is_ready"] else "INVALID")
        res_str = f"{info['residual']:.1f}" if info['residual'] is not None else "—"
        cv2.putText(left,
                    f"pts={info['n_points']}  res={res_str}  [{state}]",
                    (6, left.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        # ── Build right panel: residual plot ──────────────────────────
        validator.max_residual = max_res
        right = _draw_residual_plot(res_history, max_res)
        # Pad to match left panel height
        h_left = left.shape[0]
        right  = cv2.resize(right, (PLOT_W, h_left))

        disp = np.hstack([left, right])
        ch, cw = disp.shape[:2]
        if cw > DISPLAY_MAX_W or ch > DISPLAY_MAX_H:
            scale = min(DISPLAY_MAX_W / cw, DISPLAY_MAX_H / ch)
            disp  = cv2.resize(disp, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)

        status = "PAUSED" if paused else "PLAYING"
        cv2.putText(disp,
                    f"Frame {idx}/{cached_total-1}  "
                    f"max_res={max_res:.1f}  [{status}]",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,100), 1)

        cv2.imshow("trajectory_fit.py — self-test", disp)
        wait = 0 if paused else 30
        key  = cv2.waitKey(wait) & 0xFF

        if key in (27, ord("q")):
            break
        elif key == ord(" "):
            paused = not paused
        elif key in (83, ord("d")):
            idx = min(idx + 1, cached_total - 1)
        elif key in (81, ord("a")):
            idx = max(idx - 1, 0);  differ.reset()
        elif key == ord("t"):
            max_res = min(max_res + 1.0, 50.0)
            print(f"  TRAJECTORY_MAX_RESIDUAL → {max_res:.1f}")
        elif key == ord("g"):
            max_res = max(max_res - 1.0, 1.0)
            print(f"  TRAJECTORY_MAX_RESIDUAL → {max_res:.1f}")
        elif key == ord("n"):
            validator.reset()
            res_history.clear()
            print("  Validator reset.")
        else:
            if not paused:
                idx += 1

    cv2.destroyAllWindows()
    reader.release()
    print(f"\n  Final settings:")
    print(f"  TRAJECTORY_MAX_RESIDUAL = {max_res:.1f}")
    print("  Copy into config.py\n")
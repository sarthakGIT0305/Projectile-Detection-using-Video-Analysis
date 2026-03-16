# =============================================================================
# tracking/kalman_tracker.py
# =============================================================================
# Multi-object Kalman-filter tracker with Hungarian (optimal) assignment.
#
# HOW IT WORKS — THREE STEPS EVERY FRAME
# ----------------------------------------
# 1. PREDICT  — each existing track runs its Kalman filter forward one time
#    step, producing a predicted position (where we EXPECT the object to be).
#
# 2. ASSOCIATE — we build a cost matrix: distance from each prediction to
#    each new detection.  scipy.optimize.linear_sum_assignment solves the
#    Hungarian algorithm and returns the globally optimal one-to-one pairing
#    that minimises total distance.  Pairs whose distance exceeds
#    KALMAN_MAX_DISTANCE are rejected (these are new objects or noise).
#
# 3. UPDATE
#    • Matched tracks  → Kalman correct step with the detection's centroid.
#    • Unmatched tracks → increment a "missing" counter; keep predicting.
#      If the counter exceeds KALMAN_MAX_MISSING the track is deleted.
#    • Unmatched detections → spawn a new track.
#
# WHY KALMAN + HUNGARIAN?
# -----------------------
# The previous approach (nearest-centroid greedy matching) fails when two
# objects cross paths — the greedy matcher swaps their IDs.  The Hungarian
# algorithm considers ALL pairs simultaneously, finding the assignment that
# minimises the sum of all distances.  This is guaranteed optimal and
# prevents ID swaps.
#
# The Kalman filter adds prediction: if the ball is briefly occluded (e.g.
# behind a railing), the tracker predicts where it SHOULD be based on its
# velocity, and keeps the track alive for up to KALMAN_MAX_MISSING frames.
# When the ball reappears near the prediction, the same ID is maintained.
#
# KALMAN STATE MODEL
# -------------------
# State vector:  [x, y, vx, vy]   (position + velocity)
# Measurement:   [x, y]            (centroid from blob_filter)
# Transition:    constant-velocity model   (x' = x + vx·dt)
#
# The Kalman filter automatically estimates measurement noise and adjusts
# how much it trusts the prediction vs. the measurement each frame.
#
# PARAMETERS (set in config.py):
#   KALMAN_MAX_DISTANCE — max pixels between prediction and detection for match
#   KALMAN_MAX_MISSING  — frames a track survives without any detection
# =============================================================================

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    raise ImportError(
        "tracking/kalman_tracker.py requires scipy.\n"
        "Install it with:  pip install scipy"
    )


class _Track:
    """One tracked object with its own Kalman filter instance."""

    _next_id = 0

    def __init__(self, cx: int, cy: int):
        self.id = _Track._next_id
        _Track._next_id += 1

        self.kf = self._create_kalman(cx, cy)
        self.age = 0                # total frames this track has existed
        self.missing = 0            # consecutive frames without detection
        self.last_detection = None  # most recent detection dict (or None)
        self.predicted = False      # True if current position is prediction only

    @staticmethod
    def _create_kalman(cx: int, cy: int) -> cv2.KalmanFilter:
        """Build a 4-state (x, y, vx, vy) Kalman filter."""
        kf = cv2.KalmanFilter(4, 2)

        # Transition matrix: constant-velocity model
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        # Measurement matrix: we observe x, y only
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        # Process noise covariance — how much we expect the model to be wrong
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        kf.processNoiseCov[2, 2] = 5e-2   # velocity can change faster
        kf.processNoiseCov[3, 3] = 5e-2

        # Measurement noise covariance — how noisy the centroid detection is
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0

        # Initial state
        kf.statePre = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
        kf.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
        kf.errorCovPost = np.eye(4, dtype=np.float32)

        return kf

    def predict(self) -> Tuple[int, int]:
        """Advance the Kalman filter one step and return predicted (cx, cy)."""
        pred = self.kf.predict()
        return int(pred[0, 0]), int(pred[1, 0])

    def correct(self, cx: int, cy: int):
        """Feed a measurement to the Kalman filter."""
        measurement = np.array([[cx], [cy]], dtype=np.float32)
        self.kf.correct(measurement)

    @property
    def position(self) -> Tuple[int, int]:
        """Current estimated position."""
        s = self.kf.statePost
        return int(s[0, 0]), int(s[1, 0])

    @property
    def velocity(self) -> Tuple[float, float]:
        """Current estimated velocity (px/frame)."""
        s = self.kf.statePost
        return float(s[2, 0]), float(s[3, 0])

    @property
    def speed(self) -> float:
        """Scalar speed in px/frame."""
        vx, vy = self.velocity
        return (vx ** 2 + vy ** 2) ** 0.5

    def to_dict(self) -> Dict:
        """Convert track state to a plain dict for main.py compatibility."""
        cx, cy = self.position
        vx, vy = self.velocity
        # Use bbox from last detection, or build one from centroid
        if self.last_detection is not None:
            bbox = self.last_detection["bbox"]
        else:
            bbox = (cx - 4, cy - 4, 8, 8)
        return {
            "id":       self.id,
            "center":   (cx, cy),
            "bbox":     bbox,
            "velocity": (vx, vy),
            "speed":    self.speed,
            "missing":  self.missing,
            "age":      self.age,
            "predicted": self.predicted,
            "det":      self.last_detection,
        }


class KalmanTracker:
    """Multi-object tracker using Kalman filters + Hungarian assignment.

    Usage
    -----
    tracker = KalmanTracker(max_distance=60, max_missing=8)
    for detections in per_frame_detections:
        tracks = tracker.update(detections)
        for t in tracks:
            print(t.id, t.position, t.missing)
    """

    def __init__(self, max_distance: float = 60, max_missing: int = 8):
        self.max_distance = max_distance
        self.max_missing  = max_missing
        self._tracks: List[_Track] = []

    def update(self, detections: List[Dict]) -> List[_Track]:
        """Process one frame of detections.

        Parameters
        ----------
        detections : list of dict
            Each dict must have a ``"center"`` key → (cx, cy).
            Optionally ``"bbox"`` → (x, y, w, h) and other fields from
            blob_filter.

        Returns
        -------
        list of _Track
            All currently active tracks (includes predicting ones).
        """
        # 1. PREDICT — advance all existing tracks
        predictions = []
        for track in self._tracks:
            pred = track.predict()
            predictions.append(pred)
            track.age += 1

        # 2. ASSOCIATE — Hungarian algorithm
        n_tracks = len(self._tracks)
        n_dets   = len(detections)

        matched_track_ids = set()
        matched_det_ids   = set()

        if n_tracks > 0 and n_dets > 0:
            # Build cost matrix: distance from each prediction to each detection
            cost = np.zeros((n_tracks, n_dets), dtype=np.float32)
            for i, pred in enumerate(predictions):
                for j, det in enumerate(detections):
                    dx = pred[0] - det["center"][0]
                    dy = pred[1] - det["center"][1]
                    cost[i, j] = (dx ** 2 + dy ** 2) ** 0.5

            # Solve optimal assignment
            row_idx, col_idx = linear_sum_assignment(cost)

            for r, c in zip(row_idx, col_idx):
                if cost[r, c] <= self.max_distance:
                    matched_track_ids.add(r)
                    matched_det_ids.add(c)

                    # 3a. UPDATE matched tracks
                    track = self._tracks[r]
                    det   = detections[c]
                    cx, cy = det["center"]
                    track.correct(cx, cy)
                    track.missing = 0
                    track.predicted = False
                    track.last_detection = det

        # 3b. UNMATCHED tracks — increment missing counter
        for i in range(n_tracks):
            if i not in matched_track_ids:
                self._tracks[i].missing += 1
                self._tracks[i].predicted = True

        # 3c. UNMATCHED detections — spawn new tracks
        for j in range(n_dets):
            if j not in matched_det_ids:
                det = detections[j]
                cx, cy = det["center"]
                new_track = _Track(cx, cy)
                new_track.last_detection = det
                self._tracks.append(new_track)

        # Prune dead tracks
        self._tracks = [t for t in self._tracks if t.missing <= self.max_missing]

        return [t.to_dict() for t in self._tracks]

    def active_tracks(self) -> List[Dict]:
        """Return currently active tracks as dicts."""
        return [t.to_dict() for t in self._tracks]

    def reset(self):
        """Remove all tracks and reset ID counter."""
        self._tracks.clear()
        _Track._next_id = 0


# =============================================================================
# SELF-TEST
# Run:  python tracking/kalman_tracker.py
#
# TWO-PANEL window:
#   Left  — annotated ROI frame with:
#             • green box + track ID for each active track
#             • yellow dot = predicted position, blue dot = corrected position
#             • "PRED(n)" label if track is predicting (missing detection)
#   Right — track info table showing each track's:
#             • ID, age, missing count, speed (px/frame)
#
# The ID test:
#   Throw the ball. The ID number on the box should stay the SAME for the
#   entire flight.  If it changes mid-flight, KALMAN_MAX_DISTANCE is too low.
#   Press T to raise it.
#
# The prediction test:
#   If the ball briefly disappears, the box should persist with "PRED(n)".
#   If it vanishes immediately, press M to raise max_missing.
#
# Controls:
#   SPACE           — pause / resume
#   LEFT/RIGHT(A/D) — step frames (paused)
#   T / G           — raise / lower KALMAN_MAX_DISTANCE
#   M / N           — raise / lower KALMAN_MAX_MISSING
#   R               — reset all tracks
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
        ROI_PERCENT, ROI_ANCHOR_X, ROI_ANCHOR_Y,
        DISPLAY_MAX_W, DISPLAY_MAX_H,
        COLOR_BBOX, COLOR_CENTER, COLOR_ID,
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

    print("\n" + "="*52)
    print("  tracking/kalman_tracker.py — self-test")
    print("="*52)
    print(f"\n  KALMAN_MAX_DISTANCE = {KALMAN_MAX_DISTANCE}")
    print(f"  KALMAN_MAX_MISSING  = {KALMAN_MAX_MISSING}")
    print()
    print("  Controls:")
    print("    SPACE / A / D   — pause / step")
    print("    T / G           — raise / lower max distance")
    print("    M / N           — raise / lower max missing")
    print("    R               — reset all tracks")
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
    paused   = False
    max_dist = KALMAN_MAX_DISTANCE
    max_miss = KALMAN_MAX_MISSING

    # Pre-cache frames
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

    TABLE_W = 320  # width of the info table panel
    idx = 0

    def _draw_table(tracks, h_img):
        """Draw a track-info table as an image."""
        table = np.zeros((h_img, TABLE_W, 3), dtype=np.uint8)
        y = 24
        cv2.putText(table, "ID   age  miss  speed   pos",
                    (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
        y += 20
        for t in tracks:
            px, py = t["center"]
            color = (0, 255, 0) if not t["predicted"] else (0, 100, 255)
            pred_label = f" PRED({t['missing']})" if t["predicted"] else ""
            line = (f"{t['id']:<4d} {t['age']:<4d} {t['missing']:<5d} "
                    f"{t['speed']:>6.1f}  ({px},{py}){pred_label}")
            cv2.putText(table, line,
                        (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1)
            y += 16
            if y > h_img - 10:
                break
        return table

    while idx < cached_total:
        frame        = frames_cache[idx]
        roi_frame, _ = apply_roi(frame, ROI_PERCENT, ROI_ANCHOR_X, ROI_ANCHOR_Y)
        gray         = to_gray(roi_frame, CLAHE_CLIP_LIMIT,
                               CLAHE_TILE_GRID, ENABLE_CLAHE)

        # Motion masks
        th_mask  = apply_tophat(gray, mode=TOPHAT_MODE,
                                kernel_size=TOPHAT_KERNEL_SIZE,
                                threshold=TOPHAT_THRESHOLD) \
                   if ENABLE_TOPHAT else None
        df_mask  = differ.update(gray)  if ENABLE_FRAME_DIFF else None
        mg_mask  = bg_sub.update(gray)  if ENABLE_BG_SUB     else None
        combined = combine_masks(th_mask, df_mask, mg_mask, MASK_COMBINE_MODE)
        cleaned  = morph_clean(combined, kernel_size=MORPH_KERNEL_SIZE) \
                   if ENABLE_MORPH_CLEAN else combined

        # Detection
        contours   = detect_contours(cleaned)
        detections = filter_blobs(contours, MIN_BLOB_AREA, MAX_BLOB_AREA,
                                  MIN_CIRCULARITY, MIN_SOLIDITY,
                                  MAX_ASPECT_RATIO)

        # Tracking
        tracker.max_distance = max_dist
        tracker.max_missing  = max_miss
        tracks = tracker.update(detections)

        # ── Left panel: annotated frame ───────────────────────────────
        left = roi_frame.copy()
        for t in tracks:
            px, py = t["center"]
            vx, vy = t["velocity"]

            # Predicted position dot (yellow)
            cv2.circle(left, (px, py), 4, (0, 255, 255), -1)

            # If matched, also show corrected position (blue)
            if not t["predicted"] and t["det"] is not None:
                dcx, dcy = t["det"]["center"]
                cv2.circle(left, (dcx, dcy), 3, COLOR_CENTER, -1)

            # Bounding box
            if t["det"] is not None:
                bx, by, bw, bh = t["det"]["bbox"]
                box_color = COLOR_BBOX if not t["predicted"] else (0, 100, 255)
                cv2.rectangle(left, (bx, by), (bx+bw, by+bh), box_color, 1)

            # Label
            label = f"ID:{t['id']}"
            if t["predicted"]:
                label += f" PRED({t['missing']})"
            cv2.putText(left, label,
                        (px + 6, py - 6), FONT_FACE, FONT_SCALE,
                        COLOR_ID, FONT_THICKNESS)

        h_img = left.shape[0]

        # ── Right panel: track table ──────────────────────────────────
        right = _draw_table(tracks, h_img)

        disp = np.hstack([left, right])
        ch, cw = disp.shape[:2]
        if cw > DISPLAY_MAX_W or ch > DISPLAY_MAX_H:
            scale = min(DISPLAY_MAX_W / cw, DISPLAY_MAX_H / ch)
            disp  = cv2.resize(disp, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)

        status = "PAUSED" if paused else "PLAYING"
        cv2.putText(disp,
                    f"Frame {idx}/{cached_total-1}  tracks={len(tracks)}  "
                    f"dist={max_dist}  miss={max_miss}  [{status}]",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 100), 1)

        cv2.imshow("kalman_tracker.py — self-test", disp)
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
            max_dist = min(max_dist + 10, 300)
            print(f"  KALMAN_MAX_DISTANCE → {max_dist}")
        elif key == ord("g"):
            max_dist = max(max_dist - 10, 10)
            print(f"  KALMAN_MAX_DISTANCE → {max_dist}")
        elif key == ord("m"):
            max_miss = min(max_miss + 2, 30)
            print(f"  KALMAN_MAX_MISSING → {max_miss}")
        elif key == ord("n"):
            max_miss = max(max_miss - 1, 1)
            print(f"  KALMAN_MAX_MISSING → {max_miss}")
        elif key == ord("r"):
            tracker.reset()
            print("  All tracks reset.")
        else:
            if not paused:
                idx += 1

    cv2.destroyAllWindows()
    reader.release()
    print(f"\n  Final settings:")
    print(f"    KALMAN_MAX_DISTANCE = {max_dist}")
    print(f"    KALMAN_MAX_MISSING  = {max_miss}")
    print("  Copy into config.py\n")

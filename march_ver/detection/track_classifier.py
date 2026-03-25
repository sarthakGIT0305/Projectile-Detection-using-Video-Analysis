# =============================================================================
# detection/track_classifier.py
# =============================================================================
# Classifies tracked objects as "projectile", "noise", or "pending" using
# multiple heuristics that work across time.  Only tracks classified as
# "projectile" should be drawn / exported.
#
# WHY THIS IS NEEDED
# -------------------
# The blob-filter and trajectory-validator work well individually, but the
# pipeline still draws every active track.  Trees swaying, birds, insects,
# and compression artefacts all produce tracks that persist for several
# frames and clutter the output.
#
# This module sits AFTER tracking + trail storage and BEFORE drawing.
# It looks at each track's full history and applies physics-based rules
# that a real thrown object must satisfy:
#
#   1. Minimum age          — flicker / 1-frame blobs rejected
#   2. Minimum displacement — stationary vibration rejected (trees, poles)
#   3. Speed range          — too slow (leaves) or too fast (teleporting) rejected
#   4. Path efficiency      — net displacement / total path length;
#                             trees oscillate (ratio ~0.1), projectiles fly (ratio ~0.7)
#   5. Spatial spread       — bounding box of all points must span enough pixels
#   6. Consistent direction — random jitter doesn't keep moving one way
#   7. Parabola fit         — only y = ax² + bx + c trajectories pass
#
# A track starts as "pending".  Once it reaches CLASSIFIER_MIN_AGE it is
# evaluated.  If ALL checks pass it becomes "projectile"; otherwise "noise".
# Classification is re-evaluated every frame so a track can recover if its
# early frames were noisy but its later frames form a good parabola.
#
# PARAMETERS (set in config.py):
#   CLASSIFIER_MIN_AGE              — frames before classification starts
#   CLASSIFIER_MIN_DISPLACEMENT     — min net px from first to latest position
#   CLASSIFIER_MIN_SPEED            — min speed in px/frame
#   CLASSIFIER_MAX_SPEED            — max speed in px/frame
#   CLASSIFIER_MIN_PATH_EFFICIENCY  — min ratio of net/total path length
#   CLASSIFIER_MIN_SPATIAL_SPREAD   — min bounding-box span (px) of all points
#   CLASSIFIER_MIN_DIRECTION_RATIO  — fraction of consistent-direction steps
# =============================================================================

import math
import warnings
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class TrackClassifier:
    """Classifies tracks as 'projectile', 'noise', or 'pending'.

    Usage
    -----
    classifier = TrackClassifier(min_age=5, ...)
    # every frame:
    classifier.update(tracks, trail_store, validators)
    label = classifier.get(track_id)  # -> "projectile" / "noise" / "pending"
    """

    def __init__(self,
                 min_age: int = 8,
                 min_displacement: float = 50.0,
                 min_speed: float = 2.0,
                 max_speed: float = 80.0,
                 min_path_efficiency: float = 0.4,
                 min_spatial_spread: float = 40.0,
                 min_direction_ratio: float = 0.7):
        self.min_age = min_age
        self.min_displacement = min_displacement
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.min_path_efficiency = min_path_efficiency
        self.min_spatial_spread = min_spatial_spread
        self.min_direction_ratio = min_direction_ratio

        # track_id -> "projectile" | "noise" | "pending"
        self._labels: Dict[int, str] = {}
        # track_id -> (first_cx, first_cy) — recorded when first seen
        self._first_position: Dict[int, Tuple[int, int]] = {}

    def update(self, tracks: List[Dict], trail_store, validators: dict):
        """Classify all active tracks.

        Parameters
        ----------
        tracks      : list of track dicts from KalmanTracker.update()
        trail_store : TrailStore instance (for position history)
        validators  : dict of {track_id: TrajectoryValidator} (for parabola fit)
        """
        active_ids = set()

        for t in tracks:
            tid = t["id"]
            active_ids.add(tid)
            age = t.get("age", 0)
            cx, cy = t["center"]

            # Record first position
            if tid not in self._first_position:
                self._first_position[tid] = (cx, cy)

            # Not enough frames yet
            if age < self.min_age:
                if self._labels.get(tid) != "projectile":
                    self._labels[tid] = "pending"
                continue

            # Run all checks
            new_label = self._classify(t, trail_store, validators)
            if self._labels.get(tid) == "projectile":
                self._labels[tid] = "projectile"
            else:
                self._labels[tid] = new_label

        # Prune dead tracks
        dead = [tid for tid in self._labels if tid not in active_ids]
        for tid in dead:
            del self._labels[tid]
            self._first_position.pop(tid, None)

    def _classify(self, track: dict, trail_store, validators: dict) -> str:
        """Evaluate a single track against all heuristics."""
        tid = track["id"]
        cx, cy = track["center"]

        # ── Check 1: Net displacement ─────────────────────────────────
        first = self._first_position.get(tid, (cx, cy))
        dx = cx - first[0]
        dy = cy - first[1]
        displacement = math.sqrt(dx * dx + dy * dy)

        if displacement < self.min_displacement:
            return "noise"

        # ── Check 2: Speed range ──────────────────────────────────────
        speed = track.get("speed", 0.0)
        if speed < self.min_speed:
            return "noise"
        if speed > self.max_speed:
            return "noise"

        points = trail_store.get_observed_points(tid)

        # ── Check 3: Path efficiency ──────────────────────────────────
        # Net displacement / total path length.  A projectile flies
        # efficiently (~0.6-1.0).  Trees oscillate back and forth so
        # their total path is much longer than net displacement (~0.05-0.3).
        if len(points) >= 3:
            total_path = 0.0
            for i in range(1, len(points)):
                sdx = points[i][0] - points[i-1][0]
                sdy = points[i][1] - points[i-1][1]
                total_path += math.sqrt(sdx*sdx + sdy*sdy)

            # Net displacement from first to last observed point
            net_dx = points[-1][0] - points[0][0]
            net_dy = points[-1][1] - points[0][1]
            net_disp = math.sqrt(net_dx*net_dx + net_dy*net_dy)

            if total_path > 1.0:
                efficiency = net_disp / total_path
                if efficiency < self.min_path_efficiency:
                    return "noise"

        # ── Check 4: Spatial spread ───────────────────────────────────
        # The bounding box of all observed points must span enough
        # pixels.  Tree jitter stays in a tight cluster.
        if len(points) >= 3:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x_span = max(xs) - min(xs)
            y_span = max(ys) - min(ys)
            max_span = max(x_span, y_span)
            if max_span < self.min_spatial_spread:
                return "noise"

        # ── Check 5: Consistent motion direction ──────────────────────
        if len(points) >= 3:
            overall_dx = points[-1][0] - points[0][0]
            if abs(overall_dx) < 3:
                consistent = self._check_vertical_consistency(points)
            else:
                consistent = self._check_horizontal_consistency(points, overall_dx)

            if not consistent:
                return "noise"

        # ── Check 6: Clean Arc Fast-Track ─────────────────────────────
        # If the track forms a beautiful, unmistakable arc, skip the strict validation
        if self._is_clean_arc(points):
            return "projectile"

        # ── Check 7: Trajectory validation (strict parabola fit) ──────
        val = validators.get(tid)
        if val is not None:
            if val.is_ready():
                if not val.is_valid():
                    return "noise"
            else:
                return "pending"

        return "projectile"

    def _is_clean_arc(self, points: List[Tuple[int, int]]) -> bool:
        """
        Robust heuristic to identify clean, parabolic arcs (like a ball flight)
        even if strict per-frame velocity checks fail.
        """
        if len(points) < 8:
            return False
            
        xs = np.array([p[0] for p in points], dtype=np.float64)
        ys = np.array([p[1] for p in points], dtype=np.float64)
        
        # 1. Path Efficiency (must be highly direct)
        total_path = 0.0
        for i in range(1, len(points)):
            sdx = points[i][0] - points[i-1][0]
            sdy = points[i][1] - points[i-1][1]
            total_path += math.sqrt(sdx*sdx + sdy*sdy)
            
        net_dx = points[-1][0] - points[0][0]
        net_dy = points[-1][1] - points[0][1]
        net_disp = math.sqrt(net_dx*net_dx + net_dy*net_dy)
        
        if total_path == 0 or (net_disp / total_path) < 0.8:
            return False # Too much jitter/wobble
            
        # 2. Fit a parabola and check absolute visual residual
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                coeffs = np.polyfit(xs, ys, deg=2)
            y_pred = np.polyval(coeffs, xs)
            residuals = np.abs(ys - y_pred)
            mean_residual = float(np.mean(residuals))
            
            # If the mean pixel error is very low (visually perfect arc)
            # AND it has travelled a decent distance, it's a projectile.
            if mean_residual < 15.0 and (abs(net_dx) > 100 or abs(net_dy) > 100):
                return True
        except Exception:
            pass
            
        return False

    def _check_horizontal_consistency(self, points: list, overall_dx: float) -> bool:
        """Check that most frame-to-frame x-movements agree with overall direction."""
        if len(points) < 2:
            return True
        forward = 0
        total = 0
        for i in range(1, len(points)):
            step_dx = points[i][0] - points[i - 1][0]
            if abs(step_dx) < 1:
                continue  # skip near-zero steps
            total += 1
            if (step_dx > 0) == (overall_dx > 0):
                forward += 1
        if total == 0:
            return True
        return (forward / total) >= self.min_direction_ratio

    def _check_vertical_consistency(self, points: list) -> bool:
        """For mostly-vertical motion, check y moves consistently."""
        if len(points) < 2:
            return True
        overall_dy = points[-1][1] - points[0][1]
        if abs(overall_dy) < 3:
            return False  # barely moved at all — noise
        forward = 0
        total = 0
        for i in range(1, len(points)):
            step_dy = points[i][1] - points[i - 1][1]
            if abs(step_dy) < 1:
                continue
            total += 1
            if (step_dy > 0) == (overall_dy > 0):
                forward += 1
        if total == 0:
            return True
        return (forward / total) >= self.min_direction_ratio

    def get(self, track_id: int) -> str:
        """Return classification for a track: 'projectile', 'noise', or 'pending'."""
        return self._labels.get(track_id, "pending")

    def get_all(self) -> Dict[int, str]:
        """Return all current classifications."""
        return dict(self._labels)

    def reset(self):
        """Clear all classifications."""
        self._labels.clear()
        self._first_position.clear()

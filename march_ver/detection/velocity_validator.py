# =============================================================================
# detection/velocity_validator.py
# =============================================================================
"""Velocity profile analysis for tracked objects.

This validator checks that the horizontal velocity (dx) of a track is relatively
stable and that the vertical motion (dy) follows a roughly linear trend –
characteristics of a projectile under gravity.

The checks are configurable via the following parameters defined in `config.py`:
- VELOCITY_DX_MAX_VARIANCE: maximum allowed variance of horizontal velocity.
- VELOCITY_DY_MIN_R2: minimum R^2 of a linear fit to vertical displacement vs.
  time (frames).
- VELOCITY_MAX_DIRECTION_FLIPS: maximum allowed number of direction changes
  (sign flips) in the horizontal velocity.
"""

import numpy as np
from collections import deque
from typing import Deque, List, Tuple

class VelocityValidator:
    """Validate velocity profile of a rolling centroid history.

    The validator maintains a fixed‑length history of centroid positions and
    provides a boolean `is_valid()` method based on the configured thresholds.
    """

    def __init__(self,
                 max_history: int = 30,
                 dx_max_variance: float = 25.0,
                 dy_min_r2: float = 0.55,
                 max_direction_flips: int = 1):
        self._history: Deque[Tuple[int, int]] = deque(maxlen=max_history)
        self.dx_max_variance = dx_max_variance
        self.dy_min_r2 = dy_min_r2
        self.max_direction_flips = max_direction_flips

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def update(self, cx: int, cy: int) -> None:
        """Add a new centroid observation.

        Parameters
        ----------
        cx, cy: int
            Centroid coordinates in pixel space.
        """
        self._history.append((cx, cy))

    def is_ready(self) -> bool:
        """Return ``True`` when enough points are available for a reliable check.
        """
        return len(self._history) >= 5  # arbitrary minimum for variance calc

    def is_valid(self) -> bool:
        """Run the three velocity checks and return ``True`` only if all pass.
        """
        if not self.is_ready():
            return False

        # Compute per‑frame deltas
        xs = np.array([p[0] for p in self._history], dtype=np.float64)
        ys = np.array([p[1] for p in self._history], dtype=np.float64)
        dt = np.arange(len(self._history))

        # Horizontal velocity (dx per frame)
        dx = np.diff(xs)
        if len(dx) == 0:
            return False
        variance_dx = float(np.var(dx))
        if variance_dx > self.dx_max_variance:
            return False

        # Direction flips in horizontal velocity
        sign_changes = np.sum(np.diff(np.sign(dx)) != 0)
        if sign_changes > self.max_direction_flips:
            return False

        # Vertical motion linearity – fit dy vs time
        dy = np.diff(ys)
        if len(dy) < 2:
            return False
        # Cumulative vertical displacement over time
        cum_dy = np.cumsum(dy)
        # Linear regression (least squares) to get R^2
        A = np.vstack([dt[1:], np.ones_like(dt[1:])]).T
        slope, intercept = np.linalg.lstsq(A, cum_dy, rcond=None)[0]
        pred = slope * dt[1:] + intercept
        ss_res = np.sum((cum_dy - pred) ** 2)
        ss_tot = np.sum((cum_dy - np.mean(cum_dy)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
        if r2 < self.dy_min_r2:
            return False

        return True

    # ---------------------------------------------------------------------
    # Utility
    # ---------------------------------------------------------------------
    def reset(self) -> None:
        self._history.clear()

import re

with open("main.py", "r", encoding="utf-8") as f:
    code = f.read()

# 1. Update _check_projectile_arc
old_func = '''def _check_projectile_arc(trail_store, tracks, frame_width):
    """Check each track's trail for a big downward parabola.

    Returns a set of track IDs that look like a thrown ball:
      - Horizontal span >= ARC_MIN_SPAN_RATIO of frame_width
      - Parabola fit opens downward (positive 'a' in image y-down coords)
      - Fit residual <= ARC_MAX_RESIDUAL
      - At least ARC_MIN_POINTS observed points
    """
    flagged = set()
    min_span = frame_width * ARC_MIN_SPAN_RATIO

    for t in tracks:
        tid = t["id"]
        pts = trail_store.get_observed_points(tid)
        if len(pts) < ARC_MIN_POINTS:
            continue

        xs = np.array([p[0] for p in pts], dtype=np.float64)
        ys = np.array([p[1] for p in pts], dtype=np.float64)

        # Check horizontal span
        x_span = xs.max() - xs.min()
        if x_span < min_span:
            continue

        # Fit parabola: y = a*x^2 + b*x + c
        try:
            coeffs = np.polyfit(xs, ys, deg=2)
        except (np.linalg.LinAlgError, ValueError):
            continue

        a = coeffs[0]

        # In image coordinates y increases downward.
        # A ball thrown in an arc goes UP (y decreases) then DOWN (y increases).
        # This makes a parabola that opens DOWNWARD in real-world coords,
        # but opens UPWARD in image coords (positive 'a').
        # So we want a > 0 for a downward parabola (ball arc).
        if a <= 0:
            continue

        # Check fit quality
        y_pred = np.polyval(coeffs, xs)
        residual = float(np.mean(np.abs(ys - y_pred)))
        if residual > ARC_MAX_RESIDUAL:
            continue

        flagged.add(tid)

    return flagged'''

new_func = '''def _check_projectile_arc(trail_store, tracks, frame_width):
    """Check each track's trail for a big downward parabola.

    Returns a tuple (flagged_ids, arcs):
      - flagged_ids: set of track IDs that look like a thrown ball
      - arcs: dict mapping track ID to a list of (x,y) points for the smooth fitted arc
    """
    flagged = set()
    arcs = {}
    min_span = frame_width * ARC_MIN_SPAN_RATIO

    for t in tracks:
        tid = t["id"]
        pts = trail_store.get_observed_points(tid)
        if len(pts) < ARC_MIN_POINTS:
            continue

        xs = np.array([p[0] for p in pts], dtype=np.float64)
        ys = np.array([p[1] for p in pts], dtype=np.float64)

        # Check horizontal span
        x_span = xs.max() - xs.min()
        if x_span < min_span:
            continue

        # Fit parabola: y = a*x^2 + b*x + c
        try:
            coeffs = np.polyfit(xs, ys, deg=2)
        except (np.linalg.LinAlgError, ValueError):
            continue

        a = coeffs[0]

        if a <= 0:
            continue

        # Check fit quality
        y_pred = np.polyval(coeffs, xs)
        residual = float(np.mean(np.abs(ys - y_pred)))
        if residual > ARC_MAX_RESIDUAL:
            continue

        flagged.add(tid)
        
        # Save a smooth fitted arc for drawing permanently
        x_min, x_max = xs.min(), xs.max()
        x_range = np.linspace(x_min, x_max, 100)
        y_range = np.polyval(coeffs, x_range)
        arcs[tid] = [(int(x), int(y)) for x, y in zip(x_range, y_range)]

    return flagged, arcs'''

if old_func in code:
    code = code.replace(old_func, new_func, 1)
else:
    print("WARNING: Could not find old _check_projectile_arc function.")

# 2. Add saved_arcs initialization
old_init = '''    bgs         = BackgroundSubtractor(
                      MOG2_HISTORY, MOG2_VAR_THRESHOLD, MOG2_DETECT_SHADOWS
                  )
    tracker     = KalmanTracker(KALMAN_MAX_DISTANCE, KALMAN_MAX_MISSING)
    trail_store = TrailStore(TRAIL_LENGTH)
    classifier  = TrackClassifier() if CLASSIFIER_ACTIVE else None
    validators  = {}

    paused        = False'''

new_init = '''    bgs         = BackgroundSubtractor(
                      MOG2_HISTORY, MOG2_VAR_THRESHOLD, MOG2_DETECT_SHADOWS
                  )
    tracker     = KalmanTracker(KALMAN_MAX_DISTANCE, KALMAN_MAX_MISSING)
    trail_store = TrailStore(TRAIL_LENGTH)
    classifier  = TrackClassifier() if CLASSIFIER_ACTIVE else None
    validators  = {}
    saved_projectile_arcs = {}  # Stores permanent arcs

    paused        = False'''

if old_init in code:
    code = code.replace(old_init, new_init, 1)
else:
    print("WARNING: Could not find init block.")

# 3. Update alarm logic saving the permanent arcs
old_arc_check = '''        # ── Projectile arc check ──────────────────────────────────────
        arc_ids = set()
        if TRAIL_ACTIVE and TRACKING_ACTIVE:
            arc_ids = _check_projectile_arc(
                trail_store, tracks, display_frame.shape[1]
            )
            if arc_ids:
                n_projectiles = len(arc_ids)'''

new_arc_check = '''        # ── Projectile arc check ──────────────────────────────────────
        arc_ids = set()
        if TRAIL_ACTIVE and TRACKING_ACTIVE:
            arc_ids, new_arcs = _check_projectile_arc(
                trail_store, tracks, display_frame.shape[1]
            )
            if arc_ids:
                n_projectiles = len(arc_ids)
                # Permanently save these winning arcs
                for tid, pts in new_arcs.items():
                    saved_projectile_arcs[tid] = pts'''

if old_arc_check in code:
    code = code.replace(old_arc_check, new_arc_check, 1)
else:
    print("WARNING: Could not find arc check block.")

# 4. Draw the permanent arcs
old_draw = '''        # ── Stage 13: Draw ────────────────────────────────────────────
        if TRACKING_ACTIVE:
            output = _draw_tracks(
                display_frame, tracks, trail_store, validators,
                classifier=classifier,
                show_trail=TRAIL_ACTIVE
            )
            hud_tracks = len(tracks)
        else:
            output = _draw_projectile_detections(display_frame, projectile_detections)
            hud_tracks = 0

        output = _draw_hud(
            output, frame_idx,
            hud_tracks, len(draw_detections),
            n_projectiles,
            bgs.is_warmed_up
        )'''

new_draw = '''        # ── Stage 13: Draw ────────────────────────────────────────────
        if TRACKING_ACTIVE:
            output = _draw_tracks(
                display_frame, tracks, trail_store, validators,
                classifier=classifier,
                show_trail=TRAIL_ACTIVE
            )
            hud_tracks = len(tracks)
        else:
            output = _draw_projectile_detections(display_frame, projectile_detections)
            hud_tracks = 0

        # Draw the permanent projectile arcs boldly!
        for tid, arc_pts in saved_projectile_arcs.items():
            color = (0, 255, 255) # Bold magenta/yellow? Let's use vibrant neon BGR (0, 255, 255) is Yellow, (255, 0, 255) is Magenta
            # Cycles through some bold colors
            bold_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 255), (0, 255, 255)]
            color = bold_colors[tid % len(bold_colors)]
            h_img, w_img = output.shape[:2]
            for i in range(1, len(arc_pts)):
                p1, p2 = arc_pts[i-1], arc_pts[i]
                if (0 <= p1[0] < w_img and 0 <= p1[1] < h_img and
                    0 <= p2[0] < w_img and 0 <= p2[1] < h_img):
                    cv2.line(output, p1, p2, color, 4) # Thickness 4 for bold

        output = _draw_hud(
            output, frame_idx,
            hud_tracks, len(draw_detections),
            n_projectiles,
            bgs.is_warmed_up
        )'''

if old_draw in code:
    code = code.replace(old_draw, new_draw, 1)
else:
    print("WARNING: Could not find draw block.")

with open("main.py", "w", encoding="utf-8") as f:
    f.write(code)

print("success!")

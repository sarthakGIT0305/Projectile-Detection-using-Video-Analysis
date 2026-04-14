# =============================================================================
# main.py
# =============================================================================
# Orchestrates the full detection-tracking pipeline.
# Pipeline order every frame:
#   1.  Read frame 
#   2.  ROI crop
#   3.  CLAHE grayscale
#   4.  Top-hat spatial filter 
#   5.  3-frame temporal difference
#   6.  MOG2 background subtraction
#   7.  Combine motion masks
#   8.  Morphological cleaning
#   9.  Contour detection
#   10. Blob filtering  (area / circularity / solidity / aspect ratio)
#   11. Kalman tracker  (predict → Hungarian match → correct)
#   12. Trail store     (record history per track)
#   13. Trajectory validation  (parabola fit — optional)
#   14. Draw results
#   15. Show debug mask windows  (optional)
#   16. Display output
#
# Every stage is independently gated by its ENABLE_ flag in config.py.
# Turn a flag off to skip that stage entirely — nothing else changes.
# =============================================================================

import cv2
import numpy as np
import sys

from config import (
    VIDEO_PATH,
    PROCESS_SCALE, FRAME_SKIP,
    ENABLE_ROI, ROI_PERCENT, ROI_ANCHOR_X, ROI_ANCHOR_Y,
    ENABLE_CLAHE, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID,
    ENABLE_TOPHAT, TOPHAT_KERNEL_SIZE, TOPHAT_THRESHOLD, TOPHAT_MODE,
    ENABLE_FRAME_DIFF, FRAME_DIFF_GAP, FRAME_DIFF_THRESHOLD,
    ENABLE_BG_SUB, MOG2_HISTORY, MOG2_VAR_THRESHOLD, MOG2_DETECT_SHADOWS,
    MASK_COMBINE_MODE,
    ENABLE_MORPH_CLEAN, MORPH_KERNEL_SIZE,
    ENABLE_BLOB_FILTER,
    MIN_BLOB_AREA, MAX_BLOB_AREA,
    MIN_CIRCULARITY, MIN_SOLIDITY, MAX_ASPECT_RATIO,
    ENABLE_ISOLATION_FILTER, ISOLATION_RADIUS, ISOLATION_MIN_CLUSTER,
    ENABLE_TRAJECTORY, TRAJECTORY_MIN_POINTS, TRAJECTORY_MAX_RESIDUAL,
    ENABLE_KALMAN, KALMAN_MAX_DISTANCE, KALMAN_MAX_MISSING,
    ENABLE_TRAIL, TRAIL_LENGTH,
    ARC_MIN_SPAN_RATIO, ARC_MAX_RESIDUAL, ARC_MIN_POINTS,
    ENABLE_CLASSIFIER,
    CLASSIFIER_MIN_AGE, CLASSIFIER_MIN_DISPLACEMENT,
    CLASSIFIER_MIN_SPEED, CLASSIFIER_MAX_SPEED,
    CLASSIFIER_MIN_PATH_EFFICIENCY, CLASSIFIER_MIN_SPATIAL_SPREAD,
    CLASSIFIER_MIN_DIRECTION_RATIO,
    CLASSIFIER_SHOW_PENDING, CLASSIFIER_SHOW_NOISE,
    ENABLE_DEBUG_VIEW, TEMP_DISABLE_TRACKING,
    DEBUG_SHOW_TOPHAT, DEBUG_SHOW_FRAME_DIFF,
    DEBUG_SHOW_BG_SUB, DEBUG_SHOW_COMBINED, DEBUG_SHOW_CLEANED,
    DEBUG_SHOW_CONTOURS, DEBUG_SHOW_FILTER_AREA,
    DEBUG_SHOW_FILTER_ASPECT, DEBUG_SHOW_FILTER_CIRCULAR,
    DEBUG_SHOW_FILTER_SOLIDITY, DEBUG_SHOW_FINAL_DETECTION,
    DISPLAY_MAX_W, DISPLAY_MAX_H,
    COLOR_BBOX, COLOR_ID, COLOR_CENTER, COLOR_TRAIL,
    FONT_FACE, FONT_SCALE, FONT_THICKNESS,
)

from video_reader                import VideoReader
from preprocessing.roi           import apply_roi
from preprocessing.clahe_gray    import to_gray
from motion.tophat               import apply_tophat
from motion.frame_diff           import FrameDiffer
from motion.bg_sub               import BackgroundSubtractor
from motion.mask_combine         import combine_masks
from motion.morph_clean          import clean_mask
from detection.contour_detect    import detect_contours
from detection.blob_filter       import filter_blobs
from detection.isolation_filter  import reject_clustered_blobs
from detection.trajectory_fit    import TrajectoryValidator
from detection.track_classifier  import TrackClassifier
from tracking.kalman_tracker     import KalmanTracker
from tracking.trail_store        import TrailStore


# Colour palette that cycles by track ID
TRACK_COLORS = [
    (0,255,0), (0,200,255), (255,100,0), (200,0,255),
    (0,255,200), (255,255,0), (100,100,255), (255,0,100),
]

TRACKING_ACTIVE   = ENABLE_KALMAN and not TEMP_DISABLE_TRACKING
TRAJECTORY_ACTIVE = ENABLE_TRAJECTORY and TRACKING_ACTIVE
TRAIL_ACTIVE      = ENABLE_TRAIL and TRACKING_ACTIVE
CLASSIFIER_ACTIVE = ENABLE_CLASSIFIER and TRACKING_ACTIVE

DEBUG_WINDOWS = [
    "Debug: top-hat",
    "Debug: frame diff",
    "Debug: MOG2",
    "Debug: combined",
    "Debug: cleaned",
    "Debug: contours",
    "Debug: area pass",
    "Debug: aspect pass",
    "Debug: circularity pass",
    "Debug: solidity pass",
    "Debug: final projectile",
]

 
# ── Helpers ───────────────────────────────────────────────────────────────

def _resize_for_display(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w > DISPLAY_MAX_W or h > DISPLAY_MAX_H:
        scale = min(DISPLAY_MAX_W / w, DISPLAY_MAX_H / h)
        return cv2.resize(frame, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_LINEAR)
    return frame


def _show_debug_masks(masks: dict):
    """Open/update individual mask windows for each enabled debug stage."""
    configs = [
        ("tophat",     DEBUG_SHOW_TOPHAT,     "Debug: top-hat"),
        ("frame_diff", DEBUG_SHOW_FRAME_DIFF, "Debug: frame diff"),
        ("bg_sub",     DEBUG_SHOW_BG_SUB,     "Debug: MOG2"),
        ("combined",   DEBUG_SHOW_COMBINED,   "Debug: combined"),
        ("cleaned",    DEBUG_SHOW_CLEANED,    "Debug: cleaned"),
    ]
    for key, enabled, title in configs:
        if enabled and masks.get(key) is not None:
            m = masks[key]
            vis = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR) \
                  if len(m.shape) == 2 else m
            cv2.imshow(title, _resize_for_display(vis))


def _wants_filter_layer_debug() -> bool:
    return any([
        DEBUG_SHOW_CONTOURS,
        DEBUG_SHOW_FILTER_AREA,
        DEBUG_SHOW_FILTER_ASPECT,
        DEBUG_SHOW_FILTER_CIRCULAR,
        DEBUG_SHOW_FILTER_SOLIDITY,
        DEBUG_SHOW_FINAL_DETECTION,
    ])


def _filter_blobs_with_layers(contours) -> tuple[list, dict]:
    """Apply blob filters and keep pass-through contour sets per filter layer."""
    detections = []
    layers = {
        "contours": list(contours),
        "area": [],
        "aspect": [],
        "circularity": [],
        "solidity": [],
    }

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_BLOB_AREA or area > MAX_BLOB_AREA:
            continue
        layers["area"].append(cnt)

        x, y, w, h = cv2.boundingRect(cnt)
        long_side = max(w, h)
        short_side = min(w, h)
        aspect_ratio = long_side / short_side if short_side > 0 else 999.0
        if aspect_ratio > MAX_ASPECT_RATIO:
            continue
        layers["aspect"].append(cnt)

        perimeter = cv2.arcLength(cnt, True)
        if perimeter < 1e-6:
            continue
        circularity = float((4.0 * np.pi * area) / (perimeter ** 2))
        if circularity < MIN_CIRCULARITY:
            continue
        layers["circularity"].append(cnt)

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0
        if solidity < MIN_SOLIDITY:
            continue
        layers["solidity"].append(cnt)

        detections.append({
            "bbox": (x, y, w, h),
            "center": (x + w // 2, y + h // 2),
            "area": round(area, 2),
            "circularity": round(circularity, 3),
            "solidity": round(solidity, 3),
            "aspect_ratio": round(aspect_ratio, 2),
        })

    return detections, layers


def _select_primary_projectile(detections: list) -> list:
    """In no-tracking mode, keep only the best projectile candidate."""
    if not detections:
        return []

    target_area = (MIN_BLOB_AREA + MAX_BLOB_AREA) * 0.5

    def score(det: dict) -> float:
        circ = float(det.get("circularity", 0.0))
        solid = float(det.get("solidity", 0.0))
        aspect = float(det.get("aspect_ratio", 1.0))
        area = float(det.get("area", target_area))
        area_penalty = abs(area - target_area) / max(target_area, 1.0)
        aspect_penalty = abs(aspect - 1.0)
        return (1.8 * circ) + (1.2 * solid) - (0.8 * area_penalty) - (0.4 * aspect_penalty)

    best = max(detections, key=score)
    return [best]


def _scale_detections(detections: list, scale: float) -> list:
    """Return a copy of detections with bbox/center scaled by ``scale``."""
    scaled = []
    for det in detections:
        out = dict(det)
        cx, cy = det["center"]
        x, y, w, h = det["bbox"]
        out["center"] = (int(cx * scale), int(cy * scale))
        out["bbox"] = (int(x * scale), int(y * scale),
                       int(w * scale), int(h * scale))
        scaled.append(out)
    return scaled


def _draw_projectile_detections(frame: np.ndarray,
                                detections: list,
                                color: tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """Draw detections without IDs (for temporary no-tracking mode)."""
    out = frame.copy()
    for det in detections:
        x, y, w, h = det["bbox"]
        cx, cy = det["center"]
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.circle(out, (cx, cy), 3, color, -1)
    return out


def _draw_contour_layer(frame: np.ndarray,
                        contours: list,
                        title: str,
                        color: tuple[int, int, int]) -> np.ndarray:
    panel = frame.copy()
    if contours:
        cv2.drawContours(panel, contours, -1, color, 1)
    cv2.putText(panel, f"{title}: {len(contours)}",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return panel


def _show_filter_layers(frame: np.ndarray,
                        layer_contours: dict | None,
                        detections: list):
    """Show per-layer filtering outputs (contours + final detection)."""
    if layer_contours is None:
        return

    stage_configs = [
        ("Debug: contours", DEBUG_SHOW_CONTOURS, "contours", "Contours", (180, 180, 180)),
        ("Debug: area pass", DEBUG_SHOW_FILTER_AREA, "area", "Area pass", (0, 220, 255)),
        ("Debug: aspect pass", DEBUG_SHOW_FILTER_ASPECT, "aspect", "Aspect pass", (255, 120, 0)),
        ("Debug: circularity pass", DEBUG_SHOW_FILTER_CIRCULAR, "circularity", "Circularity pass", (255, 220, 0)),
        ("Debug: solidity pass", DEBUG_SHOW_FILTER_SOLIDITY, "solidity", "Solidity pass", (0, 255, 0)),
    ]

    for win_title, enabled, key, label, color in stage_configs:
        if not enabled:
            continue
        panel = _draw_contour_layer(frame, layer_contours.get(key, []), label, color)
        cv2.imshow(win_title, _resize_for_display(panel))

    if DEBUG_SHOW_FINAL_DETECTION:
        final_panel = _draw_projectile_detections(frame, detections, (0, 255, 0))
        cv2.putText(final_panel, f"Final projectile: {len(detections)}",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Debug: final projectile", _resize_for_display(final_panel))


def _draw_tracks(frame        : np.ndarray,
                 tracks       : list,
                 trail_store  : TrailStore,
                 validators   : dict,
                 classifier   : 'TrackClassifier | None',
                 show_trail   : bool) -> np.ndarray:
    """Draw bounding boxes, IDs, trails, and parabola arcs onto frame.

    When the classifier is active, only 'projectile' tracks are drawn
    prominently.  'pending' and 'noise' tracks are optionally shown
    based on CLASSIFIER_SHOW_PENDING / CLASSIFIER_SHOW_NOISE.
    """
    out = frame.copy()

    # Trails first so boxes are drawn on top
    if show_trail:
        # Only draw trails for tracks that will be visible
        if classifier is not None:
            visible_ids = set()
            for t in tracks:
                cls = classifier.get(t["id"])
                if cls == "projectile":
                    visible_ids.add(t["id"])
                elif cls == "pending" and CLASSIFIER_SHOW_PENDING:
                    visible_ids.add(t["id"])
                elif cls == "noise" and CLASSIFIER_SHOW_NOISE:
                    visible_ids.add(t["id"])
            # Draw only visible trails
            for tid in visible_ids:
                trail = trail_store.get(tid)
                if trail is None or len(trail) < 2:
                    continue
                cls = classifier.get(tid)
                
                if cls == "projectile":
                    base_color = COLOR_TRAIL
                elif cls == "noise":
                    base_color = (0, 0, 180)  # red for noise
                else:
                    base_color = tuple(int(c * 0.35) for c in TRACK_COLORS[tid % len(TRACK_COLORS)])
                pts = list(trail)
                n = len(pts)
                for i in range(1, n):
                    p1 = (pts[i-1][0], pts[i-1][1])
                    p2 = (pts[i][0], pts[i][1])
                    alpha = 0.3 + 0.7 * (i / (n - 1))
                    c = tuple(int(ch * alpha) for ch in base_color)
                    cv2.line(out, p1, p2, c, 2)
        else:
            trail_store.draw_all_trails(out, colors=TRACK_COLORS)

    for t in tracks:
        tid        = t["id"]
        cx, cy     = t["center"]
        x, y, w, h = t["bbox"]
        miss       = t["missing"]

        # Only draw tracks matched to a real detection this frame
        if miss > 0:
            continue

        # Determine classification
        cls = classifier.get(tid) if classifier is not None else "projectile"

        # Skip tracks based on classification
        if cls == "noise" and not CLASSIFIER_SHOW_NOISE:
            continue
        if cls == "pending" and not CLASSIFIER_SHOW_PENDING:
            continue

        # Colour coding by classification
        if cls == "projectile":
            color = TRACK_COLORS[tid % len(TRACK_COLORS)]
        elif cls == "pending":
            color = tuple(int(c * 0.35) for c in TRACK_COLORS[tid % len(TRACK_COLORS)])
        else:  # noise
            color = (0, 0, 180)  # red

        # Dim the box colour when predicting (no detection this frame)
        box_color = tuple(int(c * 0.45) for c in color) if miss > 0 else color
        cv2.rectangle(out, (x, y), (x + w, y + h), box_color, 1)

        # Build label
        val = validators.get(tid)
        label = f"ID:{tid}"
        if miss > 0:
            label += f" P{miss}"
            
        if t.get("suspect_innovation"):
            label += " G"
            
        if TRAJECTORY_ACTIVE and val is not None:
            if not val.is_ready():
                label += " ..."
            else:
                fails = []
                info = val.get_info()
                if info.get("residual") is not None and info["residual"] > val.max_residual:
                    fails.append("R")
                if hasattr(val, '_vel_validator') and not val._vel_validator.is_valid():
                    fails.append("V")
                if hasattr(val, '_check_shape_descriptors') and not val._check_shape_descriptors():
                    fails.append("S")
                
                if not fails and val.is_valid():
                    label += " OK"
                elif fails:
                    label += f" {''.join(fails)}"
                else:
                    label += " ?"
        # Classification indicator
        if cls == "noise":
            label += " X"
        elif cls == "pending":
            label += " ~"

        cv2.putText(out, label,
                    (x, max(y - 3, 10)),
                    FONT_FACE, FONT_SCALE, color, FONT_THICKNESS)
        cv2.circle(out, (cx, cy), 3, color, -1)

        # Fitted parabola arc (only for projectile tracks)
        if cls == "projectile" and TRAJECTORY_ACTIVE and val is not None and val.is_valid():
            arc = val.get_fitted_points()
            if arc and len(arc) > 1:
                h_img, w_img = out.shape[:2]
                for i in range(1, len(arc)):
                    p1, p2 = arc[i - 1], arc[i]
                    if (0 <= p1[0] < w_img and 0 <= p1[1] < h_img and
                            0 <= p2[0] < w_img and 0 <= p2[1] < h_img):
                        cv2.line(out, p1, p2, color, 1)

    return out


def _draw_hud(frame         : np.ndarray,
              frame_idx     : int,
              n_tracks      : int,
              n_dets        : int,
              n_projectiles : int,
              warmed_up     : bool) -> np.ndarray:
    """Minimal heads-up display overlay."""
    h, w = frame.shape[:2]

    # Warm-up notice along the bottom
    if not warmed_up and ENABLE_BG_SUB:
        cv2.rectangle(frame, (0, h - 18), (w, h), (20, 20, 20), -1)
        cv2.putText(frame, "MOG2 learning background — detections may be noisy",
                    (6, h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (80, 200, 80), 1)

    hud_text = f"frame={frame_idx}  tracks={n_tracks}  dets={n_dets}"
    if CLASSIFIER_ACTIVE or TEMP_DISABLE_TRACKING:
        hud_text += f"  projectiles={n_projectiles}"
    cv2.putText(frame, hud_text,
                (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 100), 1)

    # Key hint
    cv2.putText(frame,
                "SPACE=pause  S=screenshot  D=debug  Q=quit",
                (6, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)

    return frame


ALARM_HOLD_FRAMES = 30   # keep alarm visible for this many frames after last projectile

def _draw_alarm(frame: np.ndarray, active: bool, fade: float = 1.0) -> np.ndarray:
    """Draw a projectile-detected alarm overlay.

    Parameters
    ----------
    frame  : np.ndarray  BGR image.
    active : bool        True if a projectile is currently confirmed.
    fade   : float       1.0 = full intensity, fades toward 0 as alarm expires.
    """
    if not active and fade <= 0:
        return frame

    h, w = frame.shape[:2]
    alpha = max(0.0, min(1.0, fade))
    intensity = int(255 * alpha)

    # ── Green border flash ────────────────────────────────────────────
    border = 4
    color = (0, intensity, 0)
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, border)
    cv2.rectangle(frame, (border, border),
                  (w - 1 - border, h - 1 - border), color, border // 2)

    # ── Banner ────────────────────────────────────────────────────────
    banner_h = 36
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - banner_h - 24), (w, h - 24),
                  (0, 40, 0), -1)
    cv2.addWeighted(overlay, 0.6 * alpha, frame, 1 - 0.6 * alpha, 0, frame)

    text = "PROJECTILE DETECTED"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    (tw, th), _ = cv2.getTextSize(text, font, scale, 2)
    tx = (w - tw) // 2
    ty = h - 24 - (banner_h - th) // 2
    cv2.putText(frame, text, (tx, ty), font, scale,
                (0, intensity, 0), 2)

    return frame


# ── Projectile arc detector ──────────────────────────────────────────────────

def _check_projectile_arc(trail_store, tracks, frame_width):
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

    return flagged, arcs


# =============================================================================
# MAIN
# =============================================================================

def main():
    # ── Open video ────────────────────────────────────────────────────
    try:
        reader = VideoReader(VIDEO_PATH)
    except RuntimeError as e:
        print(f"\n  ERROR: {e}")
        print("  Check VIDEO_PATH in config.py\n")
        sys.exit(1)

    print(f"\n{'='*56}")
    print(f"  Ball Detector — starting")
    print(f"{'='*56}")
    print(f"  Video      : {VIDEO_PATH}")
    print(f"  Resolution : {reader.width} x {reader.height}")
    print(f"  FPS        : {reader.fps:.1f}")
    print(f"  Frames     : {reader.frame_count}")
    print(f"\n  Active stages:")
    print(f"    ROI            : {'ON' if ENABLE_ROI        else 'OFF'}")
    print(f"    CLAHE          : {'ON' if ENABLE_CLAHE      else 'OFF'}")
    print(f"    Top-Hat        : {'ON' if ENABLE_TOPHAT     else 'OFF'}  mode={TOPHAT_MODE}")
    print(f"    Frame diff     : {'ON' if ENABLE_FRAME_DIFF else 'OFF'}  gap={FRAME_DIFF_GAP}")
    print(f"    MOG2           : {'ON' if ENABLE_BG_SUB     else 'OFF'}  history={MOG2_HISTORY}")
    print(f"    Mask combine   : {MASK_COMBINE_MODE}")
    print(f"    Morph clean    : {'ON' if ENABLE_MORPH_CLEAN else 'OFF'}  kernel={MORPH_KERNEL_SIZE}")
    print(f"    Blob filter    : {'ON' if ENABLE_BLOB_FILTER else 'OFF'}")
    kalman_state = "OFF (temp disabled)" if TEMP_DISABLE_TRACKING and ENABLE_KALMAN else ("ON" if TRACKING_ACTIVE else "OFF")
    print(f"    Kalman tracker : {kalman_state}")
    print(f"    Trail          : {'ON' if TRAIL_ACTIVE else 'OFF'}  length={TRAIL_LENGTH}")
    print(f"    Trajectory     : {'ON' if TRAJECTORY_ACTIVE else 'OFF'}")
    print(f"    Classifier     : {'ON' if CLASSIFIER_ACTIVE else 'OFF'}")
    print(f"    Debug view     : {'ON' if ENABLE_DEBUG_VIEW else 'OFF'}")
    if TEMP_DISABLE_TRACKING:
        print(f"    Mode           : tracking bypass (no IDs on screen)")
    print(f"\n  Press Q or ESC to quit.")
    print(f"  Press SPACE to pause / resume.")
    print(f"  Press S to save a screenshot.")
    print(f"  Press D to toggle debug mask windows.\n")

    # ── Initialise pipeline stages ────────────────────────────────────
    differ      = FrameDiffer(FRAME_DIFF_GAP, FRAME_DIFF_THRESHOLD)

    bgs         = BackgroundSubtractor(
                      MOG2_HISTORY, MOG2_VAR_THRESHOLD, MOG2_DETECT_SHADOWS
                  )

    tracker     = KalmanTracker(KALMAN_MAX_DISTANCE, KALMAN_MAX_MISSING) \
                  if TRACKING_ACTIVE else None

    trail_store = TrailStore(TRAIL_LENGTH) \
                  if TRAIL_ACTIVE else TrailStore(0)

    classifier  = TrackClassifier(
                      min_age=CLASSIFIER_MIN_AGE,
                      min_displacement=CLASSIFIER_MIN_DISPLACEMENT,
                      min_speed=CLASSIFIER_MIN_SPEED,
                      max_speed=CLASSIFIER_MAX_SPEED,
                      min_path_efficiency=CLASSIFIER_MIN_PATH_EFFICIENCY,
                      min_spatial_spread=CLASSIFIER_MIN_SPATIAL_SPREAD,
                      min_direction_ratio=CLASSIFIER_MIN_DIRECTION_RATIO,
                  ) if CLASSIFIER_ACTIVE else None

    # Trajectory validators — one created per track ID on demand
    validators: dict = {}
    saved_projectile_arcs = {}  # permanently saves arcs of detected projectiles

    paused       = False
    frame_idx    = 0
    screenshot_n = 0
    debug_on     = ENABLE_DEBUG_VIEW
    alarm_counter = 0

    # ── Main loop ─────────────────────────────────────────────────────
    for frame in reader.frames():
        frame_idx += 1

        # ── Frame skip ────────────────────────────────────────────────
        if FRAME_SKIP > 1 and (frame_idx % FRAME_SKIP) != 0:
            continue

        # ── Stage 1: ROI ──────────────────────────────────────────────
        if ENABLE_ROI:
            roi_frame, offset = apply_roi(
                frame, ROI_PERCENT, ROI_ANCHOR_X, ROI_ANCHOR_Y
            )
        else:
            roi_frame, offset = frame, (0, 0)

        # ── Stage 1.5: Downscale for processing ──────────────────────
        display_frame = roi_frame                     # keep full-res for drawing
        if PROCESS_SCALE < 1.0:
            roi_frame = cv2.resize(roi_frame, None,
                                   fx=PROCESS_SCALE, fy=PROCESS_SCALE,
                                   interpolation=cv2.INTER_AREA)

        # ── Stage 2: CLAHE grayscale ──────────────────────────────────
        gray = to_gray(roi_frame, CLAHE_CLIP_LIMIT,
                        CLAHE_TILE_GRID, ENABLE_CLAHE)

        # ── Stage 3: Top-Hat spatial filter ───────────────────────────
        th_mask = apply_tophat(
            gray, TOPHAT_KERNEL_SIZE, TOPHAT_THRESHOLD, TOPHAT_MODE
        ) if ENABLE_TOPHAT else None

        # ── Stage 4: 3-frame temporal difference ──────────────────────
        df_mask = differ.update(gray) if ENABLE_FRAME_DIFF else None

        # ── Stage 5: MOG2 background subtraction ──────────────────────
        mg_mask = bgs.apply(roi_frame) if ENABLE_BG_SUB else None

        # ── Stage 6: Combine masks ────────────────────────────────────
        if th_mask is None and df_mask is None and mg_mask is None: 
            # All motion stages disabled — can't continue
            continue

        combined = combine_masks(th_mask, df_mask, mg_mask, MASK_COMBINE_MODE)

        # ── Stage 7: Morphological cleaning ───────────────────────────
        cleaned = clean_mask(combined, MORPH_KERNEL_SIZE) \
                  if ENABLE_MORPH_CLEAN else combined

        # ── Stage 8: Contour detection ────────────────────────────────
        contours = detect_contours(cleaned)

        # ── Stage 9: Blob filtering ───────────────────────────────────
        layer_contours = None
        if ENABLE_BLOB_FILTER:
            if debug_on and _wants_filter_layer_debug():
                detections, layer_contours = _filter_blobs_with_layers(contours)
            else:
                detections = filter_blobs(
                    contours,
                    MIN_BLOB_AREA, MAX_BLOB_AREA,
                    MIN_CIRCULARITY, MIN_SOLIDITY, MAX_ASPECT_RATIO,
                )
        else:
            # Skip shape filtering — pass all contours as raw detections
            detections = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append({
                    "bbox"  : (x, y, w, h),
                    "center": (x + w // 2, y + h // 2),
                    "area"  : cv2.contourArea(cnt),
                })
            if debug_on and _wants_filter_layer_debug():
                layer_contours = {
                    "contours": contours,
                    "area": contours,
                    "aspect": contours,
                    "circularity": contours,
                    "solidity": contours,
                }

        # ── Stage 9b: Isolation filter ─────────────────────────────────
        if ENABLE_ISOLATION_FILTER:
            detections = reject_clustered_blobs(
                detections,
                radius           = ISOLATION_RADIUS,
                min_cluster_size = ISOLATION_MIN_CLUSTER,
            )

        # ── Stage 10: Kalman tracking ─────────────────────────────────
        if tracker is not None:
            tracks = tracker.update(detections)
        else:
            tracks = []

        draw_detections = detections

        # ── Stage 10.5: Scale coordinates back to full-res ─────────────
        if PROCESS_SCALE < 1.0:
            inv = 1.0 / PROCESS_SCALE
            for t in tracks:
                cx, cy = t["center"]
                t["center"] = (int(cx * inv), int(cy * inv))
                x, y, w, h = t["bbox"]
                t["bbox"] = (int(x * inv), int(y * inv),
                             int(w * inv), int(h * inv))
            draw_detections = _scale_detections(detections, inv)

        # ── Stage 11: Trail storage ───────────────────────────────────
        if TRAIL_ACTIVE:
            trail_store.update(tracks)

        # ── Stage 12: Trajectory validation ──────────────────────────
        if TRAJECTORY_ACTIVE:
            active_ids = {t["id"] for t in tracks}

            # Remove validators for tracks that have been deleted
            for tid in list(validators.keys()):
                if tid not in active_ids:
                    del validators[tid]

            # Feed observed detections into each track's validator
            for t in tracks:
                tid = t["id"]
                if tid not in validators:
                    validators[tid] = TrajectoryValidator(
                        TRAJECTORY_MIN_POINTS, TRAJECTORY_MAX_RESIDUAL
                    )
                # Only update with observed (non-predicted) positions
                if t["missing"] == 0:
                    validators[tid].update(*t["center"])
        else:
            validators = {}

        # ── Stage 12.5: Track classification ──────────────────────────
        if classifier is not None:
            classifier.update(tracks, trail_store, validators)

        # Count projectiles for HUD
        projectile_detections = _select_primary_projectile(draw_detections) \
            if TEMP_DISABLE_TRACKING else draw_detections

        n_projectiles = len(projectile_detections) if TEMP_DISABLE_TRACKING else 0
        if classifier is not None:
            n_projectiles = sum(
                1 for t in tracks if classifier.get(t["id"]) == "projectile"
            )

        # ── Projectile arc check ──────────────────────────────────────
        arc_ids = set()
        if TRAIL_ACTIVE and TRACKING_ACTIVE:
            arc_ids, new_arcs = _check_projectile_arc(
                trail_store, tracks, display_frame.shape[1]
            )
            if arc_ids:
                n_projectiles = len(arc_ids)
                # Permanently save these winning arcs
                for tid, pts in new_arcs.items():
                    saved_projectile_arcs[tid] = pts

        # ── Alarm logic ───────────────────────────────────────────────
        if n_projectiles > 0:
            if alarm_counter == 0:
                print(f"  [!] PROJECTILE DETECTED at frame {frame_idx}!")
            alarm_counter = ALARM_HOLD_FRAMES
        else:
            alarm_counter = max(0, alarm_counter - 1)

        # ── Stage 13: Draw ────────────────────────────────────────────
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
            color = (0, 0, 0) # Black color for arc
            h_img, w_img = output.shape[:2]
            # Draw every 3rd point as a filled circle to create a bold dotted line effect
            for i in range(0, len(arc_pts), 3):
                p = arc_pts[i]
                if 0 <= p[0] < w_img and 0 <= p[1] < h_img:
                    cv2.circle(output, p, 10, color, -1) # Radius 5 = very bold dots

        output = _draw_hud(
            output, frame_idx,
            hud_tracks, len(draw_detections),
            n_projectiles,
            bgs.is_warmed_up
        )

        # ── Stage 14: Debug windows ───────────────────────────────────
        if debug_on:
            _show_debug_masks({
                "tophat"    : th_mask,
                "frame_diff": df_mask,
                "bg_sub"    : mg_mask,
                "combined"  : combined,
                "cleaned"   : cleaned,
            })
            debug_final = _select_primary_projectile(detections) \
                if TEMP_DISABLE_TRACKING else detections
            _show_filter_layers(roi_frame, layer_contours, debug_final)
            
        # ── Stage 15: Alarm overlay ───────────────────────────────────
        if alarm_counter > 0:
            fade = alarm_counter / ALARM_HOLD_FRAMES
            output = _draw_alarm(output, n_projectiles > 0, fade)

        # ── Stage 16: Display ─────────────────────────────────────────
        display = _resize_for_display(output)
        cv2.imshow("Ball Detector  —  Q to quit", display)

        # ── Key handling ──────────────────────────────────────────────
        wait = 0 if paused else 1
        key  = cv2.waitKey(wait) & 0xFF

        if key in (27, ord("q")):
            print(f"  Quit at frame {frame_idx}.")
            break

        elif key == ord(" "):
            paused = not paused
            print(f"  {'Paused' if paused else 'Resumed'} at frame {frame_idx}.")

        elif key == ord("s"):
            screenshot_n += 1
            fname = f"screenshot_{screenshot_n:04d}.png"
            cv2.imwrite(fname, output)
            print(f"  Screenshot saved: {fname}")

        elif key == ord("d"):
            debug_on = not debug_on
            if not debug_on:
                for title in DEBUG_WINDOWS:
                    try:
                        cv2.destroyWindow(title)
                    except Exception:
                        pass
            print(f"  Debug view {'ON' if debug_on else 'OFF'}.")

        # When paused, hold until space or q
        if paused:
            while True:
                k2 = cv2.waitKey(0) & 0xFF
                if k2 == ord(" "):
                    paused = False
                    break
                elif k2 in (27, ord("q")):
                    cv2.destroyAllWindows()
                    reader.release()
                    print(f"  Quit at frame {frame_idx}.")
                    sys.exit(0)

    # ── Cleanup ───────────────────────────────────────────────────────
    cv2.destroyAllWindows()
    reader.release()
    print(f"\n  Finished.  Processed {frame_idx} frames.\n")


if __name__ == "__main__":
    main()

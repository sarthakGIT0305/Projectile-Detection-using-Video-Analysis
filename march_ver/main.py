# =============================================================================
# main.py
# =============================================================================
# Orchestrates the full detection-tracking pipeline.
#
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
    ENABLE_TRAJECTORY, TRAJECTORY_MIN_POINTS, TRAJECTORY_MAX_RESIDUAL,
    ENABLE_KALMAN, KALMAN_MAX_DISTANCE, KALMAN_MAX_MISSING,
    ENABLE_TRAIL, TRAIL_LENGTH,
    ENABLE_DEBUG_VIEW,
    DEBUG_SHOW_TOPHAT, DEBUG_SHOW_FRAME_DIFF,
    DEBUG_SHOW_BG_SUB, DEBUG_SHOW_COMBINED, DEBUG_SHOW_CLEANED,
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
from detection.trajectory_fit    import TrajectoryValidator
from tracking.kalman_tracker     import KalmanTracker
from tracking.trail_store        import TrailStore


# Colour palette that cycles by track ID
TRACK_COLORS = [
    (0,255,0), (0,200,255), (255,100,0), (200,0,255),
    (0,255,200), (255,255,0), (100,100,255), (255,0,100),
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


def _draw_tracks(frame       : np.ndarray,
                 tracks      : list,
                 trail_store : TrailStore,
                 validators  : dict,
                 show_trail  : bool) -> np.ndarray:
    """Draw bounding boxes, IDs, trails, and parabola arcs onto frame."""
    out = frame.copy()

    # Trails first so boxes are drawn on top
    if show_trail:
        trail_store.draw_all_trails(out, colors=TRACK_COLORS)

    for t in tracks:
        tid        = t["id"]
        cx, cy     = t["center"]
        x, y, w, h = t["bbox"]
        color      = TRACK_COLORS[tid % len(TRACK_COLORS)]
        miss       = t["missing"]

        # Dim the box colour when predicting (no detection this frame)
        box_color = tuple(int(c * 0.45) for c in color) if miss > 0 else color
        cv2.rectangle(out, (x, y), (x + w, y + h), box_color, 1)

        # Build label
        val = validators.get(tid)
        label = f"ID:{tid}"
        if miss > 0:
            label += f" P{miss}"
        if ENABLE_TRAJECTORY and val is not None:
            if not val.is_ready():
                label += " ..."
            elif val.is_valid():
                label += " OK"
            else:
                label += " ?"

        cv2.putText(out, label,
                    (x, max(y - 3, 10)),
                    FONT_FACE, FONT_SCALE, color, FONT_THICKNESS)
        cv2.circle(out, (cx, cy), 3, color, -1)

        # Fitted parabola arc (only when trajectory validation is on + valid)
        if ENABLE_TRAJECTORY and val is not None and val.is_valid():
            arc = val.get_fitted_points()
            if arc and len(arc) > 1:
                h_img, w_img = out.shape[:2]
                for i in range(1, len(arc)):
                    p1, p2 = arc[i - 1], arc[i]
                    if (0 <= p1[0] < w_img and 0 <= p1[1] < h_img and
                            0 <= p2[0] < w_img and 0 <= p2[1] < h_img):
                        cv2.line(out, p1, p2, color, 1)

    return out


def _draw_hud(frame     : np.ndarray,
              frame_idx : int,
              n_tracks  : int,
              n_dets    : int,
              warmed_up : bool) -> np.ndarray:
    """Minimal heads-up display overlay."""
    h, w = frame.shape[:2]

    # Warm-up notice along the bottom
    if not warmed_up and ENABLE_BG_SUB:
        cv2.rectangle(frame, (0, h - 18), (w, h), (20, 20, 20), -1)
        cv2.putText(frame, "MOG2 learning background — detections may be noisy",
                    (6, h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (80, 200, 80), 1)

    cv2.putText(frame,
                f"frame={frame_idx}  tracks={n_tracks}  dets={n_dets}",
                (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 100), 1)

    # Key hint
    cv2.putText(frame,
                "SPACE=pause  S=screenshot  D=debug  Q=quit",
                (6, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)

    return frame


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
    print(f"    Kalman tracker : {'ON' if ENABLE_KALMAN     else 'OFF'}")
    print(f"    Trail          : {'ON' if ENABLE_TRAIL      else 'OFF'}  length={TRAIL_LENGTH}")
    print(f"    Trajectory     : {'ON' if ENABLE_TRAJECTORY else 'OFF'}")
    print(f"    Debug view     : {'ON' if ENABLE_DEBUG_VIEW else 'OFF'}")
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
                  if ENABLE_KALMAN else None

    trail_store = TrailStore(TRAIL_LENGTH) \
                  if ENABLE_TRAIL else TrailStore(0)

    # Trajectory validators — one created per track ID on demand
    validators: dict = {}

    paused       = False
    frame_idx    = 0
    screenshot_n = 0
    debug_on     = ENABLE_DEBUG_VIEW

    # ── Main loop ─────────────────────────────────────────────────────
    for frame in reader.frames():
        frame_idx += 1

        # ── Stage 1: ROI ──────────────────────────────────────────────
        if ENABLE_ROI:
            roi_frame, offset = apply_roi(
                frame, ROI_PERCENT, ROI_ANCHOR_X, ROI_ANCHOR_Y
            )
        else:
            roi_frame, offset = frame, (0, 0)

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
        if ENABLE_BLOB_FILTER:
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

        # ── Stage 10: Kalman tracking ─────────────────────────────────
        if tracker is not None:
            tracks = tracker.update(detections)
        else:
            tracks = [
                {"id": i, "center": d["center"], "bbox": d["bbox"],
                 "velocity": (0, 0), "missing": 0, "age": 1, "det": d}
                for i, d in enumerate(detections)
            ]

        # ── Stage 11: Trail storage ───────────────────────────────────
        trail_store.update(tracks)

        # ── Stage 12: Trajectory validation ──────────────────────────
        if ENABLE_TRAJECTORY:
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

        # ── Stage 13: Draw ────────────────────────────────────────────
        output = _draw_tracks(
            roi_frame, tracks, trail_store, validators,
            show_trail=ENABLE_TRAIL
        )
        output = _draw_hud(
            output, frame_idx,
            len(tracks), len(detections),
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

        # ── Stage 15: Display ─────────────────────────────────────────
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
                for title in ["Debug: top-hat", "Debug: frame diff",
                               "Debug: MOG2", "Debug: combined",
                               "Debug: cleaned"]:
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
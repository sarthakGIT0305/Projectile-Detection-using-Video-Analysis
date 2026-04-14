# =============================================================================
# config.py
# =============================================================================
# ONE place for every tunable value and every feature flag in the pipeline.
# To turn a stage on/off: flip its ENABLE_ flag to True / False.
# To tune a parameter:    change the value here — no other file needs editing.
# =============================================================================
import cv2 
# -----------------------------------------------------------------------------
# VIDEO
# -----------------------------------------------------------------------------
VIDEO_PATH = r"D:\\Coding++\\web_dev_and_projects\\folderAssets\\open_cv_assets\\20260408_05.mp4" 

# -----------------------------------------------------------------------------
# FEATURE FLAGS
# Each flag independently enables / disables one pipeline stage.
# Start with all True, then turn flags off one by one to understand
# each stage's individual contribution to the final result.
# -----------------------------------------------------------------------------
ENABLE_ROI          = False   # crop frame to a sub-region before processing
ENABLE_CLAHE        = True    # CLAHE contrast boost before motion detection
ENABLE_TOPHAT       = True    # top-hat spatial filter  (key for tiny objects)
ENABLE_FRAME_DIFF   = True    # 3-frame temporal difference
ENABLE_BG_SUB       = True    # MOG2 background subtraction
ENABLE_MORPH_CLEAN  = True    # morphological denoising of combined mask
ENABLE_BLOB_FILTER  = True    # area / circularity / solidity blob filter
ENABLE_TRAJECTORY   = False   # DISABLED — focus on raw detection first
ENABLE_KALMAN       = 1 # Kalman tracking with IDs
ENABLE_TRAIL        = 1   # trajectory trails
ENABLE_CLASSIFIER   = 0   # classification off for now
ENABLE_DEBUG_VIEW   = True    # show intermediate mask windows while running
TEMP_DISABLE_TRACKING = False  # not needed — use proper flags above

# All advanced tracking disabled for detection-only mode
ENABLE_VELOCITY_CHECK    = False
ENABLE_SHAPE_DESCRIPTORS = False
KALMAN_USE_GRAVITY       = False

# -----------------------------------------------------------------------------
# PERFORMANCE
# -----------------------------------------------------------------------------
# Full resolution is critical — the ball is extremely tiny (2–4 px).
# Downscaling would erase it entirely.
PROCESS_SCALE = 0.75

# Process every frame so we don't miss the ball between skips.
FRAME_SKIP = 2

# -----------------------------------------------------------------------------
# ROI  (preprocessing/roi.py)
# -----------------------------------------------------------------------------
ROI_PERCENT  = 1
ROI_ANCHOR_X = 0.5
ROI_ANCHOR_Y = 0.5

# -----------------------------------------------------------------------------
# CLAHE  (preprocessing/clahe_gray.py)
# -----------------------------------------------------------------------------
# Higher clip limit helps pull the dark ball out against bright sky.
# Larger tile grid (8×8) gives more localised enhancement across the
# sky-to-ground gradient, boosting the ball wherever it appears.
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID  = (8, 8)

# -----------------------------------------------------------------------------
# TOP-HAT FILTER  (motion/tophat.py)
# -----------------------------------------------------------------------------
# The ball is ~2–5 px at full res.  An 11×11 kernel is good — it
# suppresses anything bigger than 11 px (clouds, poles, buildings)
# while preserving the tiny ball.
# Lower threshold (12) so we don't lose the faint ball against bright sky.
TOPHAT_MODE        = "both"
TOPHAT_KERNEL_SIZE = (11, 11)
TOPHAT_THRESHOLD   = 12

# -----------------------------------------------------------------------------
# 3-FRAME DIFFERENCE  (motion/frame_diff.py)
# -----------------------------------------------------------------------------
# Gap = 2 skips slow cloud drift.  Gap = 3 if clouds are fast.
# Threshold raised to 25 — cloud edges create subtle brightness shifts
# that must be suppressed while keeping the sharp ball movement.
FRAME_DIFF_GAP       = 2
FRAME_DIFF_THRESHOLD = 25

# -----------------------------------------------------------------------------
# MOG2 BACKGROUND SUBTRACTION  (motion/bg_sub.py)
# -----------------------------------------------------------------------------
# High history (500) so MOG2 learns the slowly drifting clouds as background.
# High varThreshold (40) — clouds subtly shift brightness frame-to-frame;
# the ball moves much faster and creates a stronger pixel change.
MOG2_HISTORY        = 500
MOG2_VAR_THRESHOLD  = 40
MOG2_DETECT_SHADOWS = False

# -----------------------------------------------------------------------------
# MASK COMBINATION  (motion/mask_combine.py)
# -----------------------------------------------------------------------------
# "tophat_and_any" — requires the top-hat spatial filter to confirm
# the object is small AND at least one temporal motion mask agrees.
# This kills large cloud regions (which pass motion but fail top-hat)
# and random single-pixel noise (which passes top-hat but fails motion).
MASK_COMBINE_MODE = "tophat_and_any"

# -----------------------------------------------------------------------------
# MORPHOLOGICAL CLEANING  (motion/morph_clean.py)
# -----------------------------------------------------------------------------
# (2,2) removes isolated single-pixel salt noise without erasing the
# tiny 2–5 px ball.  (1,1) = no cleaning; (3,3) = may erase ball.
MORPH_KERNEL_SIZE = (2, 2)

# -----------------------------------------------------------------------------
# BLOB FILTER  (detection/blob_filter.py)
# -----------------------------------------------------------------------------
# Extremely tiny ball: as small as 2 px² at full res.
# Raise MIN_BLOB_AREA only if you see too many single-pixel noise dots.
MIN_BLOB_AREA    = 2
MAX_BLOB_AREA    = 80

# High circularity — the ball is round; clouds, poles, tree edges are not.
MIN_CIRCULARITY  = 0.6

# Solidity — ball is convex; cloud edge fragments are concave.
MIN_SOLIDITY     = 0.6

# Tight aspect ratio — the ball is ~1:1.
MAX_ASPECT_RATIO = 3.0

# -----------------------------------------------------------------------------
# ISOLATION FILTER  (detection/isolation_filter.py)
# -----------------------------------------------------------------------------
# Rejects blobs that are clustered together — noise sources like leaves
# and grass produce many nearby blobs, while the ball is always alone.
ENABLE_ISOLATION_FILTER  = True

# Pixel radius within which two blobs are considered neighbours.
# Set this to roughly 3–5x the expected ball diameter in pixels.
# At 50m the ball is ~6–8px wide, so start at 30–40px.
# Raise if clustered noise still passes. Lower if the ball itself
# is being rejected (it should never have neighbours this close).
ISOLATION_RADIUS         = 150

# Minimum number of blobs in a neighbourhood to trigger rejection.
# 2 = reject any blob that has even one neighbour (very strict).
# 3 = reject only when 3 or more blobs are near each other (recommended).
# The ball will never have neighbours, so 2 is safe but start at 3.
ISOLATION_MIN_CLUSTER    = 5

# -----------------------------------------------------------------------------
# TRAJECTORY VALIDATION  (detection/trajectory_fit.py)
# -----------------------------------------------------------------------------
TRAJECTORY_MIN_POINTS   = 5
TRAJECTORY_MAX_RESIDUAL = 50.0

# Velocity profile analysis thresholds (inactive when ENABLE_VELOCITY_CHECK=False)
VELOCITY_DX_MAX_VARIANCE     = 100.0
VELOCITY_DY_MIN_R2           = 0.1
VELOCITY_MAX_DIRECTION_FLIPS = 1

# Trajectory shape descriptor thresholds (inactive when ENABLE_SHAPE_DESCRIPTORS=False)
TRAJECTORY_MAX_ARC_RATIO    = 2.5
TRAJECTORY_MAX_APEX_COUNT   = 5
TRAJECTORY_MAX_SPEED_JITTER = 90.0

# -----------------------------------------------------------------------------
# KALMAN TRACKER  (tracking/kalman_tracker.py)
# -----------------------------------------------------------------------------
KALMAN_MAX_DISTANCE = 80       # wider gate — ball moves fast across open sky
KALMAN_MAX_MISSING  = 6        # keep track alive longer through brief occlusions

# Physics-constrained Kalman settings (inactive when KALMAN_USE_GRAVITY=False)
KALMAN_GRAVITY_PIXELS_PER_FRAME2 = 0.5
KALMAN_MAX_INNOVATION = 30.0

# -----------------------------------------------------------------------------
# TRACK CLASSIFIER  (detection/track_classifier.py)
# -----------------------------------------------------------------------------
# All values kept for reference but classifier is DISABLED above.
CLASSIFIER_MIN_AGE             = 4
CLASSIFIER_MIN_DISPLACEMENT    = 20.0
CLASSIFIER_MIN_SPEED           = 5.0
CLASSIFIER_MAX_SPEED           = 100.0
CLASSIFIER_MIN_PATH_EFFICIENCY = 0.3
CLASSIFIER_MIN_SPATIAL_SPREAD  = 35.0
CLASSIFIER_MIN_DIRECTION_RATIO = 0.7
CLASSIFIER_SHOW_PENDING        = True
CLASSIFIER_SHOW_NOISE          = True

# -----------------------------------------------------------------------------
# TRAIL  (tracking/trail_store.py)
# -----------------------------------------------------------------------------
TRAIL_LENGTH = 60   # longer trail to see the full arc across the sky

# -----------------------------------------------------------------------------
# PROJECTILE ARC DETECTION
# -----------------------------------------------------------------------------
# Checks each track's trail for a big downward parabola (thrown ball).
# A flag fires when a track's observed trail:
#   1. Spans at least ARC_MIN_SPAN_RATIO of the frame width horizontally
#   2. Fits a downward-opening parabola (positive 'a' in image coords)
#   3. Has a parabola fit residual below ARC_MAX_RESIDUAL
#   4. Has at least ARC_MIN_POINTS observed positions
ARC_MIN_SPAN_RATIO = 0.3     # 30% of frame width minimum horizontal coverage
ARC_MAX_RESIDUAL   = 15.0     # max mean pixel error for the parabola fit
ARC_MIN_POINTS     = 8        # minimum observed trail points before checking

# -----------------------------------------------------------------------------
# DEBUG DISPLAY
# -----------------------------------------------------------------------------
# Enable the key masks to see what's happening at each stage.
DEBUG_SHOW_TOPHAT      = True
DEBUG_SHOW_FRAME_DIFF  = False
DEBUG_SHOW_BG_SUB      = False
DEBUG_SHOW_COMBINED    = True
DEBUG_SHOW_CLEANED     = True
DEBUG_SHOW_CONTOURS        = False
DEBUG_SHOW_FILTER_AREA     = False
DEBUG_SHOW_FILTER_ASPECT   = False
DEBUG_SHOW_FILTER_CIRCULAR = False
DEBUG_SHOW_FILTER_SOLIDITY = False
DEBUG_SHOW_FINAL_DETECTION = False

DISPLAY_MAX_W = 1280
DISPLAY_MAX_H = 720

# Drawing colours (BGR)
COLOR_BBOX   = (0, 255, 0)    # green   bounding box
COLOR_ID     = (0, 0, 255)    # red     object ID label
COLOR_CENTER = (255, 0, 0)    # blue    centroid dot
COLOR_TRAIL  = (0, 255, 255)  # yellow  trajectory trail

FONT_FACE      = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE     = 0.5
FONT_THICKNESS = 1


# =============================================================================
# SELF-TEST
# Run:  python config.py
# Prints every value so you can confirm the file loaded correctly.
# =============================================================================
if __name__ == "__main__":
    import sys

    flags = {
        "ENABLE_ROI"         : ENABLE_ROI,
        "ENABLE_CLAHE"       : ENABLE_CLAHE,
        "ENABLE_TOPHAT"      : ENABLE_TOPHAT,
        "ENABLE_FRAME_DIFF"  : ENABLE_FRAME_DIFF,
        "ENABLE_BG_SUB"      : ENABLE_BG_SUB,
        "ENABLE_MORPH_CLEAN" : ENABLE_MORPH_CLEAN,
        "ENABLE_BLOB_FILTER" : ENABLE_BLOB_FILTER,
        "ENABLE_TRAJECTORY"  : ENABLE_TRAJECTORY,
        "ENABLE_KALMAN"      : ENABLE_KALMAN,
        "ENABLE_TRAIL"       : ENABLE_TRAIL,
        "ENABLE_DEBUG_VIEW"  : ENABLE_DEBUG_VIEW,
        "TEMP_DISABLE_TRACKING": TEMP_DISABLE_TRACKING,
        "ENABLE_VELOCITY_CHECK": ENABLE_VELOCITY_CHECK,
        "ENABLE_SHAPE_DESCRIPTORS": ENABLE_SHAPE_DESCRIPTORS,
        "KALMAN_USE_GRAVITY" : KALMAN_USE_GRAVITY,
    }

    params = {
        "VIDEO_PATH"           : VIDEO_PATH,
        "ROI_PERCENT"          : ROI_PERCENT,
        "CLAHE_CLIP_LIMIT"     : CLAHE_CLIP_LIMIT,
        "CLAHE_TILE_GRID"      : CLAHE_TILE_GRID,
        "TOPHAT_KERNEL_SIZE"   : TOPHAT_KERNEL_SIZE,
        "TOPHAT_THRESHOLD"     : TOPHAT_THRESHOLD,
        "FRAME_DIFF_GAP"       : FRAME_DIFF_GAP,
        "FRAME_DIFF_THRESHOLD" : FRAME_DIFF_THRESHOLD,
        "MOG2_HISTORY"         : MOG2_HISTORY,
        "MOG2_VAR_THRESHOLD"   : MOG2_VAR_THRESHOLD,
        "MOG2_DETECT_SHADOWS"  : MOG2_DETECT_SHADOWS,
        "MASK_COMBINE_MODE"    : MASK_COMBINE_MODE,
        "MORPH_KERNEL_SIZE"    : MORPH_KERNEL_SIZE,
        "MIN_BLOB_AREA"        : MIN_BLOB_AREA,
        "MAX_BLOB_AREA"        : MAX_BLOB_AREA,
        "MIN_CIRCULARITY"      : MIN_CIRCULARITY,
        "MIN_SOLIDITY"         : MIN_SOLIDITY,
        "MAX_ASPECT_RATIO"     : MAX_ASPECT_RATIO,
        "TRAJECTORY_MIN_POINTS": TRAJECTORY_MIN_POINTS,
        "TRAJECTORY_MAX_RESIDUAL": TRAJECTORY_MAX_RESIDUAL,
        "KALMAN_MAX_DISTANCE"  : KALMAN_MAX_DISTANCE,
        "KALMAN_MAX_MISSING"   : KALMAN_MAX_MISSING,
        "KALMAN_GRAVITY_PIXELS_PER_FRAME2": KALMAN_GRAVITY_PIXELS_PER_FRAME2,
        "KALMAN_MAX_INNOVATION": KALMAN_MAX_INNOVATION,
        "VELOCITY_DX_MAX_VARIANCE": VELOCITY_DX_MAX_VARIANCE,
        "VELOCITY_DY_MIN_R2": VELOCITY_DY_MIN_R2,
        "VELOCITY_MAX_DIRECTION_FLIPS": VELOCITY_MAX_DIRECTION_FLIPS,
        "TRAJECTORY_MAX_ARC_RATIO": TRAJECTORY_MAX_ARC_RATIO,
        "TRAJECTORY_MAX_APEX_COUNT": TRAJECTORY_MAX_APEX_COUNT,
        "TRAJECTORY_MAX_SPEED_JITTER": TRAJECTORY_MAX_SPEED_JITTER,
        "TRAIL_LENGTH"         : TRAIL_LENGTH,
    }

    print("\n" + "="*52)
    print("  config.py — self-test")
    print("="*52)

    print("\n[ Feature Flags ]")
    all_ok = True
    for name, val in flags.items():
        state = "ON " if val else "OFF"
        print(f"  {state}  {name}")

    print("\n[ Parameters ]")
    for name, val in params.items():
        print(f"  {name:<30} = {val}")

    # Basic sanity checks
    print("\n[ Sanity Checks ]")
    checks = [
        (MIN_BLOB_AREA > 0,                    "MIN_BLOB_AREA must be > 0"),
        (MIN_BLOB_AREA < MAX_BLOB_AREA,        "MIN_BLOB_AREA must be < MAX_BLOB_AREA"),
        (0.0 < ROI_PERCENT <= 1.0,             "ROI_PERCENT must be in (0, 1]"),
        (MORPH_KERNEL_SIZE[0] <= 3,            "MORPH_KERNEL_SIZE should be <= (3,3) for tiny objects"),
        (TOPHAT_KERNEL_SIZE[0] > MORPH_KERNEL_SIZE[0], "TOPHAT kernel should be larger than MORPH kernel"),
        (FRAME_DIFF_GAP >= 1,                  "FRAME_DIFF_GAP must be >= 1"),
        (MASK_COMBINE_MODE in ("or","and","tophat_and_any","motion_primary"), "MASK_COMBINE_MODE must be or/and/tophat_and_any/motion_primary"),
    ]
    all_ok = True
    for passed, msg in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {msg}")
        if not passed:
            all_ok = False

    print()
    if all_ok:
        print("  All checks passed. config.py is ready.\n")
        sys.exit(0)
    else:
        print("  One or more checks FAILED. Fix config.py before proceeding.\n")
        sys.exit(1)

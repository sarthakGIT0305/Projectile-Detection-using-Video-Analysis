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
VIDEO_PATH = r"D:\\Coding++\\web_dev_and_projects\\folderAssets\\open_cv_assets\\distance_3.mp4" 

# -----------------------------------------------------------------------------
# FEATURE FLAGS
# Each flag independently enables / disables one pipeline stage.
# Start with all True, then turn flags off one by one to understand
# each stage's individual contribution to the final result.
# -----------------------------------------------------------------------------
ENABLE_ROI          = True   # crop frame to a sub-region before processing
ENABLE_CLAHE        = True   # CLAHE contrast boost before motion detection
ENABLE_TOPHAT       = True   # top-hat spatial filter  (key for tiny objects)
ENABLE_FRAME_DIFF   = True   # 3-frame temporal difference
ENABLE_BG_SUB       = True   # MOG2 background subtraction
ENABLE_MORPH_CLEAN  = True   # morphological denoising of combined mask
ENABLE_BLOB_FILTER  = True   # area / circularity / solidity blob filter
ENABLE_TRAJECTORY   = False  # parabolic trajectory validation (Phase 4)
ENABLE_KALMAN       = True   # Kalman-filter tracker
ENABLE_TRAIL        = True   # draw rolling trajectory trail behind object
ENABLE_DEBUG_VIEW   = True   # show intermediate mask windows while running

# -----------------------------------------------------------------------------
# ROI  (preprocessing/roi.py)
# -----------------------------------------------------------------------------
# Fraction of the frame to keep, centered.
# 1.0 = full frame.  0.6 = keep central 60% of width and height.
# Only reduce this when you know the ball stays in a specific zone.
ROI_PERCENT = 1
ROI_ANCHOR_X = 0.5
ROI_ANCHOR_Y = 0.5

# -----------------------------------------------------------------------------
# CLAHE  (preprocessing/clahe_gray.py)
# -----------------------------------------------------------------------------
# clipLimit    — higher = stronger local contrast boost.  Range: 1.0 – 5.0
# tileGridSize — smaller tiles = more localised enhancement
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID  = (8, 8)

# -----------------------------------------------------------------------------
# TOP-HAT FILTER  (motion/tophat.py)
# -----------------------------------------------------------------------------
# The structuring element (kernel) must be LARGER than the ball but
# SMALLER than background structures (railings, clothes, walls).
#
# At 50 m a tennis ball is ~4–8 px wide.  A 15×15 kernel suppresses
# anything larger than 15 px — walls, clothing, railings all disappear.
#
# Tuning guide:
#   Ball disappears entirely → shrink kernel (try 11, then 9)
#   Background still leaks through → grow kernel (try 19, then 23)
#   Too many detections → raise TOPHAT_THRESHOLD
#   Ball not detected → lower TOPHAT_THRESHOLD
# tophat.py — run both directions
TOPHAT_MODE = "both"          # was "blackhat" — now handles all backgrounds

TOPHAT_KERNEL_SIZE = (1, 11) # works for sky AND textured backgrounds
TOPHAT_THRESHOLD   = 18       # moderate — not too sensitive for textured bg

# Frame diff — needs to work across all backgrounds
FRAME_DIFF_GAP       = 2
FRAME_DIFF_THRESHOLD = 15     # middle ground between sky (8) and textured (20)

# MOG2 — compromise history
MOG2_HISTORY        = 100
MOG2_VAR_THRESHOLD  = 18

# Blob filter — stay loose on shape, strict on size + trajectory
MIN_CIRCULARITY  = 0.25       # blurred ball won't be circular
MIN_SOLIDITY     = 0.45       # textured background blobs are very irregular
MAX_ASPECT_RATIO = 4.5        # motion blur can elongate significantly

# THE MOST IMPORTANT FILTER for multi-background footage:
# Trajectory validation becomes non-optional
ENABLE_TRAJECTORY  = True     # was False — now essential
TRAJECTORY_MIN_POINTS   = 5   # need 5 consistent positions
TRAJECTORY_MAX_RESIDUAL = 10.0 # pixels — parabola fit tolerance

# Mask combine — motion-first approach
MASK_COMBINE_MODE = "motion_primary"

# -----------------------------------------------------------------------------
# 3-FRAME DIFFERENCE  (motion/frame_diff.py)
# -----------------------------------------------------------------------------
# FRAME_GAP controls how many frames apart the two compared frames are.
#
#   Gap = 1  (consecutive)  → picks up ALL motion including slow clothes sway
#   Gap = 2  (skip 1 frame) → slow motion cancels out; fast ball remains
#   Gap = 3  (skip 2 frames)→ even more aggressive — only very fast objects
#
# For a windy background: start at Gap = 2.
# FRAME_DIFF_THRESHOLD: raise to suppress more background, lower to catch ball.
FRAME_DIFF_GAP       = 2
FRAME_DIFF_THRESHOLD = 20

# -----------------------------------------------------------------------------
# MOG2 BACKGROUND SUBTRACTION  (motion/bg_sub.py)
# -----------------------------------------------------------------------------
# history: number of frames used to model the background.
#   HIGH (500) → MOG2 eventually learns that swaying clothes ARE the
#                background and stops flagging them.  Best for windy scenes.
#                Downside: needs ~500 frames (≈17 s at 30fps) to fully adapt.
#   LOW  (100) → adapts fast but treats any slow motion as foreground.
#
# varThreshold: how different a pixel must be from the model to be foreground.
#   Lower  → more sensitive (catches subtle ball), more false positives.
#   Higher → less noise, may miss the ball.
#
# detectShadows: always False — shadows waste CPU and create false blobs.
MOG2_HISTORY        = 500
MOG2_VAR_THRESHOLD  = 25
MOG2_DETECT_SHADOWS = False

# -----------------------------------------------------------------------------
# MASK COMBINATION  (motion/mask_combine.py)
# -----------------------------------------------------------------------------
# How to merge the top-hat, frame-diff, and MOG2 masks into one final mask.
#
#   "or"             → any signal triggers (most sensitive, most noise)
#   "and"            → all signals must agree (cleanest, may miss ball)
#   "tophat_and_any" → top-hat AND (frame_diff OR mog2)  ← recommended start
#                      top-hat enforces circular small-blob shape,
#                      temporal signal confirms it actually moved.
MASK_COMBINE_MODE = "tophat_and_any"

# -----------------------------------------------------------------------------
# MORPHOLOGICAL CLEANING  (motion/morph_clean.py)
# -----------------------------------------------------------------------------
# CRITICAL: keep the kernel small.
# A (2,2) kernel only removes isolated single-pixel noise.
# A (5,5) kernel erases a 4–16 px² ball entirely — the original bug.
MORPH_KERNEL_SIZE = (2, 2)

# -----------------------------------------------------------------------------
# BLOB FILTER  (detection/blob_filter.py)
# -----------------------------------------------------------------------------
# Area bounds in pixels².
# At 50 m: ball ≈ 4–16 px².  Raise MIN if noise dots still appear.
MIN_BLOB_AREA = 5
MAX_BLOB_AREA = 100

# Circularity = 4π·area / perimeter²   (perfect circle = 1.0)
# Tennis ball: ~0.7–1.0.   Clothes / railings: ~0.1–0.4.
MIN_CIRCULARITY = 0

# Solidity = contour_area / convex_hull_area   (perfect convex shape = 1.0)
# Ball: ~0.8–1.0.   Fabric folds (concave): ~0.3–0.6.
MIN_SOLIDITY = 0.60

# Aspect ratio guard — reject blobs more elongated than this ratio.
# A tennis ball in flight is roughly 1:1 to 1.5:1.
MAX_ASPECT_RATIO = 3.0

# -----------------------------------------------------------------------------
# TRAJECTORY VALIDATION  (detection/trajectory_fit.py)
# -----------------------------------------------------------------------------
# A projectile under gravity follows a parabola: y = ax² + bx + c.
# Only tracks that fit this shape within the residual tolerance are kept.
# Turn ENABLE_TRAJECTORY on only after the tracker is producing stable IDs.
TRAJECTORY_MIN_POINTS  = 6     # min detections before fitting parabola
TRAJECTORY_MAX_RESIDUAL = 8.0  # max average pixel error allowed from fit

# -----------------------------------------------------------------------------
# KALMAN TRACKER  (tracking/kalman_tracker.py)
# -----------------------------------------------------------------------------
# max_distance: max pixels between predicted position and new detection
#   to still be considered the same object.
# max_missing:  frames a track can survive without any detection
#   (useful when ball briefly disappears behind a railing).
KALMAN_MAX_DISTANCE = 60
KALMAN_MAX_MISSING  = 8

# -----------------------------------------------------------------------------
# TRAIL  (tracking/trail_store.py)
# -----------------------------------------------------------------------------
TRAIL_LENGTH = 40   # number of past centroid positions kept per track

# -----------------------------------------------------------------------------
# DEBUG DISPLAY  (utils/debug_view.py)
# -----------------------------------------------------------------------------
# When ENABLE_DEBUG_VIEW = True these control which mask windows appear.
# Turn individual windows off to reduce screen clutter once that stage works.
DEBUG_SHOW_TOPHAT      = True
DEBUG_SHOW_FRAME_DIFF  = True
DEBUG_SHOW_BG_SUB      = True
DEBUG_SHOW_COMBINED    = True
DEBUG_SHOW_CLEANED     = True

DISPLAY_MAX_W = 1280
DISPLAY_MAX_H = 720

# Drawing colours (BGR)
COLOR_BBOX   = (0, 255, 0)    # green   bounding box
COLOR_ID     = (0, 0, 255)    # red     object ID label
COLOR_CENTER = (255, 0, 0)    # blue    centroid dot
COLOR_TRAIL  = (0, 255, 255)  # yellow  trajectory trail

FONT_FACE      = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE     = 0.45
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
        (MASK_COMBINE_MODE in ("or","and","tophat_and_any"), "MASK_COMBINE_MODE must be or/and/tophat_and_any"),
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
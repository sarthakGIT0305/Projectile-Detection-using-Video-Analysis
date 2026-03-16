# =============================================================================
# motion/bg_sub.py
# =============================================================================
# MOG2 (Mixture of Gaussians) background subtraction.
#
# WHY MOG2?
# ---------
# MOG2 models every pixel as a mixture of Gaussian distributions.  Over time
# it learns what the "normal" background looks like — including slow changes
# like swaying branches or shifting lighting.  Anything that doesn't fit the
# learned model is classified as foreground.
#
# Compared to frame differencing, MOG2:
#   + Handles gradual illumination changes gracefully
#   + Produces a cleaner foreground mask for objects that stop briefly
#   − Requires a "bootstrap" period (HISTORY frames) to build the model
#   − May suppress very slow-moving objects once they're learned
#
# The two masks (frame_diff + MOG2) complement each other and are combined
# downstream in mask_combine.py.
#
# PARAMETERS (set in config.py):
#   MOG2_HISTORY        — number of frames used to build background model
#   MOG2_VAR_THRESHOLD  — how different a pixel must be to count as foreground
#   MOG2_DETECT_SHADOWS — always False (shadow detection adds noise)
# =============================================================================

import cv2
import numpy as np


class BackgroundSubtractor:
    """Wraps cv2.BackgroundSubtractorMOG2 with project-specific defaults."""

    def __init__(self,
                 history: int       = 500,
                 var_threshold: float = 25,
                 detect_shadows: bool = False):
        self._history = history
        self._var_threshold = var_threshold
        self._detect_shadows = detect_shadows
        self._frame_count = 0
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Feed a BGR or grayscale frame and return the foreground mask.

        Parameters
        ----------
        frame : np.ndarray   BGR or single-channel image.

        Returns
        -------
        np.ndarray   8-bit binary mask (0 or 255).
        """
        # Accept either BGR or grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        self._frame_count += 1
        fg_mask = self.mog2.apply(gray)
        # MOG2 may return values like 127 for shadows; force binary
        _, mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        return mask

    def update(self, gray: np.ndarray) -> np.ndarray:
        """Backward-compatible alias for apply()."""
        return self.apply(gray)

    @property
    def is_warmed_up(self) -> bool:
        """True once enough frames have been processed to learn the background."""
        return self._frame_count >= self._history

    def reset(self):
        """Re-create the background model from scratch."""
        self._frame_count = 0
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history=self._history,
            varThreshold=self._var_threshold,
            detectShadows=self._detect_shadows,
        )


# Backward-compatible alias
BgSubtractor = BackgroundSubtractor


# =============================================================================
# SELF-TEST
# Run:  python motion/bg_sub.py
#
# What you will see — THREE panels side by side:
#   Left   — CLAHE-enhanced grayscale input
#   Middle — raw MOG2 foreground probability (bright = likely foreground)
#   Right  — binary mask after thresholding
#
# Interactive controls:
#   SPACE           — play / pause
#   LEFT / RIGHT    — step one frame (while paused)
#   R               — reset background model
#   Q / ESC         — quit
#
# What to look for:
#   During the first ~HISTORY frames the mask will be noisy as MOG2 learns.
#   After the bootstrap period, only genuinely moving objects should remain.
#   The ball should be a bright blob; swaying background should fade out.
# =============================================================================
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from config        import (VIDEO_PATH, ENABLE_CLAHE, ENABLE_ROI,
                               CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID,
                               ROI_PERCENT, ROI_ANCHOR_X, ROI_ANCHOR_Y,
                               MOG2_HISTORY, MOG2_VAR_THRESHOLD,
                               MOG2_DETECT_SHADOWS,
                               DISPLAY_MAX_W, DISPLAY_MAX_H)
    from video_reader  import VideoReader
    from preprocessing.clahe_gray import to_gray
    from preprocessing.roi        import apply_roi

    print("\n" + "="*52)
    print("  motion/bg_sub.py — self-test")
    print("="*52)
    print(f"\n  MOG2_HISTORY        = {MOG2_HISTORY}")
    print(f"  MOG2_VAR_THRESHOLD  = {MOG2_VAR_THRESHOLD}")
    print(f"  MOG2_DETECT_SHADOWS = {MOG2_DETECT_SHADOWS}")
    print()
    print("  Controls:")
    print("    SPACE       — play / pause")
    print("    LEFT/RIGHT  — step one frame (paused)")
    print("    R           — reset background model")
    print("    Q / ESC     — quit\n")

    try:
        reader = VideoReader(VIDEO_PATH)
    except RuntimeError as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    total  = reader.frame_count
    paused = False
    bg_sub = BackgroundSubtractor(history=MOG2_HISTORY,
                                   var_threshold=MOG2_VAR_THRESHOLD,
                                   detect_shadows=MOG2_DETECT_SHADOWS)

    for frame_no, frame in enumerate(reader.frames()):
        # Preprocessing
        if ENABLE_ROI:
            frame, _ = apply_roi(frame, ROI_PERCENT, ROI_ANCHOR_X, ROI_ANCHOR_Y)
        gray = to_gray(frame, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID, ENABLE_CLAHE)

        # MOG2
        raw_fg = bg_sub.mog2.apply(gray)
        mask   = bg_sub.update(gray)

        # Build display
        left   = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        middle = cv2.cvtColor(raw_fg, cv2.COLOR_GRAY2BGR)
        right  = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        h = left.shape[0]
        for panel, label in [
            (left,   "CLAHE gray"),
            (middle, "MOG2 raw fg"),
            (right,  "Binary mask"),
        ]:
            cv2.putText(panel, label,
                        (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (0, 255, 180), 1)

        combined = np.hstack([left, middle, right])
        ch, cw = combined.shape[:2]
        if cw > DISPLAY_MAX_W or ch > DISPLAY_MAX_H:
            scale    = min(DISPLAY_MAX_W / cw, DISPLAY_MAX_H / ch)
            combined = cv2.resize(combined, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_AREA)

        cv2.putText(combined,
                    f"Frame {frame_no}/{total-1}   history={MOG2_HISTORY}   "
                    f"var={MOG2_VAR_THRESHOLD}",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 100), 1)

        cv2.imshow("bg_sub.py — self-test", combined)
        wait = 0 if paused else 30
        key  = cv2.waitKey(wait) & 0xFF

        if key in (27, ord("q")):
            break
        elif key == ord(" "):
            paused = not paused
            print(f"  {'Paused' if paused else 'Playing'}")
        elif key == 83 or key == ord("d"):
            paused = True
        elif key == 81 or key == ord("a"):
            paused = True
        elif key == ord("r"):
            bg_sub.reset()
            print("  Background model reset.")

    cv2.destroyAllWindows()
    reader.release()
    print("\n  Self-test complete.\n")

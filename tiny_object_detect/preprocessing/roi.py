"""Region‑of‑interest helper.

The function crops the frame to a centered rectangle that is a given
percentage of the frame size.
"""

import cv2


def apply_roi(frame, percent: float):
    """Apply a centered ROI to *frame*.

    Parameters
    ----------
    frame : np.ndarray
        Input image.
    percent : float
        Fraction of the frame to keep (0 < percent <= 1).  Values
        outside this range are clamped.

    Returns
    -------
    np.ndarray
        Cropped image.
    """
    h, w = frame.shape[:2]
    # Clamp percent
    percent = max(0.0, min(1.0, percent))
    new_w, new_h = int(w * percent), int(h * percent)
    x = (w - new_w) // 2
    y = (h - new_h) // 2
    return frame[y : y + new_h, x : x + new_w]

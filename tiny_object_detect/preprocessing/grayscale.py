"""Grayscale conversion helper.

The function simply converts a BGR frame to an 8‑bit single channel image.
"""

import cv2


def to_gray(frame):
    """Return the grayscale version of *frame*.

    Parameters
    ----------
    frame : np.ndarray
        Input BGR image.

    Returns
    -------
    np.ndarray
        8‑bit single channel image.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

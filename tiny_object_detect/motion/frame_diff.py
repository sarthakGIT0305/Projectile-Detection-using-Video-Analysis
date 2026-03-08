"""Frame differencing helper.

Computes the absolute difference between two consecutive grayscale frames and
thresholds the result.
"""

import cv2
import numpy as np


def frame_difference(prev_gray, gray, threshold: int):
    """Return a binary mask of motion areas.

    Parameters
    ----------
    prev_gray, gray : np.ndarray
        Consecutive grayscale frames.
    threshold : int
        Pixel intensity change threshold.
    """
    diff = cv2.absdiff(gray, prev_gray)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    return mask

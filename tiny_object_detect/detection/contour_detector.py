"""Contour detection helper.

Finds external contours in a binary mask.
"""

import cv2


def detect_contours(mask):
    """Return a list of contours found in *mask*.

    The function uses `cv2.RETR_EXTERNAL` to avoid nested contours.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

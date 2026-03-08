"""Morphological mask cleaning helper.

Applies opening/closing and dilation to reduce noise and connect blobs.
"""

import cv2


def clean_mask(mask, kernel_size):
    """Return a cleaned binary mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask.
    kernel_size : tuple[int, int]
        Size of the structuring element.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cleaned

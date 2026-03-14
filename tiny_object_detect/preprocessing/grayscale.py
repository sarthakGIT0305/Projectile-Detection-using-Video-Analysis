"""Grayscale conversion helper"""

import cv2

# def to_gray(frame):
#     """Return the grayscale version of *frame*.
#     Parameters
#     ----------
#     frame : np.ndarray
#         Input BGR image.
#     Returns
#     -------
#     np.ndarray
#         8‑bit single channel image.
#     """
#     return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def to_gray(frame, clahe_clip=3.0, clahe_tile=(8, 8)):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
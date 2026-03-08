"""Background subtraction helper.

Wraps OpenCV's MOG2 background subtractor to keep a reusable instance.
"""

import cv2


class BackgroundSubtractor:
    def __init__(self, history=500, varThreshold=16, detectShadows=False):
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=varThreshold,
            detectShadows=detectShadows,
        )

    def apply(self, frame):
        return self.bg_sub.apply(frame)

"""Simple video reader wrapper.

Provides a thin abstraction over cv2.VideoCapture so that the rest of the
pipeline can obtain frames and the underlying capture object.
"""

import cv2

class VideoReader:
    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open video file: {path}")

    def release(self):
        self.cap.release()

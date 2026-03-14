"""Simple video reader wrapper"""

import cv2

class VideoReader:
    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open video file: {path}")

    def release(self):
        self.cap.release()

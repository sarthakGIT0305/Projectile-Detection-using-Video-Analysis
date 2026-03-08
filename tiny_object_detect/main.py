"""Main entry point for the tiny object detection pipeline.

The script demonstrates how the modular components are assembled and
executed frame‑by‑frame.  The pipeline is deliberately simple so that it
can be run as a script or imported into a larger application.
"""

import cv2
import time

from config import (
    VIDEO_PATH,
    ROI_PERCENT,
    FRAME_DIFF_THRESHOLD,
    MOG2_HISTORY,
    MOG2_VAR_THRESHOLD,
    MOG2_DETECT_SHADOWS,
    MORPH_KERNEL_SIZE,
    MIN_BLOB_AREA,
    MAX_BLOB_AREA,
    MAX_TRACKER_DISTANCE,
    MAX_TRACKER_MISSING_FRAMES,
    DISPLAY_SCALE,
    FONT_FACE,
    FONT_SCALE,
    FONT_THICKNESS,
    COLOR_BBOX,
    COLOR_ID,
    COLOR_CENTER,
)

# Import modular components
from video.video_reader import VideoReader
from preprocessing.grayscale import to_gray
from preprocessing.roi import apply_roi
from motion.frame_diff import frame_difference
from motion.background_subtraction import BackgroundSubtractor
from motion.mask_processing import clean_mask
from detection.contour_detector import detect_contours
from detection.blob_filter import filter_blobs
from tracking.multi_object_tracker import MultiObjectTracker
from utils.drawing import draw_objects


def main():
    """Run the detection‑tracking pipeline."""

    # ----- Setup -----
    reader = VideoReader(VIDEO_PATH)
    cap = reader.cap

    # Previous grayscale frame for frame differencing
    prev_gray = None

    # Background subtractor instance
    bg_subtractor = BackgroundSubtractor(
        history=MOG2_HISTORY,
        varThreshold=MOG2_VAR_THRESHOLD,
        detectShadows=MOG2_DETECT_SHADOWS,
    )

    # Tracker instance
    tracker = MultiObjectTracker(
        max_distance=MAX_TRACKER_DISTANCE,
        max_missing=MAX_TRACKER_MISSING_FRAMES,
    )

    # Main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. ROI restriction
        frame_roi = apply_roi(frame, ROI_PERCENT)

        # 2. Grayscale conversion
        gray = to_gray(frame_roi)

        # 3. Frame differencing
        if prev_gray is None:
            prev_gray = gray
            continue
        diff_mask = frame_difference(prev_gray, gray, FRAME_DIFF_THRESHOLD)
        prev_gray = gray

        # 4. Background subtraction
        bg_mask = bg_subtractor.apply(frame_roi)

        # 5. Combine motion masks
        motion_mask = cv2.bitwise_or(diff_mask, bg_mask)

        # 6. Morphological cleaning
        cleaned_mask = clean_mask(motion_mask, MORPH_KERNEL_SIZE)

        # 7. Contour detection
        contours = detect_contours(cleaned_mask)

        # 8. Blob filtering
        detections = filter_blobs(contours, MIN_BLOB_AREA, MAX_BLOB_AREA)

        # 9. Tracking
        objects = tracker.update(detections)

        # 10. Draw results
        output_frame = draw_objects(frame_roi, objects)

        # Display
        # Resize frame to a maximum of 1280x720 while preserving aspect ratio
        h, w = output_frame.shape[:2]
        max_w, max_h = 1280, 720
        if w > max_w or h > max_h:
            scale = min(max_w / w, max_h / h)
            resized_frame = cv2.resize(output_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            resized_frame = output_frame

        cv2.imshow("Detected objects", resized_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or 'q'
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

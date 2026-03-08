# config.py
"""Configuration parameters for the tiny object detection pipeline"""

import cv2

# Path to the video file
VIDEO_PATH = "D:\\Coding++\\web_dev_and_projects\\folderAssets\\open_cv_assets\\50m-1.mp4"

# Percentage of the frame to keep when applying ROI (relative to width and height)
# 1.0 means use the whole frame.
ROI_PERCENT = 1.0

# Frame difference threshold.  Pixels with intensity change above this
# value are considered part of motion.
FRAME_DIFF_THRESHOLD = 1

# MOG2 background subtractor parameters
MOG2_HISTORY = 300
MOG2_VAR_THRESHOLD = 16
MOG2_DETECT_SHADOWS = False

# Morphological kernel size for cleaning the motion mask
MORPH_KERNEL_SIZE = (5, 5)

# Blob area constraints (in pixels).  These should be tuned to the expected
# size of the tennis ball.  Objects smaller than min_area or larger than
# max_area are discarded.
MIN_BLOB_AREA = 5
MAX_BLOB_AREA = 500

# Tracker configuration
MAX_TRACKER_DISTANCE = 50  # max Euclidean distance to consider same object
MAX_TRACKER_MISSING_FRAMES = 5

# Display scaling factor for windows (0.5 = 50% of original size)
DISPLAY_SCALE = 0.5

# Font settings for drawing text
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1

# Colors used for drawing
COLOR_BBOX = (0, 255, 0)      # green
COLOR_ID   = (0, 0, 255)      # red
COLOR_CENTER = (255, 0, 0)    # blue

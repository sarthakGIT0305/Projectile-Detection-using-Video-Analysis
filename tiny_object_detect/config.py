# config.py
"""Configuration parameters for the tiny object detection pipeline"""

import cv2

VIDEO_PATH = "D:\\Coding++\\web_dev_and_projects\\folderAssets\\open_cv_assets\\distance_2.mp4"

# Percentage of the frame to keep when applying ROI (relative to width and height) 1.0 means use the whole frame.
ROI_PERCENT = 1.0

# Frame difference threshold. Pixels with intensity change above this value are considered part of motion.
FRAME_DIFF_THRESHOLD = 1

# MOG2 background subtractor parameters
MOG2_HISTORY = 200
MOG2_VAR_THRESHOLD = 20
MOG2_DETECT_SHADOWS = False

# Morphological kernel size for cleaning the motion mask
MORPH_KERNEL_SIZE = (2, 2)

# Blob area constraints (in pixels).  These should be tuned to the expected
# size of the tennis ball.  Objects smaller than min_area or larger than
# max_area are discarded.
MIN_BLOB_AREA = 1
MAX_BLOB_AREA = 50000

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
COLOR_BBOX = (0, 255, 0)     
COLOR_ID   = (0, 0, 255)      
COLOR_CENTER = (255, 0, 0)    

MIN_CIRCULARITY = 0.15        # NEW
CLAHE_CLIP_LIMIT = 3.0        # NEW
CLAHE_TILE_GRID  = (8, 8)     # NEW
COLOR_TRAIL = (0, 255, 255)   # NEW - for trajectory drawing
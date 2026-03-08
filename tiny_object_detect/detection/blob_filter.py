"""Blob filtering helper.
Filters contours based on area and extracts bounding box and centroid.
"""

import cv2
import numpy as np
from typing import List, Dict


def filter_blobs(contours, min_area, max_area) -> List[Dict]:
    """Filter contours and return detection objects.

    Each detection is a dictionary:
    {
        "bbox": (x, y, w, h),
        "center": (cx, cy),
        "area": area,
    }
    """
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2
        detections.append({"bbox": (x, y, w, h), "center": (cx, cy), "area": area})
    return detections

"""Blob filtering helper.
Filters contours based on area and extracts bounding box and centroid.
"""

from config import *
import cv2
import math
import numpy as np
from typing import List, Dict

def filter_blobs(contours, min_area, max_area, MIN_CIRCULARITY=0.3) -> List[Dict]:
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

        # Circularity check — rejects leaves, edges, linear noise
        perimeter = cv2.arcLength(cnt, True)
        if perimeter < 1e-6:
            continue
        circularity = (4.0 * math.pi * area) / (perimeter ** 2)
        if circularity < MIN_CIRCULARITY:   # pass min_circularity as parameter
            continue

        # Aspect ratio check — reject blobs more than 4:1
        x, y, w, h = cv2.boundingRect(cnt)
        if min(w, h) > 0 and max(w, h) / min(w, h) > 4.0:
            continue

        cx, cy = x + w // 2, y + h // 2
        detections.append({"bbox": (x, y, w, h), "center": (cx, cy),
                        "area": area, "circularity": round(circularity, 3)})
    # for cnt in contours:
    #     area = cv2.contourArea(cnt)
    #     if area < min_area or area > max_area:
    #         continue
    #     x, y, w, h = cv2.boundingRect(cnt)
    #     cx, cy = x + w // 2, y + h // 2
    #     detections.append({"bbox": (x, y, w, h), "center": (cx, cy), "area": area})
    return detections

"""Utility functions for visualizing results.

Provides a simple helper that draws bounding boxes, object IDs and
centroids on a frame.
"""

import cv2
from typing import List, Dict


def draw_objects(frame, objects: List[Dict]):
    """Draw detections and IDs on *frame*.

    Parameters
    ----------
    frame : np.ndarray
        Image to draw on.
    objects : list[dict]
        Each dict must contain "id", "bbox" and "center".
    """
    for obj in objects:
        x, y, w, h = obj["bbox"]
        cx, cy = obj["center"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{obj['id']}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)
    return frame

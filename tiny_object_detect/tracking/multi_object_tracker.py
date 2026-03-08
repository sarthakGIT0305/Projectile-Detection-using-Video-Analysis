"""Centroid‑based multi‑object tracker.

This simple implementation keeps a dictionary of object IDs mapped to
their last known centroid and a counter of missed frames.  When a new
detection is close enough to an existing centroid it is assigned that
ID, otherwise a new ID is created.  Objects that are missing for a
configurable number of frames are removed.
"""

from collections import defaultdict
import math
from typing import List, Dict


class MultiObjectTracker:
    def __init__(self, max_distance: int, max_missing: int):
        self.next_id = 0
        self.objects: Dict[int, Dict] = {}  # id -> {"center": (x,y), "missing": int}
        self.max_distance = max_distance
        self.max_missing = max_missing

    @staticmethod
    def _dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def update(self, detections: List[Dict]) -> List[Dict]:
        """Update tracker with new detections.

        Parameters
        ----------
        detections : list[dict]
            Each dict must contain a "center" key.

        Returns
        -------
        list[dict]
            List of objects with keys: "id", "bbox", "center".
        """
        # Prepare current centroids
        current_centroids = [d["center"] for d in detections]
        # Match to existing objects
        matched = set()
        for obj_id, data in list(self.objects.items()):
            # Find nearest detection
            nearest = None
            nearest_dist = None
            nearest_idx = None
            for idx, cent in enumerate(current_centroids):
                d = self._dist(cent, data["center"])
                if nearest is None or d < nearest_dist:
                    nearest = cent
                    nearest_dist = d
                    nearest_idx = idx
            if nearest_dist is not None and nearest_dist <= self.max_distance:
                # Update existing object
                self.objects[obj_id]["center"] = nearest
                self.objects[obj_id]["missing"] = 0
                matched.add(nearest_idx)
            else:
                # No match, increase missing counter
                self.objects[obj_id]["missing"] += 1

        # Remove objects that have been missing too long
        to_remove = [obj_id for obj_id, data in self.objects.items() if data["missing"] > self.max_missing]
        for obj_id in to_remove:
            del self.objects[obj_id]

        # Add new objects for unmatched detections
        for idx, det in enumerate(detections):
            if idx not in matched:
                self.objects[self.next_id] = {"center": det["center"], "missing": 0}
                det["id"] = self.next_id
                self.next_id += 1

        # Prepare output list
        output = []
        for obj_id, data in self.objects.items():
            # Find the detection that has this id
            det = next((d for d in detections if d.get("id") == obj_id), None)
            if det:
                output.append({"id": obj_id, "bbox": det["bbox"], "center": det["center"]})
        return output

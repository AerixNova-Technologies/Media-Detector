from __future__ import annotations
import numpy as np

class Track:
    """Lightweight container for a confirmed track."""
    __slots__ = ("track_id", "bbox", "conf", "cls_id")

    def __init__(self, track_id: int, bbox: list[float], conf: float = 0.0, cls_id: int = 0):
        self.track_id = track_id          # unique integer ID
        self.bbox = bbox                  # [x1, y1, x2, y2]
        self.conf = conf
        self.cls_id = cls_id

    def __repr__(self) -> str:
        return f"Track(id={self.track_id}, cls={self.cls_id}, bbox={[round(v) for v in self.bbox]})"

class PersonTracker:
    """
    Wraps ByteTrack (via YOLOv8) for tracking.
    Now uses the built-in tracking from PersonDetector.
    """
    def __init__(self, **kwargs):
        pass

    def update(self, detector_results: list[list[float]]) -> list[Track]:
        """
        Converts raw results from PersonDetector.track() into Track objects.
        """
        # detector_results is list of [x1, y1, x2, y2, conf, tid, cls_id]
        tracks: list[Track] = []
        for res in detector_results:
            if len(res) == 7:
                x1, y1, x2, y2, conf, tid, cls_id = res
            else:
                x1, y1, x2, y2, conf, tid = res
                cls_id = 0 # Default to person
            tracks.append(Track(track_id=int(tid), bbox=[x1, y1, x2, y2], conf=conf, cls_id=int(cls_id)))
        return tracks

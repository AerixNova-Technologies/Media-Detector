from __future__ import annotations
import numpy as np

class Track:
    """Lightweight container for a confirmed track."""
    __slots__ = ("track_id", "bbox", "conf")

    def __init__(self, track_id: int, bbox: list[float], conf: float = 0.0):
        self.track_id = track_id          # unique integer ID
        self.bbox = bbox                  # [x1, y1, x2, y2]
        self.conf = conf

    def __repr__(self) -> str:
        return f"Track(id={self.track_id}, bbox={[round(v) for v in self.bbox]})"

class PersonTracker:
    """
    Wraps ByteTrack (via YOLOv8) for person tracking.
    Now uses the built-in tracking from PersonDetector.
    """
    def __init__(self, **kwargs):
        # We hold nothing here as the actual tracker state is managed by the YOLO model in PersonDetector
        pass

    def update(self, detector_results: list[list[float]]) -> list[Track]:
        """
        Converts raw results from PersonDetector.track() into Track objects.
        """
        # detector_results is expected to be list of [x1, y1, x2, y2, conf, tid]
        tracks: list[Track] = []
        for x1, y1, x2, y2, conf, tid in detector_results:
            tracks.append(Track(track_id=int(tid), bbox=[x1, y1, x2, y2], conf=conf))
        return tracks

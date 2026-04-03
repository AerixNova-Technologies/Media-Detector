"""
utils/stream.py
─────────────────────────────────────────────────────────────────────────────
Video stream management helpers.

Supports:
    - Webcam (integer index)
    - RTSP / HTTP CCTV streams (URL string)
    - Video files (path string)
"""

from __future__ import annotations

import os
import time
import logging
import threading

import cv2
import numpy as np
from urllib.parse import urlparse, quote

import re

def encode_rtsp_url(url: str) -> str:
    """Safely encodes credentials in an RTSP URL if they contain special characters."""
    if not isinstance(url, str) or "@" not in url:
        return url
        
    # Pattern to match: scheme://user:password@host...
    # We look for the last '@' that comes before the path start (or end of string)
    match = re.match(r"^(rtsp|http|https)://(.+):(.+)@([^/?#]+)(.*)$", url)
    if not match:
        return url
        
    scheme, user, password, host, rest = match.groups()
    
    # Only encode if not already percent-encoded
    u = user if "%" in user else quote(user, safe='')
    pw = password if "%" in password else quote(password, safe='')
    
    return f"{scheme}://{u}:{pw}@{host}{rest}"

log = logging.getLogger(__name__)


class VideoStream:
    """
    Thin OpenCV VideoCapture wrapper with:
      * automatic reconnect on RTSP streams
      * frame skipping to target a desired FPS
      * basic frame validation
    """

    def __init__(
        self,
        source: int | str = 0,
        width: int = 1280,
        height: int = 720,
        target_fps: float = 25.0,
        reconnect_delay: float = 15.0,
        transport: str = "tcp",
    ):
        self.source = source
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.reconnect_delay = reconnect_delay
        self.transport = transport.lower() if transport else "tcp"

        self._cap: cv2.VideoCapture | None = None
        
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._latest_frame = None
        self._ok = False
        
        self._open()

    # ------------------------------------------------------------------
    def _open(self) -> None:
        if self._cap is not None:
            self.release()

        src = self.source
        is_url = isinstance(src, str) and (
            src.startswith("rtsp://")
            or src.startswith("http://")
            or src.startswith("https://")
            or ".m3u8" in src
        )

        if is_url:
            # Safely encode credentials (handles passwords with @)
            src = encode_rtsp_url(src)

            # --- FIRMWARE COMPATIBILITY FIX ---
            # Increase stimeout to 5s for older cameras (was 3s)
            # Use user-defined transport (TCP/UDP)
            trans = self.transport if self.transport in ["tcp", "udp"] else "tcp"
            options = f"rtsp_transport;{trans}|fflags;nobuffer|probesize;32|analyzeduration;0|stimeout;10000000|rtsp_flags;prefer_tcp"
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = options
            
            try:
                self._cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG) # CAP_FFMPEG is better on Windows
            finally:
                # IMPORTANT: Clear it so it doesn't break Webcams or other captures
                if "OPENCV_FFMPEG_CAPTURE_OPTIONS" in os.environ:
                    del os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]
        else:
            # On Windows, DirectShow (CAP_DSHOW) is MUCH faster for webcam init
            if os.name == 'nt' and isinstance(src, int):
                self._cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            else:
                self._cap = cv2.VideoCapture(src)

        if self._cap.isOpened():
            # Request MJPG and set properties - wrap in try/except to prevent hardware driver crashes
            try:
                if not is_url:
                    self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self._cap.set(cv2.CAP_PROP_FPS,          self.target_fps)
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # absolute lowest latency
            except Exception as e:
                log.warning("Could not set all camera properties: %s", e)
            
            # Start the frame-purging background thread
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._update, daemon=True, name="VideoStream")
            self._thread.start()
            
            log.info("Stream opened (Threaded): %s", src)
        else:
            log.warning("Failed to open stream: %s", src)

    # ------------------------------------------------------------------
    def _update(self):
        """Background thread that constantly reads frames to prevent lagging behind."""
        while not self._stop_event.is_set():
            if self._cap is None or not self._cap.isOpened():
                # Attempt to reconnect if stream is closed
                log.info("VideoStream: Attempting background reconnect...")
                self._open_capture()
                if self._cap is None or not self._cap.isOpened():
                    time.sleep(self.reconnect_delay)
                    continue
            
            # --- 1. CAPTURE DATA FIRST ---
            ok, frame = self._cap.read()
            
            # --- 2. FRAME QUALITY GUARD (Detecting Static / Black Screen / Driver Crash) ---
            # If the frame has extremely high variance (colored static) or is black (all zeros)
            is_bad = False
            if ok and frame is not None:
                # Optimized health check: Standard deviation of a small center crop
                h, w = frame.shape[:2]
                center_crop = frame[h//4:3*h//4, w//4:3*w//4]
                std = np.std(center_crop)
                
                # 1. Colored static noise (High Variance)
                if std > 120: 
                    is_bad = True
                    log.warning(f"VideoStream: High-frequency noise detected (StdDev: {std:.1f}). Possible driver crash.")
                
                # 2. Black Screen / Frozen Buffer (EXTREMELY Low Variance)
                elif std < 0.5:
                    is_bad = True
                    log.warning(f"VideoStream: Black screen / Dead stream detected (StdDev: {std:.1f}). Possible driver stall.")
            
            with self._lock:
                if ok and not is_bad:
                    # CRITICAL FIX: Use .copy() to decouple the frame memory from the camera driver's internal buffer.
                    # This prevents 'HUD Ghosting' and 'Doubling' seen on some Windows drivers.
                    self._latest_frame = frame.copy() 
                    self._ok = True # Internal
                    self.ok = True  # Public (for CameraManager)
                    self._last_bad_frame_count = 0 
                else:
                    self._ok = False
                    self.ok = False # Signal to UI: "RECONNECTING"
                    if is_bad:
                        self._last_bad_frame_count = getattr(self, "_last_bad_frame_count", 0) + 1
                        if self._last_bad_frame_count > 10:
                            log.error("VideoStream: Persistent BAD quality (Black/Static). Forcing HARD RESET.")
                            self._open_capture() # Full re-init of cap
                            self._last_bad_frame_count = 0
            
            if not ok or is_bad:
                time.sleep(0.5) # Throttle on failure
                continue

    # ------------------------------------------------------------------
    def _open_capture(self) -> None:
        """Internal helper to physically open the CV2 capture."""
        src = self.source
        is_url = isinstance(src, str) and any(src.startswith(p) for p in ["rtsp://", "http://", "https://"])
        
        if is_url:
            # Safely encode credentials (handles passwords with @)
            src = encode_rtsp_url(src)

            # PROFESSIONAL SURVEILLANCE FLAGS (REFINED FOR OLDER FIRMWARE)
            trans = self.transport if self.transport in ["tcp", "udp"] else "tcp"
            options = f"rtsp_transport;{trans}|fflags;nobuffer|probesize;32|analyzeduration;0|stimeout;10000000|rtsp_flags;prefer_tcp"
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = options
            try:
                self._cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            finally:
                if "OPENCV_FFMPEG_CAPTURE_OPTIONS" in os.environ:
                    del os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]
        else:
            if os.name == 'nt' and isinstance(src, int):
                # Try index 0, then 1, then 2 if 0 is busy/unresponsive
                for idx in [src, 1, 2]:
                    log.info(f"VideoStream: Attempting DSHOW open on index {idx}")
                    self._cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                    if self._cap.isOpened():
                        self.source = idx # Update source to the working one
                        break
            else:
                self._cap = cv2.VideoCapture(src)

        if self._cap and self._cap.isOpened():
            try:
                if not is_url:
                    self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
                log.info(f"VideoStream: Stream opened successfully on {self.source}")
            except: pass
        else:
            self._cap = None

    # ------------------------------------------------------------------
    def read(self) -> tuple[bool, np.ndarray | None]:
        """Grab the absolute most recent frame (Non-blocking)."""
        with self._lock:
            return self._ok, self._latest_frame

    # ------------------------------------------------------------------
    def release(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ------------------------------------------------------------------
    @property
    def fps(self) -> float:
        if self._cap and self._cap.isOpened():
            return float(self._cap.get(cv2.CAP_PROP_FPS) or self.target_fps)
        return self.target_fps

    # ------------------------------------------------------------------
    def __enter__(self) -> "VideoStream":
        return self

    def __exit__(self, *_) -> None:
        self.release()

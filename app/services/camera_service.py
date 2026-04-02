"""
core/camera_manager.py  –  Camera Manager
──────────────────────────────────────────────────────────────────────────────
Manages camera switching with ZERO model reloads.
"""
from __future__ import annotations
import logging
import os
import queue
import threading
import time
import cv2
import numpy as np

from app.core import config as cfg
from app.core.data_types import AIResult
from app.pipelines.ai_pipeline import AIPipeline
from app.utils.drawing import draw_face, draw_motion_mask, draw_person, draw_status_bar
from app.utils.fps_counter import FPSCounter
from app.utils.stream import VideoStream

log = logging.getLogger("camera_manager")

# Shared motion events counter (read by /api/dashboard_info)
_total_motion_events: int = 0

class CameraManager:
    """
    Manages the complete video → AI → display pipeline.
    AI models are loaded exactly once and reused across camera switches.
    """
    def __init__(self):
        self._result_store: dict[str, AIResult] = {} # {cam_name: AIResult}
        self._result_lock     = threading.Lock()
        self._frame_queue     = queue.Queue(maxsize=2)
        self._jpeg_buf        = b""
        self._jpeg_lock       = threading.Lock()
        self._render_thread   = None
        self._render_stop_evt = threading.Event()
        self._fps_val         = 0.0
        self._active          = False
        self.camera_name      = ""
        self.stream_id        = 0
        self._monitoring_nodes: dict[str, dict] = {} # {name: {"id": slug, "stream": VideoStream, "thread": Thread, "stop_evt": Event, "last_frame": ndarray, "roles": list}}
        self._monitoring_lock = threading.Lock()
        self._pipeline        = None

        # Start background initialization to prevent Flask hang
        threading.Thread(target=self._init_in_background, daemon=True, name="CmMgrInit").start()

    def _init_in_background(self):
        """Heavy AI loading handled in a separate thread."""
        log.info("Loading AI models (one-time startup) in background …")
        upload_folder = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "static", "uploads",
        )
        self._pipeline = AIPipeline(
            in_queue=self._frame_queue,
            result_store=self._result_store,
            result_lock=self._result_lock,
            upload_folder=upload_folder,
        )
        self._pipeline.start()
        log.info("Camera Manager: AI models ready.")

    # ── Start / Stop ──────────────────────────────────────────────────────────
    def start(self, source, name: str = "Camera", **kwargs) -> None:
        """Switch to a new camera instantly by promoting its background monitoring node."""
        log.info("Switching view to: %s", name)
        roles = kwargs.get("roles", [])
        gate_id = kwargs.get("gate_id")
        transport = kwargs.get("transport", "tcp")
        
        # Ensure the camera is being monitored in the background
        self.start_monitoring(source, name, roles=roles, gate_id=gate_id, transport=transport)
        
        self.camera_name = name
        if self._pipeline:
            self._pipeline.camera_name = name
            self._pipeline.camera_roles = roles
            self._pipeline.camera_gate = gate_id
            
        self._active = True
        self.stream_id += 1 

        if self._render_thread is None or not self._render_thread.is_alive():
            self._render_stop_evt = threading.Event()
            self._render_thread = threading.Thread(
                target=self._render_loop,
                args=(self._render_stop_evt,),
                daemon=True,
                name="GlobalRender",
            )
            self._render_thread.start()

    def reload_faces(self) -> None:
        if self._pipeline:
            self._pipeline.reload_faces()

    def stop_stream(self) -> None:
        self._active = False
        with self._jpeg_lock:
            self._jpeg_buf = b""
        self._fps_val = 0.0

    def start_monitoring(self, source, name: str, roles: list[str], camera_id: str | None = None, gate_id: str | None = None, transport: str = "tcp") -> bool:
        with self._monitoring_lock:
            if name in self._monitoring_nodes:
                self._monitoring_nodes[name]["roles"] = roles
                self._monitoring_nodes[name]["gate_id"] = gate_id
                if camera_id:
                    self._monitoring_nodes[name]["id"] = camera_id
                return True
            
            from app.utils.stream import encode_rtsp_url
            source = encode_rtsp_url(source)
            
            log.info(f"AI: Starting background monitoring for '{name}' (source={source})")
            stop_evt = threading.Event()
            try:
                stream = VideoStream(source=source, width=cfg.FRAME_WIDTH, height=cfg.FRAME_HEIGHT, target_fps=5.0, transport=transport)
                thread = threading.Thread(
                    target=self._monitoring_loop,
                    args=(stop_evt, stream, name, roles, gate_id),
                    daemon=True,
                    name=f"Monitor_{name}"
                )
                thread.start()
                self._monitoring_nodes[name] = {
                    "id": camera_id or name,
                    "stream": stream, 
                    "stop_evt": stop_evt, 
                    "thread": thread, 
                    "last_frame": None,
                    "roles": roles,
                    "gate_id": gate_id
                }
                return True
            except Exception as e:
                log.error(f"AI: Failed to start monitoring for '{name}': {e}")
                return False

    def stop_monitoring(self, name: str) -> None:
        with self._monitoring_lock:
            if name in self._monitoring_nodes:
                node = self._monitoring_nodes.pop(name)
                node["stop_evt"].set()

    def stop_monitoring_by_ip(self, ip: str) -> None:
        to_stop = []
        with self._monitoring_lock:
            for name, node in self._monitoring_nodes.items():
                src = str(node["stream"].source)
                if ip in src:
                    to_stop.append(name)
        for name in to_stop:
            log.info(f"AI: Stopping monitor for '{name}' to allow manual test on IP {ip}")
            self.stop_monitoring(name)

    def stop_all(self) -> None:
        self.stop_stream()
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline.join(timeout=8)
            self._pipeline = None

    # ── Loops ─────────────────────────────────────────────────────────────────
    def _render_loop(self, stop_evt: threading.Event) -> None:
        fps_ctr = FPSCounter(window=30)
        while not stop_evt.is_set():
            if not self._active or not self.camera_name:
                time.sleep(0.1)
                continue

            node = None
            with self._monitoring_lock:
                node = self._monitoring_nodes.get(self.camera_name)
            
            # Stream Health Check (Using new public .ok signal from VideoStream)
            stream_ok = getattr(node["stream"], "ok", False) if node and "stream" in node else False

            if not node or node.get("last_frame") is None or not stream_ok:
                # IMPORTANT: Use a CLEAN black frame to avoid HUD Ghosting/Doubling
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                msg = f"RECONNECTING: {self.camera_name}..." if node else f"LOADING: {self.camera_name}..."
                cv2.putText(frame, msg, (420, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 140, 255), 2)
                
                # Draw a simple clean status bar on the placeholder
                draw_status_bar(frame, motion=False, n_persons=0, fps=self._fps_val, entries=getattr(result, 'entries', 0) if 'result' in locals() else 0, exits=0)
            else:
                # Take a fresh copy of the CLEAN frame from the monitoring node
                frame = node["last_frame"].copy()
                
                # RE-FIX: Update the FPS display value for the HUD
                # This ensures the counter in the top-left corner reflects actual render speed.
                self._fps_val = fps_ctr.tick()

                is_paused = "paused" in node.get("roles", []) if node else False
                if not is_paused:
                    with self._result_lock:
                        result = self._result_store.get(self.camera_name, AIResult())
                    
                    # RE-FIX: Change detection grace period from 2.0 to 10.0 seconds
                    # This prevents the green/red boxes from "flickering" out during CPU spikes.
                    age = time.monotonic() - result.timestamp
                    tracks = result.tracks if age < 10.0 else []

                    # Draw boxes on the FRESH copy
                    for tr in tracks:
                        draw_person(frame, tr.bbox_full, tr.track_id, emotion=tr.emotion, action=tr.action, identity=tr.identity)
                    
                    # Draw status bar last
                    draw_status_bar(frame, motion=result.motion if age < 10.0 else False, 
                                    n_persons=len(tracks), fps=self._fps_val, entries=result.entries, exits=result.exits)
                else:
                    self._fps_val = fps_ctr.tick()
                    draw_status_bar(frame, motion=False, n_persons=0, fps=self._fps_val, entries=0, exits=0, paused=True)

            ok2, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, cfg.JPEG_QUALITY_HIGH])
            if ok2:
                with self._jpeg_lock:
                    self._jpeg_buf = jpg.tobytes()
            
            time.sleep(1.0 / cfg.TARGET_FPS)

    def _monitoring_loop(self, stop_evt: threading.Event, stream: VideoStream, name: str, roles: list[str], gate_id: str | None = None):
        cap_idx = 0
        last_frame_ts = time.monotonic()
        last_log_ts = 0
        try:
            while not stop_evt.is_set():
                ok, frame = stream.read()
                if not ok or frame is None:
                    now = time.monotonic()
                    if now - last_frame_ts > 5.0:
                        if now - last_log_ts > 60.0:
                            log.warning(f"RTSP Reconnect: Stream lost for {name}, attempting recovery...")
                            last_log_ts = now
                        try:
                            stream.release()
                            time.sleep(1.0)
                            stream.open()
                        except: pass
                        last_frame_ts = now
                    time.sleep(0.1)
                    continue

                last_frame_ts = time.monotonic()
                with self._monitoring_lock:
                    if name in self._monitoring_nodes:
                        self._monitoring_nodes[name]["last_frame"] = frame
                        current_roles = self._monitoring_nodes[name]["roles"]
                        current_gate  = self._monitoring_nodes[name]["gate_id"]

                if "paused" not in current_roles:
                    if cap_idx % cfg.AI_THREAD_FRAME_SKIP == 0:
                        fh, fw = frame.shape[:2]
                        ai_f = cv2.resize(frame, (cfg.AI_FRAME_WIDTH, cfg.AI_FRAME_HEIGHT))
                        try:
                            self._frame_queue.put_nowait((ai_f, name, current_roles, (fw, fh), current_gate))
                        except queue.Full: pass

                cap_idx += 1
                time.sleep(0.01)
        finally:
            try: stream.release()
            except: pass
            log.info(f"AI: Monitoring thread exited for: {name}")

    # ── Public Accessors ──────────────────────────────────────────────────────
    def get_jpeg(self) -> bytes:
        with self._jpeg_lock:
            return self._jpeg_buf

    def get_stats(self) -> dict:
        all_results = {}
        with self._result_lock:
            store_snapshot = dict(self._result_store.items())

        now = time.monotonic()
        for cam_name, r in store_snapshot.items():
            age = now - r.timestamp
            if age > cfg.RESULT_MAX_AGE_SEC: continue
            
            is_paused = False
            with self._monitoring_lock:
                node = self._monitoring_nodes.get(cam_name)
                if node: is_paused = "paused" in node.get("roles", [])

            all_results[cam_name] = {
                "paused": is_paused,
                "motion": r.motion if not is_paused else False,
                "persons": len(r.tracks) if not is_paused else 0,
                "entries": r.entries,
                "exits": r.exits,
                "tracks": [
                    {"id": t.track_id, "emotion": t.emotion or "–", "action": t.action or "–", "identity": t.identity or "", "staff_id": t.display_id or ""}
                    for t in (r.tracks if not is_paused else [])
                ]
            }
        
        active_data = all_results.get(self.camera_name, {"paused": False, "motion": False, "persons": 0, "entries": 0, "exits": 0, "tracks": []})
        return {
            "active": self._active, "camera": self.camera_name, "fps": round(self._fps_val, 1),
            "paused": active_data["paused"], "motion": active_data["motion"], "persons": active_data["persons"],
            "entries": active_data["entries"], "exits": active_data["exits"], "tracks": active_data["tracks"],
            "all_results": all_results
        }

    def get_background_jpeg(self, camera_id: str) -> bytes | None:
        with self._monitoring_lock:
            target_name = None
            for name, node in self._monitoring_nodes.items():
                if str(node.get("id")) == str(camera_id) or name == str(camera_id):
                    target_name = name
                    break
            if not target_name: return None
            node = self._monitoring_nodes.get(target_name)
            if not node or node.get("last_frame") is None: return None
            frame = node["last_frame"].copy()

        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, cfg.JPEG_QUALITY_LOW])
        return jpeg.tobytes() if ok else None

"""
app/routes/camera.py    Camera Management Routes Blueprint

Handles camera streaming, selection, and MJPEG video feed.

Routes:
  GET  /api/cameras             JSON list of all cameras
  POST /api/select              Start streaming a selected camera
  POST /api/stop                Stop current camera stream
  GET  /api/stream_id           Returns current stream ID for frontend polling
  GET  /video_feed             MJPEG live stream endpoint
  GET  /api/local_cameras      List user's local IP cameras
  GET  /api/local_cameras/detected Auto-detect local/LAN cameras
  POST /api/local_cameras/detected/metadata Query ONVIF device metadata
  POST /api/local_cameras      Add a new local IP camera
  DELETE /api/local_cameras/<id> Remove a local camera
"""

from __future__ import annotations

import logging
import os
import time
import threading
import uuid

import base64
import cv2
from flask import Blueprint, Response, jsonify, request, session

from app.core.security import login_required
from app.db.session import get_db_connection
from app.core.extensions import cam_mgr
from app.services.camera_discovery import detect_cameras, fetch_onvif_device_info

log = logging.getLogger("camera")

camera_bp = Blueprint("camera", __name__)

# ΓöÇΓöÇ Imou credentials (loaded from environment) ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
APP_ID     = os.environ.get("IMOU_APP_ID", "")
APP_SECRET = os.environ.get("IMOU_APP_SECRET", "")


# ΓöÇΓöÇ Camera cache ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
_camera_cache: list[dict] = []
_camera_cache_ts: float   = 0.0
_camera_lock = __import__("threading").Lock()

_preview_tokens: dict[str, dict] = {}
_preview_lock = threading.Lock()
_preview_token_ttl_sec = 30


def _cleanup_expired_preview_tokens(now: float | None = None) -> None:
    ts_now = now if now is not None else time.time()
    expired = [token for token, meta in _preview_tokens.items() if meta["expires_at"] <= ts_now]
    for token in expired:
        _preview_tokens.pop(token, None)


def _normalize_rtsp_path(path: str) -> str:
    value = (path or "").strip()
    if not value:
        return ""
    if value.startswith("/"):
        return value
    return f"/{value}"


from urllib.parse import quote

def _build_rtsp_url(ip: str, port: int, user: str, password: str, path: str) -> str:
    normalized_path = _normalize_rtsp_path(path)
    if user and password:
        quoted_user = quote(user, safe='')
        quoted_pass = quote(password, safe='')
        return f"rtsp://{quoted_user}:{quoted_pass}@{ip}:{port}{normalized_path}"
    return f"rtsp://{ip}:{port}{normalized_path}"


def _issue_preview_token(rtsp_url: str, owner_email: str | None) -> str:
    token = uuid.uuid4().hex
    now = time.time()
    with _preview_lock:
        _cleanup_expired_preview_tokens(now)
        _preview_tokens[token] = {
            "rtsp_url": rtsp_url,
            "owner_email": owner_email,
            "expires_at": now + _preview_token_ttl_sec,
        }
    return token


from app.utils.stream import encode_rtsp_url

def _test_rtsp_connection(rtsp_url: str, transport: str = "tcp") -> tuple[bool, object | None]:
    # Use safe encoding (handles passwords with @)
    rtsp_url = encode_rtsp_url(rtsp_url)

    # Use robust surveillance flags FOR TESTING (Refined for firmware compatibility)
    trans = transport if transport in ["tcp", "udp"] else "tcp"
    options = f"rtsp_transport;{trans}|fflags;nobuffer|probesize;32|analyzeduration;0|stimeout;5000000|rtsp_flags;prefer_tcp"
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = options
    
    log.info("Opening RTSP connection: %s", rtsp_url)
    try:
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    finally:
        if "OPENCV_FFMPEG_CAPTURE_OPTIONS" in os.environ:
            del os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]
    
    if not cap.isOpened():
        log.warning("OpenCV failed to open RTSP stream")
        cap.release()
        return False, None

    ok, frame = False, None
    # Retry for up to ~5 seconds (faster timeout for UI responsiveness)
    for i in range(25):
        ok, frame = cap.read()
        if ok and frame is not None:
            log.info("RTSP Test SUCCESS: Frame read after %d attempts", i+1)
            break
        time.sleep(0.2)
    
    cap.release()
    if not ok or frame is None:
        log.warning("RTSP Test FAILED: Stream opened but no frames could be read")
        return False, None
    return True, frame


def _generate_mjpeg(rtsp_url: str, max_duration_sec: float | None = None):
    # Safely encode credentials (handles passwords with @)
    rtsp_url = encode_rtsp_url(rtsp_url)
    
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        return

    deadline = (time.time() + max_duration_sec) if max_duration_sec else None
    try:
        while True:
            if deadline and time.time() >= deadline:
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            encoded, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if not encoded:
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
            time.sleep(0.04)
    finally:
        cap.release()


def _fetch_imou_cameras() -> list[dict]:
    """Return list of Imou cameras with {id, name, status, type, dev_id, base}."""
    try:
        from app.services.ai.imou_connector import ImouAPI, _find_working_datacenter, _get_device_status
        base = _find_working_datacenter()
        if not base:
            return []
        api = ImouAPI(APP_ID, APP_SECRET, base)
        api.get_token()
        devices = api.list_devices()
        result = []
        for i, dev in enumerate(devices):
            dev_id = dev.get("deviceId") or dev.get("deviceID") or f"dev{i}"
            name   = dev.get("name") or dev.get("deviceName") or "Imou Camera"
            status = _get_device_status(dev)
            result.append({
                "id":     f"imou_{dev_id}",
                "name":   name,
                "status": status,
                "type":   "imou",
                "dev_id": dev_id,
                "base":   base,
            })
        return result
    except Exception as e:
        log.warning("Imou fetch failed: %s", e)
        return []


def get_all_cameras(force: bool = False, user_email: str | None = None) -> list[dict]:
    """
    Return all available cameras: webcam + Imou cloud + local IP cameras.
    Imou cameras are cached for 60 seconds to avoid re-authenticating on each request.
    """
    global _camera_cache, _camera_cache_ts

    with _camera_lock:
        if force or (time.time() - _camera_cache_ts) >= 60 or not _camera_cache:
            base_cams = [{"id": "webcam", "name": "Webcam (Built-in)", "status": "≡ƒƒó Online", "type": "webcam"}]
            # Imou cloud fetching disabled per user request
            # base_cams.extend(_fetch_imou_cameras())
            _camera_cache    = base_cams
            _camera_cache_ts = time.time()
        cameras = list(_camera_cache)

    # Append user-specific local cameras from database
    if user_email:
        try:
            conn = get_db_connection()
            cur  = conn.cursor()
            cur.execute(
                "SELECT id, name, brand, ip_address, roles, gate_id, transport FROM local_cameras WHERE owner_email = %s",
                (user_email,),
            )
            import json
            for row in cur.fetchall():
                # Parse roles (Handle both JSONB list and string)
                r_list = row.get("roles", [])
                if isinstance(r_list, str):
                    try: r_list = json.loads(r_list)
                    except: r_list = []
                if not isinstance(r_list, list):
                    r_list = []

                cameras.append({
                    "id":     f"local_{row['id']}",
                    "name":   row["name"],
                    "status": "Online",
                    "type":   "local",
                    "brand":  row["brand"],
                    "ip":     row["ip_address"],
                    "roles":  r_list,
                    "gate_id": row.get("gate_id"),
                    "transport": row.get("transport", "tcp")
                })
            
            # Also fetch metadata for hardcoded cameras (like webcam)
            cur.execute("SELECT camera_id, roles, gate_id FROM camera_metadata")
            meta_map = {r["camera_id"]: {"roles": r["roles"], "gate": r["gate_id"]} for r in cur.fetchall()}
            for cam in cameras:
                if cam["id"] in meta_map:
                    m_data = meta_map[cam["id"]]
                    r_list = m_data["roles"]
                    if isinstance(r_list, str):
                        try: r_list = json.loads(r_list)
                        except: r_list = []
                    if isinstance(r_list, list):
                        cam["roles"] = r_list
                    cam["gate_id"] = m_data["gate"]

            cur.close()
            conn.close()
        except Exception as e:
            log.error("Error fetching local cameras: %s", e)

    return cameras


# ΓöÇΓöÇΓöÇ Routes ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

@camera_bp.route("/api/cameras")
@login_required
def api_cameras():
    user_email = session.get("user")
    refresh = request.args.get("refresh") == "1"
    cameras = get_all_cameras(user_email=user_email, force=refresh)
    return jsonify(cameras)


@camera_bp.route("/api/monitoring/start_all", methods=["POST"])
@login_required
def api_start_all_monitoring():
    """Trigger background monitoring for all local cameras."""
    try:
        count = start_all_monitoring()
        return jsonify({"success": True, "count": count})
    except Exception as e:
        log.error("Start all monitoring error: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


@camera_bp.route("/api/cameras/<cam_id>/roles", methods=["POST"])
@login_required
def api_update_camera_roles(cam_id: str):
    """Save role assignments (Entry/Exit/General) for any camera."""
    data = request.get_json(force=True)
    roles = data.get("roles", [])
    gate_id = data.get("gate_id")
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        roles_json = __import__("json").dumps(roles)
        
        if cam_id.startswith("local_"):
            db_id = cam_id.replace("local_", "")
            cur.execute(
                "UPDATE local_cameras SET roles = %s, gate_id = %s WHERE id = %s",
                (roles_json, gate_id, db_id)
            )
        else:
            # For non-local (webcam/imou), use camera_metadata table
            cur.execute("""
                INSERT INTO camera_metadata (camera_id, roles, gate_id)
                VALUES (%s, %s, %s)
                ON CONFLICT (camera_id) DO UPDATE SET roles = EXCLUDED.roles, gate_id = EXCLUDED.gate_id
            """, (cam_id, roles_json, gate_id))
            
        conn.commit()
        cur.close()
        conn.close()
        
        # Ensure we clear any caches or notify the manager if needed
        # (Though monitoring nodes usually poll or get updated on next restart)
        global _camera_cache_ts
        _camera_cache_ts = 0
        
        return jsonify({"success": True, "roles": roles})
    except Exception as e:
        log.error("Error updating camera roles: %s", e)
        return jsonify({"error": str(e)}), 500


def start_all_monitoring():
    """Initializes monitoring for all cameras currently in the database."""
    try:
        from flask import session
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute("SELECT id, name, ip_address, port, username, password, stream_path, transport, roles, gate_id FROM local_cameras")
        rows = cur.fetchall()
        cur.close()
        conn.close()

        active_names = [r[1] for r in rows]
        # Prune ghosts: Stop any monitor NOT in the DB
        if cam_mgr:
            current_nodes = list(cam_mgr._monitoring_nodes.keys()) if hasattr(cam_mgr, '_monitoring_nodes') else []
            for name in current_nodes:
                if name not in active_names and name != "Webcam (Built-in)":
                   log.warning("AI: Pruning orphan ghost monitor: %s", name)
                   cam_mgr.stop_monitoring(name)

        for r in rows:
            cam_id, name, ip, port, user, pw, path, transport, roles, gate_id = r
            rtsp = _build_rtsp_url(ip, port, user, pw, path)
            log.info("AI: Auto-starting monitor for %s (%s)", name, ip)
            cam_mgr.start_monitoring(rtsp, name, roles or ["general"], camera_id=str(cam_id), gate_id=gate_id, transport=transport)
    except Exception as e:
        log.error("AI: Failed to start all monitoring: %s", e)


@camera_bp.route("/api/select", methods=["POST"])
def api_select():
    data   = request.get_json(force=True)
    cam_id = data.get("id", "webcam")
    user_email = session.get("user")
    cameras = get_all_cameras(user_email=user_email)
    cam = next((c for c in cameras if c["id"] == cam_id), None)

    if cam is None:
        return jsonify({"error": "Camera not found"}), 404

    if cam["type"] == "webcam":
        cam_mgr.start(0, cam["name"], roles=cam.get("roles", []), gate_id=cam.get("gate_id"))

    elif cam["type"] == "imou":
        try:
            from app.services.ai.imou_connector import ImouAPI
            base       = cam["base"]
            api        = ImouAPI(APP_ID, APP_SECRET, base)
            api.get_token()
            stream_url = api.get_rtsp(cam["dev_id"])
            if not stream_url:
                return jsonify({"error": "Could not get stream URL from Imou"}), 500
            cam_mgr.start(stream_url, cam["name"])
        except Exception as e:
            log.error("Imou start error: %s", e)
            return jsonify({"error": str(e)}), 500

    elif cam["type"] == "local":
        try:
            db_id = cam_id.replace("local_", "")
            conn  = get_db_connection()
            cur   = conn.cursor()
            cur.execute(
                "SELECT ip_address, port, username, password, stream_path, transport FROM local_cameras WHERE id = %s",
                (db_id,),
            )
            row = cur.fetchone()
            cur.close()
            conn.close()

            if not row:
                return jsonify({"error": "Local camera record not found"}), 404

            rtsp_url = _build_rtsp_url(row["ip_address"], row["port"], row["username"], row["password"], row["stream_path"])
            log.info("Starting local RTSP: %s (Transport: %s)", rtsp_url, row.get("transport"))
            cam_mgr.start(rtsp_url, cam["name"], roles=cam.get("roles", []), gate_id=cam.get("gate_id"), transport=row.get("transport", "tcp"))
        except Exception as e:
            log.error("Local camera start error: %s", e)
            return jsonify({"error": str(e)}), 500

    return jsonify({"success": True, "camera": cam["name"]})


@camera_bp.route("/api/stop", methods=["POST"])
def api_stop():
    cam_mgr.stop_stream()
    return jsonify({"success": True})


@camera_bp.route("/api/stream_id")
def api_stream_id():
    """Returns current stream ID so the frontend can detect camera switches."""
    return jsonify({"stream_id": cam_mgr.stream_id, "active": cam_mgr._active})


@camera_bp.route("/video_feed")
def video_feed():
    """MJPEG live stream endpoint — one frame per ~33ms (Main View)."""
    def generate():
        while True:
            frame = cam_mgr.get_jpeg()
            if frame:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.033)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@camera_bp.route("/api/camera_preview_stream/<cam_id>")
@login_required
def camera_preview_stream(cam_id: str):
    """Specific MJPEG preview stream for the dashboard grid (Low FPS)."""
    def generate():
        while True:
            # Use the existing background accessor for specific cameras
            frame_bytes = cam_mgr.get_background_jpeg(cam_id)
            if frame_bytes:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                time.sleep(0.5) # Optimized for dashboard performance
            else:
                # If specific monitor not found, tiny sleep and retry
                time.sleep(1.0)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ΓöÇΓöÇΓöÇ Local camera CRUD ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

@camera_bp.route("/api/local_cameras", methods=["GET", "POST"])
@login_required
def api_local_cameras():
    email = session.get("user")

    if request.method == "POST":
        data = request.get_json(force=True)
        name = data.get("name")
        brand = data.get("brand", "Generic")
        ip    = data.get("ip")
        port  = data.get("port", 554)
        user  = data.get("username", "")
        pw    = data.get("password", "")
        path  = data.get("path", "")
        transport = data.get("transport", "tcp")

        if not name or not ip:
            return jsonify({"success": False, "error": "Name and IP are required"}), 400

        try:
            conn = get_db_connection()
            cur  = conn.cursor()
            cur.execute(
                """INSERT INTO local_cameras
                   (name, brand, ip_address, port, username, password, stream_path, owner_email, transport)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (name, brand, ip, port, user, pw, path, email, transport),
            )
            conn.commit()
            cur.close()
            conn.close()
            return jsonify({"success": True})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    # GET ΓÇô list cameras for this user
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute(
            "SELECT id, name, brand, ip_address, port, username, password, stream_path, transport "
            "FROM local_cameras WHERE owner_email = %s",
            (email,),
        )
        cameras = [
            {
                "id": r["id"], "name": r["name"], "brand": r["brand"],
                "ip": r["ip_address"], "port": r["port"],
                "username": r["username"], "password": r["password"],
                "path": r["stream_path"], "transport": r.get("transport", "tcp")
            }
            for r in cur.fetchall()
        ]
        cur.close()
        conn.close()
        return jsonify(cameras)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@camera_bp.route("/api/local_cameras/detected", methods=["GET"])
@login_required
def api_detected_local_cameras():
    force_refresh = request.args.get("refresh", "0") == "1"
    try:
        return jsonify(detect_cameras(force=force_refresh))
    except Exception as e:
        log.error("Camera auto-detection failed: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


@camera_bp.route("/api/local_cameras/test", methods=["POST"])
@login_required
def api_test_local_camera():
    data = request.get_json(force=True)
    ip    = data.get("ip")
    port  = data.get("port", 554)
    user  = data.get("username", "")
    pw    = data.get("password", "")
    path  = data.get("path", "")
    transport = data.get("transport", "tcp")

    if not ip:
        return jsonify({"success": False, "error": "IP Address is required"}), 400

    rtsp_url = _build_rtsp_url(ip=ip, port=port, user=user, password=pw, path=path)
    
    # Senior Fix: Stop any existing background monitor for this IP to avoid connection limits/conflicts
    if cam_mgr:
        cam_mgr.stop_monitoring_by_ip(ip)

    log.info("Testing camera connection: %s (Transport: %s)", rtsp_url, transport)
    ok, frame = _test_rtsp_connection(rtsp_url, transport=transport)
    if not ok:
        return jsonify({"success": False, "error": "Connection Failed"})

    token = _issue_preview_token(rtsp_url=rtsp_url, owner_email=session.get("user"))

    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        return jsonify({"success": False, "error": "Connection Failed"})

    base64_img = base64.b64encode(buffer).decode("utf-8")
    return jsonify({
        "success": True, 
        "message": "Connection Successful!",
        "snapshot": f"data:image/jpeg;base64,{base64_img}",
        "preview_token": token,
        "preview_url": f"/api/local_cameras/test/stream?token={token}",
        "stream_url": f"/camera_test_feed?token={token}",
    })


@camera_bp.route("/api/test_camera", methods=["POST"])
@login_required
def api_test_camera():
    data = request.get_json(force=True)
    ip = (data.get("ip") or "").strip()
    port = data.get("port", 554)
    user = data.get("username", "")
    pw = data.get("password", "")
    path = data.get("path", "")
    transport = data.get("transport", "tcp")

    if not ip:
        return jsonify({"success": False, "error": "IP Address is required"}), 400

    rtsp_url = _build_rtsp_url(ip=ip, port=port, user=user, password=pw, path=path)
    
    # Senior Fix: Stop any existing background monitor for this IP to avoid connection limits/conflicts
    if cam_mgr:
        cam_mgr.stop_monitoring_by_ip(ip)

    ok, _ = _test_rtsp_connection(rtsp_url, transport=transport)
    if not ok:
        return jsonify({"success": False, "error": "Connection Failed"})

    token = _issue_preview_token(rtsp_url=rtsp_url, owner_email=session.get("user"))
    return jsonify({
        "success": True,
        "stream_url": f"/camera_test_feed?token={token}",
    })


@camera_bp.route("/api/local_cameras/test/stream", methods=["GET"])
@login_required
def api_test_local_camera_stream():
    token = (request.args.get("token") or "").strip()
    if not token:
        return Response("Connection Failed", status=400, mimetype="text/plain")

    with _preview_lock:
        _cleanup_expired_preview_tokens()
        token_data = _preview_tokens.get(token)
        current_user = session.get("user")
        if not token_data or token_data.get("owner_email") != current_user:
            return Response("Connection Failed", status=404, mimetype="text/plain")
        rtsp_url = token_data["rtsp_url"]

    return Response(_generate_mjpeg(rtsp_url, max_duration_sec=10.0), mimetype="multipart/x-mixed-replace; boundary=frame")


@camera_bp.route("/camera_test_feed", methods=["GET"])
@login_required
def camera_test_feed():
    token = (request.args.get("token") or "").strip()
    if not token:
        return Response("Connection Failed", status=400, mimetype="text/plain")
    with _preview_lock:
        _cleanup_expired_preview_tokens()
        token_data = _preview_tokens.get(token)
        current_user = session.get("user")
        if not token_data or token_data.get("owner_email") != current_user:
            return Response("Connection Failed", status=404, mimetype="text/plain")
        rtsp_url = token_data["rtsp_url"]

    return Response(_generate_mjpeg(rtsp_url, max_duration_sec=10.0), mimetype="multipart/x-mixed-replace; boundary=frame")


@camera_bp.route("/api/local_cameras/detected/metadata", methods=["POST"])
@login_required
def api_detected_local_camera_metadata():
    data = request.get_json(force=True)
    ip = (data.get("ip") or "").strip()
    port = int(data.get("port") or 80)
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    if not ip:
        return jsonify({"success": False, "error": "ip is required"}), 400
    if not username or not password:
        return jsonify({"success": False, "error": "username and password are required"}), 400

    result = fetch_onvif_device_info(ip=ip, port=port, username=username, password=password)
    return jsonify(result)


@camera_bp.route("/api/local_cameras/<int:cam_id>", methods=["DELETE", "PUT", "POST"])
@login_required
def api_local_camera_item(cam_id: int):
    email = session.get("user")
    try:
        if request.method == "DELETE":
            conn = get_db_connection()
            cur  = conn.cursor()
            
            # Senior Fix: Stop background AI monitoring before deleting from DB
            cur.execute("SELECT name FROM local_cameras WHERE id = %s", (cam_id,))
            row = cur.fetchone()
            if row:
                cam_name = row[0]
                log.info("AI: Stopping monitor for '%s' before database removal", cam_name)
                cam_mgr.stop_monitoring(cam_name)

            cur.execute(
                "DELETE FROM local_cameras WHERE id = %s AND owner_email = %s",
                (cam_id, email),
            )
            conn.commit()
            cur.close()
            conn.close()
            return jsonify({"success": True})

        data = request.get_json(force=True)
        name = data.get("name")
        brand = data.get("brand", "Generic")
        ip = data.get("ip")
        port = data.get("port", 554)
        user = data.get("username", "")
        pw = data.get("password", "")
        path = data.get("path", "")
        transport = data.get("transport", "tcp")

        if not name or not ip:
            return jsonify({"success": False, "error": "Name and IP are required"}), 400

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """UPDATE local_cameras
               SET name = %s,
                   brand = %s,
                   ip_address = %s,
                   port = %s,
                   username = %s,
                   password = %s,
                   stream_path = %s,
                   transport = %s
               WHERE id = %s AND owner_email = %s""",
            (name, brand, ip, port, user, pw, path, transport, cam_id, email),
        )
        conn.commit()
        updated = cur.rowcount
        cur.close()
        conn.close()

        if updated == 0:
            return jsonify({"success": False, "error": "Camera not found"}), 404
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# [REMOVED] Auto-start monitoring at top-level. 
# This is now handled asynchronously in run.py to prevent blocking the UI.

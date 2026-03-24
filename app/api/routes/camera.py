"""
app/routes/camera.py  –  Camera Management Routes Blueprint
──────────────────────────────────────────────────────────────────────────────
Handles camera streaming, selection, and MJPEG video feed.

Routes:
  GET  /api/cameras            → JSON list of all cameras
  POST /api/select             → Start streaming a selected camera
  POST /api/stop               → Stop current camera stream
  GET  /api/stream_id          → Returns current stream ID for frontend polling
  GET  /video_feed             → MJPEG live stream endpoint
  GET  /api/camera_preview_stream/<id> → Live preview for dashboard/grid
  GET  /api/local_cameras      → List user's local IP cameras
  GET  /api/local_cameras/detected → Auto-detect local/LAN cameras
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

# ── Imou credentials (loaded from environment) ────────────────────────────────
APP_ID     = os.environ.get("IMOU_APP_ID", "")
APP_SECRET = os.environ.get("IMOU_APP_SECRET", "")


# ── Camera cache ─────────────────────────────────────────────────────────────
_camera_cache: list[dict] = []
_camera_cache_ts: float   = 0.0
_camera_lock = threading.Lock()

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
        encoded_user = quote(user)
        encoded_pass = quote(password)
        return f"rtsp://{encoded_user}:{encoded_pass}@{ip}:{port}{normalized_path}"
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


def _test_rtsp_connection(rtsp_url: str) -> tuple[bool, object | None]:
    # Set FFMPEG open timeout to 5 seconds before VideoCapture
    cap = cv2.VideoCapture()
    cap.setExceptionMode(False)
    # CAP_PROP_OPEN_TIMEOUT_MSEC and READ_TIMEOUT_MSEC require OpenCV 4.x+
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
    # Using CAP_PROP_READ_TIMEOUT_MSEC for frame reading timeout
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
    cap.open(str(rtsp_url), cv2.CAP_FFMPEG)

    if not cap.isOpened():
        cap.release()
        return False, None

    ok, frame = False, None
    # Try for up to ~5 seconds — Dahua/Hikvision can take 3-5s to open
    for _ in range(15):
        ok, frame = cap.read()
        if ok and frame is not None:
            break
        time.sleep(0.3)
    cap.release()
    if not ok or frame is None:
        return False, None
    return True, frame


def _generate_mjpeg(rtsp_url: str, max_duration_sec: float | None = None):
    # Ensure source is string or 0 for webcam
    source = rtsp_url
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        return

    deadline = (time.time() + max_duration_sec) if max_duration_sec else None
    try:
        while True:
            if deadline is not None and time.time() >= deadline:
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            encoded, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not encoded:
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
            time.sleep(0.04)
    finally:
        cap.release()


def get_all_cameras(force: bool = False, user_email: str | None = None) -> list[dict]:
    """
    Return all available cameras: webcam + local IP cameras.
    """
    global _camera_cache, _camera_cache_ts

    with _camera_lock:
        if force or (time.time() - _camera_cache_ts) >= 60 or not _camera_cache:
            base_cams = [{"id": "webcam", "name": "Webcam (Built-in)", "status": "🟢 Online", "type": "webcam", "roles": []}]
            
            # Fetch roles for base_cams from camera_metadata
            try:
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute("SELECT camera_id, roles FROM camera_metadata")
                meta_map = {row["camera_id"]: row["roles"] for row in cur.fetchall()}
                cur.close()
                conn.close()
                for c in base_cams:
                    c["roles"] = meta_map.get(c["id"], [])
            except Exception as e:
                log.error("Error fetching camera_metadata for base: %s", e)

            _camera_cache    = base_cams
            _camera_cache_ts = time.time()
        cameras = [dict(c) for c in _camera_cache] # Deep copy dicts

    # Append user-specific local cameras from database
    if user_email:
        try:
            conn = get_db_connection()
            cur  = conn.cursor()
            cur.execute(
                "SELECT id, name, brand, ip_address, roles FROM local_cameras WHERE owner_email = %s",
                (user_email,),
            )
            for row in cur.fetchall():
                cameras.append({
                    "id":     f"local_{row['id']}",
                    "name":   row["name"],
                    "status": "🟢 Online",
                    "type":   "local",
                    "brand":  row["brand"],
                    "ip":     row["ip_address"],
                    "roles":   row["roles"] if row["roles"] else [],
                })
            cur.close()
            conn.close()
        except Exception as e:
            log.error("Error fetching local cameras: %s", e)

    return cameras


# ─── Routes ──────────────────────────────────────────────────────────────────

@camera_bp.route("/api/cameras")
def api_cameras():
    """Get all cameras (cached or refreshed if ?refresh=1)."""
    user_email = session.get("user")
    force_refresh = request.args.get("refresh", "0") == "1"
    cameras = get_all_cameras(force=force_refresh, user_email=user_email)
    return jsonify(cameras)


@camera_bp.route("/api/cameras/<cam_id>/roles", methods=["POST"])
def api_camera_roles(cam_id):
    """Update roles for a specific camera (JSONB)."""
    try:
        data = request.get_json(force=True)
        roles = data.get("roles", [])
        if not isinstance(roles, list):
            return jsonify({"error": "Roles must be a list"}), 400

        import json
        roles_json = json.dumps(roles)
        conn = get_db_connection()
        cur = conn.cursor()

        if cam_id.startswith("local_"):
            db_id = cam_id.replace("local_", "")
            cur.execute("UPDATE local_cameras SET roles = %s WHERE id = %s", (roles_json, db_id))
        else:
            cur.execute("""
                INSERT INTO camera_metadata (camera_id, roles) 
                VALUES (%s, %s) 
                ON CONFLICT (camera_id) DO UPDATE SET roles = EXCLUDED.roles
            """, (cam_id, roles_json))
        
        conn.commit()
        cur.close()
        conn.close()
        global _camera_cache_ts
        _camera_cache_ts = 0
        return jsonify({"success": True, "roles": roles})
    except Exception as e:
        log.error("Error updating camera roles: %s", e)
        return jsonify({"error": str(e)}), 500


@camera_bp.route("/api/select", methods=["POST"])
def api_select():
    data   = request.get_json(force=True)
    cam_id = data.get("id", "webcam")
    user_email = session.get("user")
    cameras = get_all_cameras(user_email=user_email)
    cam = next((c for c in cameras if str(c["id"]) == str(cam_id)), None)

    if cam is None:
        return jsonify({"error": "Camera not found"}), 404

    if cam["type"] == "webcam":
        cam_mgr.start(0, cam["name"], roles=cam.get("roles", []))
    elif cam["type"] == "local":
        try:
            db_id = str(cam_id).replace("local_", "")
            conn  = get_db_connection()
            cur   = conn.cursor()
            cur.execute(
                "SELECT ip_address, port, username, password, stream_path FROM local_cameras WHERE id = %s",
                (db_id,),
            )
            row = cur.fetchone()
            cur.close()
            conn.close()
            if not row:
                return jsonify({"error": "Local camera record not found"}), 404
            rtsp_url = _build_rtsp_url(row["ip_address"], row["port"], row["username"], row["password"], row["stream_path"])
            cam_mgr.start(rtsp_url, cam["name"], roles=cam.get("roles", []))
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
    return jsonify({"stream_id": cam_mgr.stream_id, "active": cam_mgr._active})


@camera_bp.route("/video_feed")
def video_feed():
    def generate():
        while True:
            frame = cam_mgr.get_jpeg()
            if frame:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.04)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@camera_bp.route("/api/camera_preview_stream/<cam_id>")
@login_required
def api_camera_preview_stream(cam_id):
    """Serve a low-fps live MJPEG preview for any specific camera."""
    user_email = session.get("user")
    cameras = get_all_cameras(user_email=user_email)
    cam = next((c for c in cameras if str(c["id"]) == str(cam_id)), None)
    
    if not cam:
        return Response("Camera not found", status=404)

    source = None
    if cam["type"] == "webcam":
        source = 0
    elif cam["type"] == "local":
        try:
            db_id = str(cam_id).replace("local_", "")
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT ip_address, port, username, password, stream_path FROM local_cameras WHERE id = %s", (db_id,))
            row = cur.fetchone()
            cur.close()
            conn.close()
            if row:
                source = _build_rtsp_url(row["ip_address"], row["port"], row["username"], row["password"], row["stream_path"])
        except Exception as e:
            log.error("Preview DB error: %s", e)
            return Response("DB error", status=500)

    if source is None:
        return Response("Could not resolve stream source", status=400)

    return Response(_generate_mjpeg(source, max_duration_sec=60.0), mimetype="multipart/x-mixed-replace; boundary=frame")


# ─── Local camera CRUD & Testing ──────────────────────────────────────────────

@camera_bp.route("/api/local_cameras", methods=["GET", "POST"])
@login_required
def api_local_cameras():
    email = session.get("user")
    if request.method == "POST":
        data = request.get_json(force=True)
        try:
            conn = get_db_connection()
            cur  = conn.cursor()
            cur.execute(
                """INSERT INTO local_cameras
                   (name, brand, ip_address, port, username, password, stream_path, owner_email)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (data.get("name"), data.get("brand", "Generic"), data.get("ip"), data.get("port", 554), 
                 data.get("username", ""), data.get("password", ""), data.get("path", ""), email),
            )
            conn.commit()
            cur.close()
            conn.close()
            return jsonify({"success": True})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute("SELECT id, name, brand, ip_address, port, username, password, stream_path FROM local_cameras WHERE owner_email = %s", (email,))
        cameras = [dict(r) for r in cur.fetchall()]
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
        return jsonify({"success": False, "error": str(e)}), 500


@camera_bp.route("/api/local_cameras/test", methods=["POST"])
@login_required
def api_test_local_camera():
    data = request.get_json(force=True)
    rtsp_url = _build_rtsp_url(data.get("ip"), data.get("port", 554), data.get("username", ""), data.get("password", ""), data.get("path", ""))
    ok, frame = _test_rtsp_connection(rtsp_url)
    if not ok:
        return jsonify({"success": False, "error": "Connection Failed"})
    token = _issue_preview_token(rtsp_url=rtsp_url, owner_email=session.get("user"))
    ok, buffer = cv2.imencode(".jpg", frame)
    base64_img = base64.b64encode(buffer).decode("utf-8") if ok else ""
    return jsonify({
        "success": True, 
        "snapshot": f"data:image/jpeg;base64,{base64_img}",
        "stream_url": f"/camera_test_feed?token={token}"
    })


@camera_bp.route("/camera_test_feed", methods=["GET"])
@login_required
def camera_test_feed():
    token = (request.args.get("token") or "").strip()
    with _preview_lock:
        _cleanup_expired_preview_tokens()
        tdata = _preview_tokens.get(token)
        if not tdata or tdata.get("owner_email") != session.get("user"):
            return Response("Invalid token", status=403)
        url = tdata.get("rtsp_url")
    return Response(_generate_mjpeg(url, max_duration_sec=15.0), mimetype="multipart/x-mixed-replace; boundary=frame")


@camera_bp.route("/api/local_cameras/<int:cam_id>", methods=["DELETE", "POST"])
@login_required
def api_local_camera_item(cam_id: int):
    email = session.get("user")
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        if request.method == "DELETE":
            cur.execute("DELETE FROM local_cameras WHERE id = %s AND owner_email = %s", (cam_id, email))
        else:
            data = request.get_json(force=True)
            cur.execute(
                "UPDATE local_cameras SET name=%s, brand=%s, ip_address=%s, port=%s, username=%s, password=%s, stream_path=%s WHERE id=%s AND owner_email=%s",
                (data.get("name"), data.get("brand"), data.get("ip"), data.get("port"), data.get("username"), data.get("password"), data.get("path"), cam_id, email)
            )
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

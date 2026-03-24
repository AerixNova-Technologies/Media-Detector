"""
app/extensions.py  –  Shared Application-Level Singletons
──────────────────────────────────────────────────────────────────────────────
Creates global objects that are shared across the application:
  • cam_mgr  – CameraManager (holds the AI pipeline + render thread)
  • notifier – TelegramNotifier (for Telegram alerts)
  • AI_TOGGLES – runtime on/off switches for AI features

These are imported from blueprints rather than created inside the Flask factory
so they survive across requests without being re-created on app context teardown.
"""

from __future__ import annotations

import logging

log = logging.getLogger("extensions")

from app.services.sse_service import sse_manager

# ── AI Feature Toggles (runtime flags, mutated via /api/toggles) ─────────────
AI_TOGGLES: dict[str, bool] = {
    "person":  True,
    "action":  False,
    "emotion": False,
}

# ── Lazy-initialized singletons ──────────────────────────────────────────────
# We use module-level __getattr__ (Python 3.7+) to load models ONLY when accessed.
# This makes internal scripts and the Flask reloader start instantly.

_cam_mgr  = None
_notifier = None

def _make_camera_manager():
    from app.services.camera_service import CameraManager
    log.info("Initializing Camera Manager and AI Pipeline …")
    mgr = CameraManager()
    log.info("Camera Manager ready.")
    return mgr

def _make_notifier():
    from app.services.ai.notifier import TelegramNotifier
    return TelegramNotifier()

def __getattr__(name):
    global _cam_mgr, _notifier
    if name == "cam_mgr":
        if _cam_mgr is None:
            _cam_mgr = _make_camera_manager()
        return _cam_mgr
    if name == "notifier":
        if _notifier is None:
            _notifier = _make_notifier()
        return _notifier
    raise AttributeError(f"module {__name__} has no attribute {name}")



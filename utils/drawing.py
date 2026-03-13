"""
utils/drawing.py
─────────────────────────────────────────────────────────────────────────────
Overlay rendering helpers: bounding boxes, labels, status banners.
"""

from __future__ import annotations

import cv2
import numpy as np

# Colour palette – Reference Video Style (Red Boxes)
BOX_COLOR: tuple = (0, 0, 255)   # Red
ID_COLOR:  tuple = (0, 255, 0)   # Green
TEXT_COLOR: tuple = (255, 255, 255) # White

# Colour palette – one colour per track ID (cycles)
_PALETTE = [
    (255,  80,  80),
    ( 80, 200, 255),
    ( 80, 255, 120),
    (255, 200,  80),
    (200,  80, 255),
    (255, 255,  80),
    ( 80, 255, 255),
    (255, 150, 150),
]


def _track_color(track_id: int) -> tuple[int, int, int]:
    return _PALETTE[track_id % len(_PALETTE)]


# ---------------------------------------------------------------------------

def draw_person(
    frame: np.ndarray,
    bbox: list[float],
    track_id: int,
    emotion: str = "",
    action: str = "",
    identity: str = "",
) -> None:
    """Draw a bounding box and text labels for one tracked person."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    # Main box (Red - Surveillance Style)
    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

    # Labels (Reference Style: Person in Red, ID in Green)
    identity_str = identity if (identity and identity != "Unknown") else "PERSON"
    id_str = f"ID: {track_id}"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.50
    font_thickness = 1
    line_height = 16

    # Label 1: TYPE (Red Background)
    (tw1, th1), _ = cv2.getTextSize(identity_str, font, font_scale, font_thickness)
    cv2.rectangle(frame, (x1, y1 - th1 - 10), (x1 + tw1 + 10, y1), BOX_COLOR, -1)
    cv2.putText(frame, identity_str, (x1 + 5, y1 - 5), font, font_scale, TEXT_COLOR, font_thickness, cv2.LINE_AA)

    # Label 2: ID (Below Type, or adjacent)
    cv2.putText(frame, id_str, (x1 + tw1 + 15, y1 - 5), font, font_scale, ID_COLOR, 2, cv2.LINE_AA)

    # Label 3: Action/Emotion (optional extra line)
    if action or emotion:
        extra = f"{action} {emotion}".strip()
        cv2.putText(frame, extra, (x1, y1 + 15), font, 0.4, ID_COLOR, 1, cv2.LINE_AA)


def draw_face(
    frame: np.ndarray,
    face_bbox: list[float],
    track_id: int,
) -> None:
    """Draw a thin face bounding box."""
    fx1, fy1, fx2, fy2 = (int(v) for v in face_bbox)
    cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), BOX_COLOR, 1)


def draw_status_bar(
    frame: np.ndarray,
    motion: bool,
    n_persons: int,
    fps: float,
    entries: int = 0,
    exits: int = 0,
) -> None:
    """Top-left status overlay."""
    h, w = frame.shape[:2]
    lines = [
        f"FPS: {fps:.1f}",
        f"Motion: {'YES' if motion else 'NO'}",
        f"Persons: {n_persons}",
        f"In: {entries} | Out: {exits}",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fscale = 0.55
    fthick = 1
    lh = 22
    pad = 8
    max_w = max(cv2.getTextSize(l, font, fscale, fthick)[0][0] for l in lines)

    # Solid dark background for status bar
    cv2.rectangle(frame, (0, 0), (max_w + pad * 2, len(lines) * lh + pad * 2), (20, 20, 20), -1)
    cv2.rectangle(frame, (0, 0), (max_w + pad * 2, len(lines) * lh + pad * 2), (60, 60, 60), 1)

    for i, line in enumerate(lines):
        color = (0, 255, 100) if ("YES" in line or "Motion" not in line) else (0, 120, 255)
        if "Motion" in line and not motion:
            color = (180, 180, 180)
        cv2.putText(
            frame, line,
            (pad, pad + (i + 1) * lh - 4),
            font, fscale, color, fthick, cv2.LINE_AA,
        )


def draw_motion_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.15,
) -> None:
    """Tint moving pixels green for debugging."""
    tint = np.zeros_like(frame)
    tint[mask > 0] = (0, 255, 80)
    cv2.addWeighted(tint, alpha, frame, 1 - alpha, 0, frame)


def draw_tripwire(frame: np.ndarray, position: float, direction: str, color_in: tuple, color_out: tuple) -> None:
    """Draw a horizontal or vertical crossing line."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    if direction == "vertical":
        x = int(position * w)
        cv2.line(frame, (x, 0), (x, h), (100, 100, 100), 2, cv2.LINE_AA)
        cv2.putText(frame, "ENTRY >", (x + 10, 30), font, 0.6, color_in, 2)
        cv2.putText(frame, "< EXIT", (x - 80, 30), font, 0.6, color_out, 2)
    else:
        y = int(position * h)
        cv2.line(frame, (0, y), (w, y), (100, 100, 100), 2, cv2.LINE_AA)
        cv2.putText(frame, "ENTRY - DOWN", (10, y + 25), font, 0.6, color_in, 2)
        cv2.putText(frame, "EXIT - UP", (10, y - 10), font, 0.6, color_out, 2)

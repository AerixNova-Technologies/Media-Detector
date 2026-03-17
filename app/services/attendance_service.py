from __future__ import annotations

import logging
from datetime import datetime, date

from app.db.session import get_db_connection
from app.services.telegram_service import send_message
from app.services.telegram_user_model import get_chat_id_by_phone
from app.services.telegram_utils import (
    format_attendance_message,
    is_valid_phone_number,
    normalize_phone_number,
)

log = logging.getLogger("attendance_service")


def log_movement(camera_id: str, image_path: str):
    """Log any motion detected by the camera."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO movement_log(camera_id, image_path)
            VALUES (%s, %s)
        """, (camera_id, image_path))
        conn.commit()
    except Exception as e:
        log.error(f"Failed to log movement: {e}")
    finally:
        conn.close()


def log_person(camera_id: str, person_type: str, staff_id: int | None, image_path: str, confidence: float):
    """Log person detection (staff or unknown)."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO member_timestamp
            (camera_name, person_type, staff_id, entry_image, confidence_score)
            VALUES (%s, %s, %s, %s, %s)
        """, (camera_id, person_type, staff_id, image_path, confidence))
        conn.commit()
    except Exception as e:
        log.error(f"Failed to log person detection: {e}")
    finally:
        conn.close()


def track_staff_attendance(staff_id: int):
    """Manage staff attendance (first-entry/last-exit)."""
    today = date.today()
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        # Check if record for staff_id and current_date exists
        cur.execute("""
            SELECT id FROM attendance
            WHERE staff_id = %s AND attendance_date = %s
        """, (staff_id, today))
        record = cur.fetchone()

        if not record:
            # First time today
            cur.execute("""
                INSERT INTO attendance
                (staff_id, attendance_date, first_entry_time)
                VALUES (%s, %s, NOW())
            """, (staff_id, today))
        else:
            # Exist today, update last_exit_time
            cur.execute("""
                UPDATE attendance
                SET last_exit_time = NOW()
                WHERE staff_id = %s AND attendance_date = %s
            """, (staff_id, today))
        conn.commit()
    except Exception as e:
        log.error(f"Failed to track staff attendance: {e}")
    finally:
        conn.close()


def mark_attendance(employee_name: str, phone_number: str = "", status: str = "", image_path: str = None) -> dict:
    """Legacy bridge for marking attendance. Attempts to match staff name."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        # Try to find staff by name
        cur.execute("SELECT id FROM staff_profiles WHERE name = %s", (employee_name,))
        row = cur.fetchone()
        if row:
            staff_db_id = row['id']
            track_staff_attendance(staff_db_id)
            return {"status": "success", "staff_id": staff_db_id, "name": employee_name}
        else:
            # Log as unknown person detection if no match
            log_person("Telegram/Manual", "unknown", None, image_path or "", 0.0)
            return {"status": "unknown", "name": employee_name}
    except Exception as e:
        log.error(f"Error in legacy mark_attendance: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        conn.close()


def update_attendance_name(attendance_id: int, employee_name: str) -> bool:
    """Legacy bridge for updating attendance name."""
    # Since we use staff_id linked to staff_profiles now, 
    # 'updating name' is basically redundant if staff_id is set.
    # We'll just return True to avoid breaking callers.
    return True

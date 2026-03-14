
from __future__ import annotations
import logging
from app.db.session import get_db_connection

log = logging.getLogger("telegram_user_model")

def get_chat_id_by_phone(phone_number: str) -> int | None:
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT chat_id FROM telegram_users WHERE phone_number = %s", (phone_number,))
        row = cur.fetchone()
        return row['chat_id'] if row else None
    except Exception as e:
        log.error("DB Error in get_chat_id_by_phone: %s", e)
        return None
    finally:
        conn.close()

def register_telegram_user(phone_number: str, chat_id: int):
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO telegram_users (phone_number, chat_id)
            VALUES (%s, %s)
            ON CONFLICT (phone_number) DO UPDATE SET chat_id = EXCLUDED.chat_id
        """, (phone_number, chat_id))
        conn.commit()
    except Exception as e:
        log.error("DB Error in register_telegram_user: %s", e)
    finally:
        conn.close()

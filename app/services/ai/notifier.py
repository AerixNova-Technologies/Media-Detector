import requests
import logging
import time
import os

log = logging.getLogger("notifier")

class TelegramNotifier:
    def __init__(self, token=None, chat_id=None):
        self.token = token
        self.chat_id = chat_id
        self.last_notify_time = {} 
        self.cooldown = 60 

    def _get_active_bots(self):
        """Fetch all active bots from database."""
        from app.db.session import get_db_connection
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT id, bot_name, bot_token, chat_ids FROM telegram_bots WHERE is_active = TRUE")
            rows = cur.fetchall()
            cur.close()
            conn.close()
            return rows
        except Exception as e:
            log.error(f"Error fetching active bots: {e}")
            return []

    def send_message(self, text, track_id=None, cam_name=None, action=None):
        """Send message across all active bots and log to database."""
        active_bots = self._get_active_bots()
        
        # FALLBACK: If no bots in DB, use .env (legacy/emergency support)
        if not active_bots:
            env_token = os.environ.get("TELEGRAM_BOT_TOKEN")
            env_chat_id = os.environ.get("TELEGRAM_CHAT_ID")
            if env_token and env_chat_id:
                active_bots = [{'bot_token': env_token, 'chat_ids': env_chat_id}]
                log.info("AI Notifier: Falling back to .env Telegram configuration.")

        any_success = False
        success_cids = []
        
        if active_bots:
            for bot in active_bots:
                token = bot['bot_token']
                raw_ids = bot['chat_ids']
                chat_ids = [i.strip() for i in raw_ids.split(",") if i.strip()]
                
                updated_ids = list(chat_ids)
                needs_db_sync = False

                for idx, cid in enumerate(chat_ids):
                    url = f"https://api.telegram.org/bot{token}/sendMessage"
                    payload = {
                        "chat_id": cid,
                        "text": text,
                        "parse_mode": "Markdown"
                    }
                    try:
                        resp = requests.post(url, json=payload, timeout=5)
                        data = resp.json()
                        
                        # Handle Migration (Chat moved to supergroup)
                        if resp.status_code == 400 and "migrate_to_chat_id" in data.get("parameters", {}):
                            new_cid = str(data["parameters"]["migrate_to_chat_id"])
                            log.warning(f"Telegram Migration: {cid} -> {new_cid}. Updating database.")
                            updated_ids[idx] = new_cid
                            needs_db_sync = True
                            # Retry with new ID
                            payload["chat_id"] = new_cid
                            resp = requests.post(url, json=payload, timeout=5)
                            data = resp.json()

                        if resp.status_code == 200:
                            log.info(f"Telegram notification sent via bot to {cid}.")
                            any_success = True
                            success_cids.append(cid)
                        else:
                            log.error(f"Telegram error for {cid}: {resp.text}")
                    except Exception as e:
                        log.error(f"Failed to send Telegram to {cid}: {e}")
                
                # AUTO-SYNC MIGRATIONS BACK TO DB
                if needs_db_sync and 'id' in bot:
                    try:
                        from app.db.session import get_db_connection
                        conn = get_db_connection()
                        cur = conn.cursor()
                        cur.execute("UPDATE telegram_bots SET chat_ids = %s WHERE id = %s", (",".join(updated_ids), bot['id']))
                        conn.commit()
                        conn.close()
                        log.info(f"Sync: Updated group ID migration for Bot #{bot['id']}")
                    except Exception as db_e:
                        log.error(f"Failed to sync migration to DB: {db_e}")
        
        # LOG TO DATABASE (Structured Data)
        from app.db.session import get_db_connection
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            status_str = 'sent' if any_success else 'failed'
            
            chat_id_str = ",".join(success_cids) if success_cids else ""
            
            cur.execute("""
                INSERT INTO telegram_alerts (track_id, camera_name, action, message_text, status, chat_id)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (track_id, cam_name, action, text, status_str, chat_id_str))
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            log.error(f"Failed to log Telegram alert to DB: {e}")

        return any_success

    def send_photo(self, photo_path, caption=None, track_id=None, cam_name=None, action=None):
        """Send photo with optional caption across all active bots."""
        if not os.path.exists(photo_path):
            log.warning(f"Telegram Photo: File not found at {photo_path}")
            return False

        active_bots = self._get_active_bots()
        if not active_bots: return False

        any_success = False
        for bot in active_bots:
            token = bot['bot_token']
            chat_ids = [i.strip() for i in bot['chat_ids'].split(",") if i.strip()]
            
            for cid in chat_ids:
                url = f"https://api.telegram.org/bot{token}/sendPhoto"
                try:
                    with open(photo_path, 'rb') as photo:
                        files = {'photo': photo}
                        data = {
                            "chat_id": cid,
                            "caption": caption,
                            "parse_mode": "Markdown"
                        }
                        resp = requests.post(url, data=data, files=files, timeout=10)
                        if resp.status_code == 200:
                            log.info(f"Telegram photo sent to {cid}.")
                            any_success = True
                        else:
                            log.error(f"Telegram photo error for {cid}: {resp.text}")
                except Exception as e:
                    log.error(f"Failed to send Telegram photo to {cid}: {e}")
        return any_success

    def notify_person(self, track_id, cam_name, identity="Unknown", action="", image_path=None):
        now = time.monotonic()
        
        # Identity Awareness: Store the last name we notified for
        last_data = self.last_notify_time.get(track_id, (0, "Unknown"))
        last_time, last_name = last_data
        
        # SMART COOLDOWN: 
        is_same_id = (str(identity) == str(last_name))
        is_cooldown = (now - last_time) < self.cooldown
        
        if is_cooldown and is_same_id:
            return 
            
        log.info(f"Telegram Trigger: Track {track_id} as {identity} (Previous: {last_name})")
        
        is_first_alert = (str(last_name) == "Unknown")
        is_correction = (not is_first_alert and str(identity) != str(last_name))
        
        if is_correction:
            msg = f"🔄 *IDENTITY CONFIRMED*\n\n"
        else:
            msg = f"🚨 *MISSION CONTROL ALERT*\n\n"
            
        if identity and str(identity) != "Unknown":
            msg += f"👤 *Staff Recognized: {identity}*\n"
        else:
            msg += f"👤 *New Person Tracked*\n"
            
        msg += f"📍 *Camera:* {cam_name}\n"
        msg += f"🆔 *Track ID:* {track_id}\n"
        if action:
            msg += f"🏃 *Activity:* {action}\n"
        
        # Determine notification method (Photo vs Message)
        success = False
        if image_path and os.path.exists(image_path):
            success = self.send_photo(image_path, caption=msg, track_id=track_id, cam_name=cam_name, action=action)
        else:
            success = self.send_message(msg, track_id=track_id, cam_name=cam_name, action=action)

        if success:
            self.last_notify_time[track_id] = (now, str(identity))

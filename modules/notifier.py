import requests
import logging
import time
import os
from concurrent.futures import ThreadPoolExecutor

log = logging.getLogger("notifier")

class TelegramNotifier:
    def __init__(self, token=None, chat_id=None):
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
        raw_ids = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
        self.chat_ids = [i.strip() for i in raw_ids.split(",") if i.strip()]
        self.enabled = os.environ.get("ENABLE_TELEGRAM", "False").lower() == "true"
        
        # Cooldown per track ID (60 seconds)
        self.last_notify_time = {} 
        self.cooldown = 60 
        
        # Executor for non-blocking notifications
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tg_notif")

    def send_message(self, text):
        """Asynchronously send message to all chat IDs."""
        if not self.enabled or not self.token or not self.chat_ids:
            return False
            
        # Submit each chat ID to the background pool
        for cid in self.chat_ids:
            self._executor.submit(self._bg_send, cid, text)
        return True

    def _bg_send(self, cid, text):
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": cid,
            "text": text,
            "parse_mode": "Markdown"
        }
        try:
            resp = requests.post(url, json=payload, timeout=8)
            if resp.status_code == 200:
                log.info(f"Telegram notification sent to {cid}.")
            else:
                log.error(f"Telegram error for {cid}: {resp.text}")
        except Exception as e:
            log.error(f"Failed to send Telegram to {cid}: {e}")

    def notify_person(self, track_id, cam_name, action="", cooldown=None):
        now = time.monotonic()
        last_time = self.last_notify_time.get(track_id, 0)
        
        limit = cooldown if cooldown is not None else self.cooldown
        if (now - last_time) < limit:
            return 
            
        msg = f"🚨 *MISSION CONTROL ALERT*\n\n"
        msg += f"👤 *New Person Tracked*\n"
        msg += f"📍 *Camera:* {cam_name}\n"
        msg += f"🆔 *Track ID:* {track_id}\n"
        if action:
            msg += f"🏃 *Activity:* {action}\n"
        
        # Send without blocking
        self.send_message(msg)
        self.last_notify_time[track_id] = now

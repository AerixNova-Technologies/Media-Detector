import queue
import json
import logging
from datetime import datetime

log = logging.getLogger("sse_service")

class SSEService:
    """
    Manages Server-Sent Events (SSE) subscriptions and broadcasting.
    Allows multiple browser clients to receive real-time updates.
    """
    def __init__(self):
        self.listeners = []

    def listen(self):
        """Creates a new queue for a client and yields SSE formatted messages."""
        q = queue.Queue(maxsize=100)
        self.listeners.append(q)
        log.info(f"New SSE listener joined. Total listeners: {len(self.listeners)}")
        
        try:
            while True:
                msg = q.get() # Blocks until a message is sent
                yield msg
        except GeneratorExit:
            # Client disconnected
            self.listeners.remove(q)
            log.info(f"SSE listener disconnected. Total listeners: {len(self.listeners)}")

    def announce(self, data: dict, event_type: str = "update"):
        """Broadcasts a message to all connected listeners."""
        # Standard SSE format: 
        # event: event_type\n
        # data: JSON_STRING\n\n
        
        msg = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
        
        # Remove dead listeners by checking if they are full or closed
        for i in reversed(range(len(self.listeners))):
            try:
                self.listeners[i].put_nowait(msg)
            except queue.Full:
                del self.listeners[i]
                log.warning("Removed full/stalled SSE listener")

sse_manager = SSEService()

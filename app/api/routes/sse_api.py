from flask import Blueprint, Response, stream_with_context
from app.core.extensions import sse_manager
import logging

log = logging.getLogger("sse_api")
sse_bp = Blueprint('sse_bp', __name__)

@sse_bp.route('/api/sse/stream')
def stream():
    """
    SSE Endpoint: Browsers connect here to receive live updates.
    """
    log.info("Client requested SSE stream")
    return Response(
        stream_with_context(sse_manager.listen()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no' # Disable buffering for Nginx if present
        }
    )

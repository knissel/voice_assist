"""
Edge Status Dashboard + UI Gateway.
Serves the built React UI, exposes status APIs, and provides SSE/command endpoints.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import requests
from dotenv import load_dotenv
from flask import (
    Flask,
    Response,
    jsonify,
    render_template_string,
    request,
    send_from_directory,
    stream_with_context,
)

load_dotenv()

app = Flask(__name__)

# Shared status state (updated by main.py event callbacks)
_state = {
    "status": "initializing",
    "last_transcript": None,
    "last_response": None,
    "last_activity": None,
    "remote_url": os.getenv("XTTS_SERVER_URL", "http://localhost:5001"),
    "remote_status": "unknown",
    "remote_latency_ms": None,
    "end_to_final_ms": None,
}

# UI gateway wiring (configured by main.py)
_event_bus = None
_event_bus_callback = None
_command_dispatcher: Optional[Callable[[str, dict[str, Any]], str]] = None

# SSE fan-out state
_sse_clients: list[queue.Queue] = []
_sse_lock = threading.Lock()

# Built frontend location
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIST_DIR = PROJECT_ROOT / "frontend" / "dist"


def update_state(key, value):
    """Update shared status state."""
    _state[key] = value
    if key in ["status", "last_transcript", "last_response"]:
        _state["last_activity"] = time.strftime("%H:%M:%S")


def get_state():
    """Get a shallow copy of current status."""
    return _state.copy()


def _format_sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def publish_event(event_payload: dict[str, Any]) -> None:
    """Broadcast an event payload to all SSE subscribers."""
    with _sse_lock:
        subscribers = list(_sse_clients)

    for client_q in subscribers:
        try:
            client_q.put_nowait(event_payload)
        except queue.Full:
            # Drop oldest client events under pressure; keep server responsive.
            continue


def set_event_bus(event_bus) -> None:
    """Attach EventBus and bridge all events into SSE."""
    global _event_bus, _event_bus_callback

    if event_bus is _event_bus:
        return

    # Unsubscribe old callback if any.
    if _event_bus is not None and _event_bus_callback is not None:
        try:
            _event_bus.unsubscribe("*", _event_bus_callback)
        except Exception:
            pass

    _event_bus = event_bus

    if _event_bus is None:
        _event_bus_callback = None
        return

    def on_event(event):
        publish_event(event.to_dict())

    _event_bus_callback = on_event
    _event_bus.subscribe("*", _event_bus_callback)


def set_command_dispatcher(dispatcher: Callable[[str, dict[str, Any]], str]) -> None:
    """Register a command dispatcher used by /api/ui/command."""
    global _command_dispatcher
    _command_dispatcher = dispatcher


# Fallback HTML used when the React build is not present.
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>Voice Assistant - Edge Dashboard</title>
  <style>
    :root {
      --bg: #101322;
      --card: #1a1f33;
      --text: #f4f7ff;
      --muted: #9aa3bd;
      --accent: #17d4ff;
      --accent2: #ff7a18;
      --ok: #32d583;
      --err: #f97066;
      --bd: #2b3350;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Avenir Next", "Trebuchet MS", sans-serif;
      color: var(--text);
      background:
        radial-gradient(900px 420px at 15% -5%, rgba(23,212,255,0.22), transparent 60%),
        radial-gradient(840px 480px at 85% 8%, rgba(255,122,24,0.18), transparent 60%),
        var(--bg);
      display: grid;
      place-items: center;
      padding: 24px;
    }
    .panel {
      width: min(960px, 100%);
      background: color-mix(in oklab, var(--card), black 5%);
      border: 1px solid var(--bd);
      border-radius: 20px;
      padding: 20px;
      box-shadow: 0 22px 50px rgba(0,0,0,0.35);
    }
    h1 {
      margin: 0 0 8px;
      font-size: 1.35rem;
      letter-spacing: 0.04em;
    }
    p { margin: 0 0 16px; color: var(--muted); }
    .grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }
    .card {
      border: 1px solid var(--bd);
      border-radius: 14px;
      padding: 12px;
      background: rgba(255,255,255,0.02);
    }
    .label { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }
    .value { margin-top: 6px; font-size: 1.05rem; font-weight: 700; }
    .ok { color: var(--ok); }
    .err { color: var(--err); }
    @media (max-width: 860px) {
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <section class=\"panel\">
    <h1>Edge UI Build Not Found</h1>
    <p>The Flask API is running. Build the React frontend with <code>npm run build</code> in <code>services/edge/frontend</code>.</p>
    <div class=\"grid\">
      <article class=\"card\"><div class=\"label\">API</div><div class=\"value ok\">/api/status</div></article>
      <article class=\"card\"><div class=\"label\">Events</div><div class=\"value ok\">/api/ui/events</div></article>
      <article class=\"card\"><div class=\"label\">Commands</div><div class=\"value\">/api/ui/command</div></article>
    </div>
  </section>
</body>
</html>
"""


@app.route("/api/status")
def api_status():
    """Polling status API used by the frontend for health/metrics."""
    try:
        start = time.time()
        resp = requests.get(f"{_state['remote_url']}/health", timeout=1.0)
        latency = int((time.time() - start) * 1000)
        if resp.status_code == 200:
            _state["remote_status"] = "online"
            _state["remote_latency_ms"] = latency
        else:
            _state["remote_status"] = "offline"
            _state["remote_latency_ms"] = None
    except Exception:
        _state["remote_status"] = "offline"
        _state["remote_latency_ms"] = None

    return jsonify(_state)


@app.route("/api/ui/events")
def api_ui_events():
    """Server-Sent Events stream for realtime assistant events."""

    @stream_with_context
    def stream():
        client_q: queue.Queue = queue.Queue(maxsize=300)
        with _sse_lock:
            _sse_clients.append(client_q)

        try:
            yield "retry: 3000\n\n"
            yield _format_sse(
                {
                    "type": "ui_connected",
                    "data": {"message": "event stream connected"},
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
            )

            while True:
                try:
                    payload = client_q.get(timeout=15)
                    yield _format_sse(payload)
                except queue.Empty:
                    yield ": keep-alive\n\n"
        finally:
            with _sse_lock:
                if client_q in _sse_clients:
                    _sse_clients.remove(client_q)

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.route("/api/ui/command", methods=["POST"])
def api_ui_command():
    """Command endpoint used by touch UI actions."""
    payload = request.get_json(silent=True) or {}
    tool = payload.get("tool")
    args = payload.get("args", {})

    if not isinstance(tool, str) or not tool.strip():
        return jsonify({"ok": False, "result": "Field 'tool' is required"}), 400

    if not isinstance(args, dict):
        return jsonify({"ok": False, "result": "Field 'args' must be an object"}), 400

    if _command_dispatcher is None:
        return jsonify({"ok": False, "result": "Command dispatcher unavailable"}), 503

    start = time.time()
    tool = tool.strip()

    # Mirror legacy semantics for UI-originating calls.
    if _event_bus is not None:
        _event_bus.emit(
            "tool_call",
            {
                "tool_name": tool,
                "arguments": args,
                "origin": "ui",
            },
        )
    else:
        publish_event(
            {
                "type": "tool_call",
                "data": {"tool_name": tool, "arguments": args, "origin": "ui"},
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )

    try:
        result = _command_dispatcher(tool, args)
        result_text = str(result)
        success = not (
            result_text.startswith("Unknown tool:")
            or result_text.startswith("Tool execution failed:")
        )
    except Exception as exc:
        success = False
        result_text = f"Tool execution failed: {exc}"

    duration_ms = int((time.time() - start) * 1000)

    if _event_bus is not None:
        _event_bus.emit(
            "tool_result",
            {
                "tool_name": tool,
                "success": success,
                "result": result_text,
                "duration_ms": duration_ms,
                "origin": "ui",
            },
        )
    else:
        publish_event(
            {
                "type": "tool_result",
                "data": {
                    "tool_name": tool,
                    "success": success,
                    "result": result_text,
                    "duration_ms": duration_ms,
                    "origin": "ui",
                },
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )

    return jsonify({"ok": success, "result": result_text})


@app.route("/api/health")
def api_health():
    return jsonify(
        {
            "status": "ok",
            "frontend_dist": FRONTEND_DIST_DIR.exists(),
            "sse_clients": len(_sse_clients),
            "command_ready": _command_dispatcher is not None,
        }
    )


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def dashboard(path: str):
    """
    Serve built React app from frontend/dist. If absent, render fallback HTML.
    """
    if FRONTEND_DIST_DIR.exists() and (FRONTEND_DIST_DIR / "index.html").exists():
        candidate = FRONTEND_DIST_DIR / path
        if path and candidate.exists() and candidate.is_file():
            return send_from_directory(FRONTEND_DIST_DIR, path)

        # SPA fallback
        return send_from_directory(FRONTEND_DIST_DIR, "index.html")

    return render_template_string(DASHBOARD_HTML)


def run_dashboard(host="0.0.0.0", port=5000):
    """Run dashboard/server in Flask dev server (threaded)."""
    import logging

    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)

    app.run(host=host, port=port, threaded=True, use_reloader=False)


def start_dashboard_thread(host="0.0.0.0", port=5000):
    """Start dashboard/server in a background thread."""
    thread = threading.Thread(target=run_dashboard, args=(host, port), daemon=True)
    thread.start()
    print(f"[DASHBOARD] Started at http://{host}:{port}")
    return thread

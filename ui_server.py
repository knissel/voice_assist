#!/usr/bin/env python3
"""
Simple WebSocket server for the Jarvis UI.
Serves the touch-friendly UI and bridges events from the assistant.

Usage:
    python ui_server.py

Then open http://localhost:8765 in a browser (or on the Pi's touchscreen).
"""
import asyncio
import json
import os
import threading
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from functools import partial

try:
    import websockets
except ImportError:
    print("Installing websockets...")
    import subprocess
    subprocess.run(["pip", "install", "websockets"], check=True)
    import websockets

# Get the UI directory
UI_DIR = Path(__file__).parent / "ui"
HTTP_PORT = 8765
WS_PORT = 8766

# Connected WebSocket clients
clients = set()


class UIHTTPHandler(SimpleHTTPRequestHandler):
    """HTTP handler that serves files from the UI directory."""
    
    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=str(UI_DIR), **kwargs)
    
    def log_message(self, format, *args):
        # Suppress HTTP logs
        pass


async def websocket_handler(websocket, path=None):
    """Handle WebSocket connections from UI clients."""
    clients.add(websocket)
    print(f"🖥️  UI client connected ({len(clients)} total)")
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                await handle_client_message(data, websocket)
            except json.JSONDecodeError:
                print(f"Invalid JSON: {message}")
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        clients.discard(websocket)
        print(f"🖥️  UI client disconnected ({len(clients)} total)")


async def handle_client_message(data: dict, websocket):
    """Handle messages from UI clients (button presses, etc.)."""
    msg_type = data.get("type")
    
    if msg_type == "tool_call":
        # Forward tool calls to the assistant (wakeword bridge) for execution
        tool_name = data.get("tool")
        args = data.get("args", {})

        await broadcast_event({
            "type": "tool_call",
            "data": {
                "tool_name": tool_name,
                "arguments": args,
                "origin": "ui"
            }
        })
    
    elif msg_type == "ping":
        await websocket.send(json.dumps({"type": "pong"}))
    
    elif msg_type == "event":
        event_data = data.get("data")
        if isinstance(event_data, dict):
            await broadcast_event(event_data)


async def broadcast_event(event_data: dict):
    """Broadcast an event to all connected UI clients."""
    if clients:
        message = json.dumps(event_data)
        await asyncio.gather(
            *[client.send(message) for client in clients],
            return_exceptions=True
        )


def event_bus_callback(event):
    """Callback for EventBus events - bridges to WebSocket clients."""
    event_data = event.to_dict()
    
    # Schedule broadcast in the asyncio event loop
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(broadcast_event(event_data), loop)
    except RuntimeError:
        pass


def start_http_server():
    """Start the HTTP server for serving UI files."""
    handler = partial(UIHTTPHandler, directory=str(UI_DIR))
    httpd = HTTPServer(("0.0.0.0", HTTP_PORT), handler)
    print(f"🌐 HTTP server running at http://0.0.0.0:{HTTP_PORT}")
    httpd.serve_forever()


async def start_websocket_server():
    """Start the WebSocket server."""
    async with websockets.serve(websocket_handler, "0.0.0.0", WS_PORT):
        print(f"🔌 WebSocket server running at ws://0.0.0.0:{WS_PORT}")
        await asyncio.Future()  # Run forever


def setup_event_bridge():
    """Set up bridge from assistant's EventBus to WebSocket clients."""
    try:
        # Try to import the event bus from wakeword module
        import sys
        if 'wakeword' in sys.modules:
            from wakeword import event_bus
            event_bus.subscribe("*", event_bus_callback)
            print("✅ Event bridge connected to assistant")
            return True
    except Exception as e:
        print(f"⚠️  Could not connect to assistant event bus: {e}")
    
    return False


def main():
    """Main entry point."""
    print("=" * 50)
    print("🤖 Jarvis UI Server")
    print("=" * 50)
    
    # Check if UI files exist
    if not UI_DIR.exists():
        print(f"❌ UI directory not found: {UI_DIR}")
        return
    
    if not (UI_DIR / "index.html").exists():
        print(f"❌ index.html not found in {UI_DIR}")
        return
    
    # Start HTTP server in a thread
    http_thread = threading.Thread(target=start_http_server, daemon=True)
    http_thread.start()
    
    # Try to set up event bridge
    setup_event_bridge()
    
    print()
    print(f"📱 Open in browser: http://localhost:{HTTP_PORT}")
    print(f"📱 On Pi touchscreen: http://127.0.0.1:{HTTP_PORT}")
    print()
    print("Press Ctrl+C to stop")
    print()
    
    # Run WebSocket server
    try:
        asyncio.run(start_websocket_server())
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")


if __name__ == "__main__":
    main()

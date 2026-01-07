"""
FastAPI + WebSocket server for UI clients.
Bridges the EventBus to WebSocket connections for real-time UI updates.

Usage:
    # Start standalone:
    uvicorn ui_gateway.server:app --host 0.0.0.0 --port 8000
    
    # Or integrate with assistant:
    from ui_gateway.server import create_app, setup_event_bridge
    app = create_app()
    setup_event_bridge(event_bus, app.state.connection_manager)
"""
import asyncio
import json
import logging
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections to UI clients."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")
    
    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, message: str) -> None:
        """Send message to all connected clients."""
        async with self._lock:
            connections = list(self.active_connections)
        
        disconnected = []
        for connection in connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.debug(f"Failed to send to client: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            await self.disconnect(conn)
    
    async def send_json(self, data: Dict[str, Any]) -> None:
        """Send JSON data to all connected clients."""
        await self.broadcast(json.dumps(data))
    
    @property
    def connection_count(self) -> int:
        return len(self.active_connections)


# Global connection manager
manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("UI Gateway starting...")
    yield
    logger.info("UI Gateway shutting down...")


def create_app() -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="Voice Assistant UI Gateway",
        description="WebSocket server for real-time UI updates",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Store connection manager in app state
    app.state.connection_manager = manager
    
    # Register routes
    app.include_router(router)
    
    return app


# Router for API endpoints
from fastapi import APIRouter
router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "connections": manager.connection_count
    }


@router.get("/")
async def root():
    """Serve basic info page."""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Voice Assistant UI Gateway</title>
        <style>
            body { font-family: system-ui; max-width: 800px; margin: 50px auto; padding: 20px; }
            .status { padding: 10px; border-radius: 5px; margin: 10px 0; }
            .connected { background: #d4edda; }
            .event { background: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 5px; font-family: monospace; }
        </style>
    </head>
    <body>
        <h1>🎤 Voice Assistant UI Gateway</h1>
        <div id="status" class="status">Connecting...</div>
        <h2>Events</h2>
        <div id="events"></div>
        <script>
            const ws = new WebSocket(`ws://${window.location.host}/ws`);
            const status = document.getElementById('status');
            const events = document.getElementById('events');
            
            ws.onopen = () => {
                status.textContent = '✅ Connected';
                status.className = 'status connected';
            };
            
            ws.onclose = () => {
                status.textContent = '❌ Disconnected';
                status.className = 'status';
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                const div = document.createElement('div');
                div.className = 'event';
                div.textContent = JSON.stringify(data, null, 2);
                events.insertBefore(div, events.firstChild);
                
                // Keep only last 50 events
                while (events.children.length > 50) {
                    events.removeChild(events.lastChild);
                }
            };
        </script>
    </body>
    </html>
    """)


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for UI clients."""
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive messages from UI (e.g., button presses)
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                await handle_ui_message(message, websocket)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from client: {data}")
            
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await manager.disconnect(websocket)


async def handle_ui_message(message: Dict[str, Any], websocket: WebSocket) -> None:
    """Handle incoming messages from UI clients."""
    msg_type = message.get("type")
    
    if msg_type == "ping":
        await websocket.send_text(json.dumps({"type": "pong"}))
    
    elif msg_type == "ui_action":
        # Forward UI actions to the assistant
        action_id = message.get("action_id")
        payload = message.get("payload")
        logger.info(f"UI action: {action_id} with payload: {payload}")
        # TODO: Dispatch to assistant via callback
    
    elif msg_type == "cancel":
        # User wants to cancel current operation
        logger.info("Cancel requested from UI")
        # TODO: Signal cancellation to assistant
    
    else:
        logger.debug(f"Unknown message type: {msg_type}")


def setup_event_bridge(event_bus, connection_manager: ConnectionManager = None):
    """
    Bridge EventBus events to WebSocket clients.
    
    Args:
        event_bus: The assistant's EventBus instance
        connection_manager: Optional custom ConnectionManager
    """
    cm = connection_manager or manager
    
    def on_event(event):
        """Forward event to WebSocket clients."""
        # Run in asyncio event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(cm.broadcast(event.to_json()))
            else:
                loop.run_until_complete(cm.broadcast(event.to_json()))
        except RuntimeError:
            # No event loop, create one
            asyncio.run(cm.broadcast(event.to_json()))
    
    # Subscribe to all events
    event_bus.subscribe("*", on_event)
    logger.info("Event bridge configured")


# Create default app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "ui_gateway.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

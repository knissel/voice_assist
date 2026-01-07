"""
Edge Status Dashboard - Lightweight web UI for monitoring the voice assistant.
"""
import threading
from flask import Flask, jsonify, render_template_string
import time
import requests
import os

app = Flask(__name__)

# Shared state (will be set by main.py)
_state = {
    "status": "initializing",
    "last_transcript": None,
    "last_response": None,
    "last_activity": None,
    "compute_url": os.getenv("COMPUTE_SERVER_URL", "http://localhost:8000"),
    "compute_status": "unknown",
    "compute_latency_ms": None,
}

def update_state(key, value):
    """Update the shared state."""
    _state[key] = value
    if key in ["status", "last_transcript", "last_response"]:
        _state["last_activity"] = time.strftime("%H:%M:%S")

def get_state():
    """Get the current state."""
    return _state.copy()

# HTML Template - Modern dark theme dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Voice Assistant - Edge Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            color: #e8e8e8;
            min-height: 100vh;
            padding: 2rem;
        }
        .container { max-width: 800px; margin: 0 auto; }
        h1 {
            text-align: center;
            margin-bottom: 2rem;
            font-size: 2rem;
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            backdrop-filter: blur(10px);
        }
        .status-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }
        .status-row:last-child { border-bottom: none; }
        .label { color: #888; font-size: 0.9rem; }
        .value { font-weight: 600; font-size: 1.1rem; }
        .status-badge {
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.8rem;
        }
        .status-listening { background: #00ff8840; color: #00ff88; }
        .status-recording { background: #ff880040; color: #ff8800; animation: pulse 1s infinite; }
        .status-processing { background: #00d9ff40; color: #00d9ff; animation: pulse 0.5s infinite; }
        .status-initializing { background: #88888840; color: #888888; }
        .status-online { background: #00ff8840; color: #00ff88; }
        .status-offline { background: #ff444440; color: #ff4444; }
        .status-unknown { background: #88888840; color: #888888; }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .transcript-box {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 8px;
            padding: 1rem;
            margin-top: 0.5rem;
            font-family: monospace;
            min-height: 60px;
            color: #00d9ff;
        }
        .response-box {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 8px;
            padding: 1rem;
            margin-top: 0.5rem;
            min-height: 60px;
            color: #00ff88;
        }
        .section-title {
            font-size: 0.8rem;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 1rem;
        }
        .nodes-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }
        .node-card {
            background: rgba(0, 0, 0, 0.2);
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
        }
        .node-name { font-size: 0.9rem; color: #888; margin-bottom: 0.5rem; }
        .node-status { font-size: 1.2rem; font-weight: bold; }
        .latency { font-size: 0.8rem; color: #666; margin-top: 0.25rem; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Voice Assistant Dashboard</h1>
        
        <div class="card">
            <div class="section-title">System Status</div>
            <div class="nodes-grid">
                <div class="node-card">
                    <div class="node-name">Edge Node</div>
                    <div class="node-status">
                        <span id="edge-status" class="status-badge status-initializing">Initializing</span>
                    </div>
                    <div class="latency" id="edge-time">--</div>
                </div>
                <div class="node-card">
                    <div class="node-name">Compute Node</div>
                    <div class="node-status">
                        <span id="compute-status" class="status-badge status-unknown">Unknown</span>
                    </div>
                    <div class="latency" id="compute-latency">--</div>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="section-title">Last Interaction</div>
            <div class="status-row">
                <span class="label">You said:</span>
            </div>
            <div class="transcript-box" id="transcript">Waiting for input...</div>
            <div class="status-row" style="margin-top: 1rem;">
                <span class="label">Assistant response:</span>
            </div>
            <div class="response-box" id="response">--</div>
        </div>
    </div>

    <script>
        function updateDashboard() {
            fetch('/api/status')
                .then(r => r.json())
                .then(data => {
                    // Edge status
                    const edgeEl = document.getElementById('edge-status');
                    edgeEl.textContent = data.status;
                    edgeEl.className = 'status-badge status-' + data.status.toLowerCase();
                    document.getElementById('edge-time').textContent = 
                        data.last_activity ? 'Last: ' + data.last_activity : '--';

                    // Compute status
                    const computeEl = document.getElementById('compute-status');
                    computeEl.textContent = data.compute_status;
                    computeEl.className = 'status-badge status-' + data.compute_status.toLowerCase();
                    document.getElementById('compute-latency').textContent = 
                        data.compute_latency_ms ? data.compute_latency_ms + 'ms' : '--';

                    // Transcript
                    document.getElementById('transcript').textContent = 
                        data.last_transcript || 'Waiting for input...';
                    document.getElementById('response').textContent = 
                        data.last_response || '--';
                });
        }

        // Update every second
        setInterval(updateDashboard, 1000);
        updateDashboard();
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/status')
def api_status():
    # Check compute node health
    try:
        start = time.time()
        resp = requests.get(f"{_state['compute_url']}/health", timeout=2)
        latency = int((time.time() - start) * 1000)
        if resp.status_code == 200:
            _state["compute_status"] = "online"
            _state["compute_latency_ms"] = latency
        else:
            _state["compute_status"] = "offline"
            _state["compute_latency_ms"] = None
    except Exception:
        _state["compute_status"] = "offline"
        _state["compute_latency_ms"] = None
    
    return jsonify(_state)

def run_dashboard(host="0.0.0.0", port=5000):
    """Run the dashboard in a background thread."""
    app.run(host=host, port=port, threaded=True, use_reloader=False)

def start_dashboard_thread(host="0.0.0.0", port=5000):
    """Start the dashboard in a background thread."""
    thread = threading.Thread(target=run_dashboard, args=(host, port), daemon=True)
    thread.start()
    print(f"[DASHBOARD] Started at http://{host}:{port}")
    return thread

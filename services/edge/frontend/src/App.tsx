import { useEffect, useState } from 'react';
import AudioVisualizer from './components/AudioVisualizer';

// Define the shape of data from dashboard.py
interface SystemState {
  status: string; // 'initializing', 'listening', 'recording', 'processing', 'speaking', etc.
  last_transcript: string | null;
  last_response: string | null;
  last_activity: string | null;
  remote_status: 'online' | 'offline' | 'unknown';
  remote_latency_ms: number | null;
  end_to_final_ms: number | null;
}

function App() {
  const [state, setState] = useState<SystemState>({
    status: 'initializing',
    last_transcript: null,
    last_response: null,
    last_activity: null,
    remote_status: 'unknown',
    remote_latency_ms: null,
    end_to_final_ms: null,
  });

  const [connectionError, setConnectionError] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch('/api/status');
        if (!res.ok) throw new Error('Failed to fetch status');
        const data = await res.json();
        setState(data);
        setConnectionError(false);
      } catch (err) {
        console.error(err);
        setConnectionError(true);
      }
    };

    const interval = setInterval(fetchData, 1000);
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online': return 'dot-online';
      case 'offline': return 'dot-offline';
      case 'recording': return 'dot-working';
      case 'processing': return 'dot-working';
      default: return 'dot-offline';
    }
  };

  return (
    <div className="dashboard-container">
      <header>
        <h1>VOICE ASSIST <span style={{ fontSize: '0.5em', opacity: 0.5, border: '1px solid currentColor', borderRadius: '4px', padding: '2px 6px', verticalAlign: 'middle' }}>EDGE</span></h1>
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
          {connectionError && <span style={{ color: 'var(--accent-danger)', fontSize: '0.9rem' }}>⚠ Dashboard Disconnected</span>}
        </div>
      </header>

      <div className="main-grid">
        {/* Left Sidebar: Status & Config */}
        <div className="status-sidebar">
          {/* Agent Status Card */}
          <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', minHeight: '240px' }}>
            <div className="status-label">CURRENT STATE</div>
            <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', width: '100%' }}>
              <AudioVisualizer state={state.status as any} />
            </div>
            <div className="status-value" style={{ color: 'var(--accent-primary)', textTransform: 'uppercase', letterSpacing: '1px' }}>
              {state.status}
            </div>
          </div>

          {/* System Health */}
          <div className="glass-panel">
            <div className="status-item">
              <span className="status-label">Edge Brain</span>
              <span className="status-value">
                <div className={`status-dot ${getStatusColor(connectionError ? 'offline' : 'online')}`}></div>
                {connectionError ? 'Disconnected' : 'Active'}
              </span>
            </div>
            <div style={{ height: '1rem' }}></div>
            <div className="status-item">
              <span className="status-label">Remote GPU (5090)</span>
              <span className="status-value">
                <div className={`status-dot ${getStatusColor(state.remote_status)}`}></div>
                {state.remote_status === 'online' ? `${state.remote_latency_ms}ms` : 'Offline'}
              </span>
            </div>
          </div>

          {/* Performance Stats */}
          <div className="glass-panel">
            <div className="status-item">
              <span className="status-label">Last Latency</span>
              <span className="status-value" style={{ fontFamily: 'monospace' }}>
                {state.end_to_final_ms ? `${state.end_to_final_ms}ms` : '--'}
              </span>
            </div>
            <div style={{ height: '0.5rem' }}></div>
            <div className="status-item">
              <span className="status-label">Last Activity</span>
              <span className="status-value" style={{ fontSize: '0.9rem' }}>
                {state.last_activity || '--'}
              </span>
            </div>
          </div>
        </div>

        {/* Right Content: Conversation */}
        <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column' }}>
          <div className="status-label" style={{ marginBottom: '1rem' }}>LIVE TRANSCRIPT</div>

          <div className="conversation-feed">
            {/* If we had a list of history, we'd map it. For now, we only show last interaction. */}

            {/* Show previous if available (placeholder logic as backend only gives "last") */}

            {state.last_transcript && (
              <div className="message user">
                {state.last_transcript}
              </div>
            )}

            {(state.last_response || state.status === 'processing' || state.status === 'thinking') && (
              <div className="message assistant">
                {state.status === 'processing' || state.status === 'thinking'
                  ? <span className="status-dot dot-working" style={{ display: 'inline-block', marginRight: '8px' }}></span>
                  : null
                }
                {state.last_response || "..."}
              </div>
            )}

            {!state.last_transcript && !state.last_response && (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                Waiting for voice input...
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;

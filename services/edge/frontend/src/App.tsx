import { useMemo, useState } from 'react';
import { Header } from './components/Header';
import { ConversationPanel } from './components/panels/ConversationPanel';
import { TimersPanel } from './components/panels/TimersPanel';
import { RecipePanel } from './components/panels/RecipePanel';
import { TimerModal } from './components/modals/TimerModal';
import { useWebSocket } from './contexts/WebSocketContext';
import './styles/app.css';

function App() {
  const { assistantState, connectionState, systemState, timers } = useWebSocket();
  const [timerModalOpen, setTimerModalOpen] = useState(false);

  const activeTimerCount = timers.filter((timer) => timer.remaining_seconds > 0).length;

  const assistantStatusLabel = useMemo(() => {
    const mapping: Record<string, string> = {
      initializing: 'Booting',
      idle: 'Ready',
      listening: 'Listening',
      recording: 'Recording',
      transcribing: 'Transcribing',
      processing: 'Processing',
      thinking: 'Thinking',
      thinking_local: 'Local Thinking',
      executing: 'Executing',
      speaking: 'Speaking',
      error: 'Error',
    };
    return mapping[assistantState] ?? assistantState;
  }, [assistantState]);

  return (
    <div className="app-shell">
      <div className="ambient ambient-left" aria-hidden="true"></div>
      <div className="ambient ambient-right" aria-hidden="true"></div>

      <div className="app-frame">
        <Header />

        <main className="app-layout">
          <aside className="left-rail">
            <section className="rail-card">
              <div className="rail-card-title">Quick Actions</div>
              <div className="quick-actions" role="group" aria-label="Quick actions">
                <button type="button" className="quick-action" disabled>
                  <span className="quick-action-icon">💡</span>
                  <span>Lights</span>
                </button>
                <button type="button" className="quick-action" disabled>
                  <span className="quick-action-icon">⏱</span>
                  <span>Set Timer</span>
                </button>
                <button type="button" className="quick-action" disabled>
                  <span className="quick-action-icon">⏹</span>
                  <span>Stop Audio</span>
                </button>
                <button type="button" className="quick-action" disabled>
                  <span className="quick-action-icon">📅</span>
                  <span>Calendar</span>
                </button>
              </div>
              <p className="rail-note">Touch actions are visual-only in this phase.</p>
            </section>

            <section className="rail-card status-card">
              <div className="rail-card-title">System</div>

              <div className="status-row">
                <span>Assistant</span>
                <span className={`status-pill status-${assistantState}`}>{assistantStatusLabel}</span>
              </div>

              <div className="status-row">
                <span>Remote GPU</span>
                <span className={`status-pill status-${systemState.remote_status}`}>
                  {systemState.remote_status === 'online' ? `${systemState.remote_latency_ms ?? '--'} ms` : 'Offline'}
                </span>
              </div>

              <div className="status-row">
                <span>Active Timers</span>
                <span className="status-value">{activeTimerCount}</span>
              </div>

              <div className="status-row">
                <span>Last Latency</span>
                <span className="status-value mono">
                  {systemState.end_to_final_ms !== null ? `${systemState.end_to_final_ms} ms` : '--'}
                </span>
              </div>

              <div className="status-row">
                <span>Last Activity</span>
                <span className="status-value mono">{systemState.last_activity ?? '--'}</span>
              </div>
            </section>
          </aside>

          <section className="conversation-column">
            <ConversationPanel />
          </section>

          <section className="tool-column">
            <RecipePanel />
            <TimersPanel onOpenTimerModal={() => setTimerModalOpen(true)} />
          </section>
        </main>

        {connectionState !== 'connected' && (
          <div className="connection-status-banner" role="status" aria-live="polite">
            {connectionState === 'reconnecting'
              ? 'Realtime feed reconnecting...'
              : 'Realtime feed disconnected. Retrying...'}
          </div>
        )}
      </div>

      <TimerModal isOpen={timerModalOpen} onClose={() => setTimerModalOpen(false)} />
    </div>
  );
}

export default App;

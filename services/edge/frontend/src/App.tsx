import { useMemo, useState } from 'react';
import { Header } from './components/Header';
import { ConversationPanel } from './components/panels/ConversationPanel';
import { TimersPanel } from './components/panels/TimersPanel';
import { RecipePanel } from './components/panels/RecipePanel';
import { TimerModal } from './components/modals/TimerModal';
import { useWebSocket } from './contexts/WebSocketContext';
import './styles/app.css';

type QuickActionId = 'lights' | 'stop_audio' | 'calendar';

const summarizeResult = (result: string, fallback: string): string => {
  const compact = result.replace(/\s+/g, ' ').trim();
  if (!compact) {
    return fallback;
  }
  if (compact.length > 92) {
    return `${compact.slice(0, 89)}...`;
  }
  return compact;
};

function App() {
  const { assistantState, connectionState, systemState, timers, sendToolCall } = useWebSocket();
  const [timerModalOpen, setTimerModalOpen] = useState(false);
  const [lightsOn, setLightsOn] = useState(false);
  const [busyQuickAction, setBusyQuickAction] = useState<QuickActionId | null>(null);
  const [quickActionNote, setQuickActionNote] = useState('Tap an action to run it instantly.');

  const activeTimerCount = timers.filter((timer) => timer.remaining_seconds > 0).length;
  const quickActionsDisabled = busyQuickAction !== null;
  const openTimerModal = () => setTimerModalOpen(true);

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

  const handleTimerQuickAction = () => {
    if (quickActionsDisabled) {
      return;
    }
    setQuickActionNote('Choose a duration, then start the timer.');
    openTimerModal();
  };

  const handleLightsQuickAction = async () => {
    if (quickActionsDisabled) {
      return;
    }

    const previousState = lightsOn;
    const nextState = !previousState;
    setLightsOn(nextState);
    setBusyQuickAction('lights');
    setQuickActionNote(nextState ? 'Turning lights on...' : 'Turning lights off...');

    try {
      const response = await sendToolCall('control_home_lighting', {
        device_id: 999,
        brightness: nextState ? 100 : 0,
      });

      if (!response.ok) {
        setLightsOn(previousState);
        setQuickActionNote(summarizeResult(response.result, 'Lights command failed.'));
        return;
      }

      setQuickActionNote(nextState ? 'All lights are on.' : 'All lights are off.');
    } finally {
      setBusyQuickAction(null);
    }
  };

  const handleStopAudioQuickAction = async () => {
    if (quickActionsDisabled) {
      return;
    }

    setBusyQuickAction('stop_audio');
    setQuickActionNote('Stopping audio...');

    try {
      const response = await sendToolCall('stop_music', {});
      if (response.ok) {
        setQuickActionNote(summarizeResult(response.result, 'Audio stopped.'));
        return;
      }
      setQuickActionNote(summarizeResult(response.result, 'Stop audio command failed.'));
    } finally {
      setBusyQuickAction(null);
    }
  };

  const handleCalendarQuickAction = async () => {
    if (quickActionsDisabled) {
      return;
    }

    setBusyQuickAction('calendar');
    setQuickActionNote('Getting today\'s agenda...');

    try {
      const response = await sendToolCall('get_calendar_agenda', { when: 'today' });
      if (response.ok) {
        setQuickActionNote(summarizeResult(response.result, 'Agenda loaded.'));
        return;
      }
      setQuickActionNote(summarizeResult(response.result, 'Calendar request failed.'));
    } finally {
      setBusyQuickAction(null);
    }
  };

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
                <button
                  type="button"
                  className={`quick-action ${lightsOn ? 'quick-action-active' : ''} ${busyQuickAction === 'lights' ? 'quick-action-busy' : ''}`}
                  onClick={handleLightsQuickAction}
                  disabled={quickActionsDisabled}
                  aria-pressed={lightsOn}
                >
                  <span className="quick-action-icon">💡</span>
                  <span>{busyQuickAction === 'lights' ? 'Working...' : lightsOn ? 'Lights Off' : 'Lights On'}</span>
                </button>
                <button
                  type="button"
                  className="quick-action"
                  onClick={handleTimerQuickAction}
                  disabled={quickActionsDisabled}
                >
                  <span className="quick-action-icon">⏱</span>
                  <span>Set Timer</span>
                </button>
                <button
                  type="button"
                  className={`quick-action ${busyQuickAction === 'stop_audio' ? 'quick-action-busy' : ''}`}
                  onClick={handleStopAudioQuickAction}
                  disabled={quickActionsDisabled}
                >
                  <span className="quick-action-icon">⏹</span>
                  <span>{busyQuickAction === 'stop_audio' ? 'Working...' : 'Stop Audio'}</span>
                </button>
                <button
                  type="button"
                  className={`quick-action ${busyQuickAction === 'calendar' ? 'quick-action-busy' : ''}`}
                  onClick={handleCalendarQuickAction}
                  disabled={quickActionsDisabled}
                >
                  <span className="quick-action-icon">📅</span>
                  <span>{busyQuickAction === 'calendar' ? 'Working...' : 'Calendar'}</span>
                </button>
              </div>
              <p className="rail-note" role="status" aria-live="polite">
                {quickActionNote}
              </p>
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
            <TimersPanel onOpenTimerModal={openTimerModal} />
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

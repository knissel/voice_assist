import React from 'react';
import { useWebSocket } from '../../contexts/WebSocketContext';
import styles from './TimersPanel.module.css';

interface TimersPanelProps {
  onOpenTimerModal?: () => void;
}

export const TimersPanel: React.FC<TimersPanelProps> = ({ onOpenTimerModal }) => {
  const { timers, sendToolCall } = useWebSocket();

  const activeTimers = timers.filter((timer) => timer.remaining_seconds > 0).length;

  const cancelTimer = (name?: string) => {
    sendToolCall('cancel_timer', { name: name ?? '' });
  };

  const formatTime = (seconds: number) => {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    if (h > 0) {
      return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    }
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  return (
    <section className={`${styles.timersPanel} ${timers.length === 0 ? styles.isEmpty : ''}`}>
      <div className={styles.panelHeader}>
        <h2 className={styles.panelTitle}>Timers</h2>
        <div className={styles.panelActions}>
          {onOpenTimerModal && (
            <button className={styles.panelBtn} type="button" onClick={onOpenTimerModal}>
              New
            </button>
          )}
          <span className={styles.panelMeta}>{activeTimers > 0 ? `${activeTimers} active` : 'None'}</span>
        </div>
      </div>
      <div className={styles.timersList}>
        {timers.length === 0 ? (
          <div className={styles.noTimers}>
            <div className={styles.noTimersIcon}>
              <svg viewBox="0 0 24 24">
                <circle cx="12" cy="13" r="8" fill="none" strokeWidth="2"></circle>
                <path d="M12 13V9" strokeWidth="2" strokeLinecap="round"></path>
                <path d="M9 2h6" strokeWidth="2" strokeLinecap="round"></path>
              </svg>
            </div>
            <div className={styles.noTimersTitle}>No active timers</div>
            <div className={styles.noTimersHint}>Say "Set a timer for 5 minutes"</div>
          </div>
        ) : (
          timers.map((timer) => {
            const remaining = timer.remaining_seconds || 0;
            const duration = timer.duration_seconds || 1;
            const progress = Math.max(0, Math.min(100, ((duration - remaining) / duration) * 100));
            const isUrgent = remaining <= 10 && remaining > 0;
            const isComplete = remaining <= 0;

            return (
              <div key={timer.id} className={`${styles.timerCard} ${isUrgent ? styles.urgent : ''} ${isComplete ? styles.complete : ''}`}>
                <div className={styles.timerIcon}>
                  {isComplete ? (
                    <svg viewBox="0 0 24 24">
                      <path d="M20 6L9 17l-5-5" fill="none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"></path>
                    </svg>
                  ) : (
                    <svg viewBox="0 0 24 24">
                      <circle cx="12" cy="13" r="8" fill="none" strokeWidth="2"></circle>
                      <path d="M12 13V9" strokeWidth="2" strokeLinecap="round"></path>
                      <path d="M9 2h6" strokeWidth="2" strokeLinecap="round"></path>
                    </svg>
                  )}
                </div>
                <div className={styles.timerInfo}>
                  <div className={styles.timerName}>{timer.name || 'Timer'}</div>
                  <div className={styles.timerRemaining}>{isComplete ? 'Done!' : formatTime(remaining)}</div>
                  <div className={styles.timerProgress}>
                    <div className={styles.timerProgressBar} style={{ width: `${progress}%` }}></div>
                  </div>
                </div>
                <button className={styles.timerCancel} type="button" onClick={() => cancelTimer(timer.name)}>
                  ×
                </button>
              </div>
            );
          })
        )}
      </div>
    </section>
  );
};

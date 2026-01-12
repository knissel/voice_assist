import React from 'react';
import { useWebSocket } from '../contexts/WebSocketContext';
import styles from './Header.module.css';

export const Header: React.FC = () => {
    const { assistantState, timers } = useWebSocket();

    const getStatusText = (state: string) => {
        switch (state) {
            case 'listening': return 'Listening...';
            case 'transcribing': return 'Processing...';
            case 'thinking': return 'Thinking...';
            case 'executing': return 'Executing...';
            case 'speaking': return 'Speaking...';
            default: return 'Ready';
        }
    };

    // Find the timer with the least remaining time (that isn't 0) to show in header
    const activeTimers = timers.filter(t => t.remaining_seconds > 0);
    const primaryTimer = activeTimers.length > 0
        ? activeTimers.sort((a, b) => a.remaining_seconds - b.remaining_seconds)[0]
        : null;

    const formatTime = (seconds: number) => {
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        const s = seconds % 60;
        if (h > 0) return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
        return `${m}:${s.toString().padStart(2, '0')}`;
    };

    return (
        <header className={styles.header}>
            <div className={styles.logo}>
                <div className={styles.logoIcon}>AI</div>
                <span className={styles.logoText}>Computer</span>
            </div>

            <div className={`${styles.headerTimer} ${!primaryTimer ? styles.isHidden : ''}`}>
                <span className={styles.headerTimerIcon}>
                    <svg viewBox="0 0 24 24">
                        <circle cx="12" cy="13" r="8" fill="none" strokeWidth="2"></circle>
                        <path d="M12 13V9" strokeWidth="2" strokeLinecap="round"></path>
                        <path d="M9 2h6" strokeWidth="2" strokeLinecap="round"></path>
                    </svg>
                </span>
                <span className={styles.headerTimerText}>
                    {primaryTimer ? `${formatTime(primaryTimer.remaining_seconds)} ${primaryTimer.label ? `(${primaryTimer.label})` : ''}` : 'No active timers'}
                </span>
            </div>

            <div className={`${styles.statusBadge} ${assistantState === 'listening' ? styles.wakeword : ''}`}>
                <div className={`${styles.statusDot} ${styles[assistantState] || ''}`}></div>
                <span>{getStatusText(assistantState)}</span>
            </div>
        </header>
    );
};

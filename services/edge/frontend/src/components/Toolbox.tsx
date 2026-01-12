import React, { useState } from 'react';
import { useWebSocket } from '../contexts/WebSocketContext';
import styles from './Toolbox.module.css';

interface ToolboxProps {
    onOpenTimerModal: () => void;
}

export const Toolbox: React.FC<ToolboxProps> = ({ onOpenTimerModal }) => {
    const { sendToolCall } = useWebSocket();
    const [lightsOn, setLightsOn] = useState(false);

    const toggleLights = () => {
        const nextState = !lightsOn;
        setLightsOn(nextState);
        // device_id 999 is "All Lights" convention
        sendToolCall('control_home_lighting', {
            device_id: 999,
            brightness: nextState ? 100 : 0
        });
    };

    const stopMusic = () => {
        sendToolCall('stop_music', {});
    };

    return (
        <aside className={styles.toolbox} aria-label="Toolbox">
            <button
                className={`${styles.actionBtn} ${lightsOn ? styles.active : ''}`}
                type="button"
                onClick={toggleLights}
                aria-pressed={lightsOn}
            >
                <span className={styles.actionIcon}>
                    {lightsOn ? (
                        <svg viewBox="0 0 24 24">
                            <path d="M9 21h6" strokeWidth="2" strokeLinecap="round"></path>
                            <path d="M10 17h4" strokeWidth="2" strokeLinecap="round"></path>
                            <path d="M12 3a6 6 0 0 0-3.5 10.9c.7.5 1.2 1.3 1.4 2.1h4.2c.2-.8.7-1.6 1.4-2.1A6 6 0 0 0 12 3z" fill="none" strokeWidth="2"></path>
                        </svg>
                    ) : (
                        <svg viewBox="0 0 24 24">
                            <path d="M21 14.5A8.5 8.5 0 0 1 9.5 3a7 7 0 1 0 11.5 11.5z" fill="none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"></path>
                        </svg>
                    )}
                </span>
                <span className={styles.actionLabel}>{lightsOn ? 'Lights On' : 'Lights Off'}</span>
            </button>

            <button className={styles.actionBtn} type="button" onClick={onOpenTimerModal}>
                <span className={styles.actionIcon}>
                    <svg viewBox="0 0 24 24">
                        <circle cx="12" cy="13" r="8" fill="none" strokeWidth="2"></circle>
                        <path d="M12 13V9" strokeWidth="2" strokeLinecap="round"></path>
                        <path d="M9 2h6" strokeWidth="2" strokeLinecap="round"></path>
                    </svg>
                </span>
                <span className={styles.actionLabel}>Set Timer</span>
            </button>

            <button className={styles.actionBtn} type="button" onClick={stopMusic}>
                <span className={styles.actionIcon}>
                    <svg viewBox="0 0 24 24">
                        <rect x="7" y="7" width="10" height="10" fill="currentColor"></rect>
                    </svg>
                </span>
                <span className={styles.actionLabel}>Stop Audio</span>
            </button>
        </aside>
    );
};

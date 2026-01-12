import React, { useState } from 'react';
import { useWebSocket } from '../../contexts/WebSocketContext';
import styles from './TimerModal.module.css';

interface TimerModalProps {
    isOpen: boolean;
    onClose: () => void;
}

export const TimerModal: React.FC<TimerModalProps> = ({ isOpen, onClose }) => {
    const { sendToolCall } = useWebSocket();
    const [hours, setHours] = useState(0);
    const [minutes, setMinutes] = useState(5);
    const [seconds, setSeconds] = useState(0);
    const [name, setName] = useState('');

    if (!isOpen) return null;

    const handleCreate = () => {
        const totalMinutes = hours * 60 + minutes + (seconds / 60);
        if (totalMinutes > 0) {
            // API expects duration_minutes as float or int
            sendToolCall('set_timer', {
                duration_minutes: Number(totalMinutes.toFixed(2)),
                name
            });
            onClose();
            // Reset
            setHours(0);
            setMinutes(5);
            setSeconds(0);
            setName('');
        }
    };

    return (
        <div className={`${styles.modalOverlay} ${isOpen ? styles.active : ''}`}>
            <div className={styles.modal}>
                <h3 className={styles.modalTitle}>Set Timer</h3>
                <div className={styles.timerInputGroup}>
                    <div className={styles.timerInput}>
                        <input
                            type="number"
                            value={hours}
                            onChange={(e) => setHours(Math.max(0, parseInt(e.target.value) || 0))}
                            min="0" max="23"
                        />
                        <label>Hours</label>
                    </div>
                    <div className={styles.timerInput}>
                        <input
                            type="number"
                            value={minutes}
                            onChange={(e) => setMinutes(Math.max(0, parseInt(e.target.value) || 0))}
                            min="0" max="59"
                        />
                        <label>Minutes</label>
                    </div>
                    <div className={styles.timerInput}>
                        <input
                            type="number"
                            value={seconds}
                            onChange={(e) => setSeconds(Math.max(0, parseInt(e.target.value) || 0))}
                            min="0" max="59"
                        />
                        <label>Seconds</label>
                    </div>
                </div>
                <input
                    type="text"
                    className={styles.timerNameInput}
                    placeholder="Timer name (optional)"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                />
                <div className={styles.modalActions}>
                    <button className={`${styles.modalBtn} ${styles.cancel}`} onClick={onClose}>Cancel</button>
                    <button className={`${styles.modalBtn} ${styles.confirm}`} onClick={handleCreate}>Start Timer</button>
                </div>
            </div>
        </div>
    );
};

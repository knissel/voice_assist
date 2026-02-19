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

  if (!isOpen) {
    return null;
  }

  const handleCreate = () => {
    const totalSeconds = hours * 3600 + minutes * 60 + seconds;
    if (totalSeconds <= 0) {
      return;
    }

    const durationMinutes = Math.max(1, Math.ceil(totalSeconds / 60));
    sendToolCall('set_timer', {
      duration_minutes: durationMinutes,
      name: name.trim(),
    });

    onClose();
    setHours(0);
    setMinutes(5);
    setSeconds(0);
    setName('');
  };

  return (
    <div className={`${styles.modalOverlay} ${isOpen ? styles.active : ''}`} role="dialog" aria-modal="true" aria-label="Set timer">
      <div className={styles.modal}>
        <h3 className={styles.modalTitle}>Set Timer</h3>
        <div className={styles.timerInputGroup}>
          <div className={styles.timerInput}>
            <input
              type="number"
              value={hours}
              onChange={(e) => setHours(Math.max(0, parseInt(e.target.value, 10) || 0))}
              min="0"
              max="23"
            />
            <label>Hours</label>
          </div>
          <div className={styles.timerInput}>
            <input
              type="number"
              value={minutes}
              onChange={(e) => setMinutes(Math.max(0, parseInt(e.target.value, 10) || 0))}
              min="0"
              max="59"
            />
            <label>Minutes</label>
          </div>
          <div className={styles.timerInput}>
            <input
              type="number"
              value={seconds}
              onChange={(e) => setSeconds(Math.max(0, parseInt(e.target.value, 10) || 0))}
              min="0"
              max="59"
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
          <button className={`${styles.modalBtn} ${styles.cancel}`} type="button" onClick={onClose}>Cancel</button>
          <button className={`${styles.modalBtn} ${styles.confirm}`} type="button" onClick={handleCreate}>Start Timer</button>
        </div>
      </div>
    </div>
  );
};

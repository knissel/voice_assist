import React, { useEffect, useRef } from 'react';
import { useWebSocket } from '../../contexts/WebSocketContext';
import type { ChatMessage } from '../../types';
import styles from './ConversationPanel.module.css';

export const ConversationPanel: React.FC = () => {
  const { messages, clearMessages } = useWebSocket();
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
  }, [messages]);

  const formatTime = (timestamp: number) => {
    return new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <section className={styles.conversationPanel}>
      <div className={styles.panelHeader}>
        <h2 className={styles.panelTitle}>Conversation</h2>
        <div className={styles.panelActions}>
          <button className={styles.panelBtn} type="button" onClick={clearMessages}>Clear</button>
        </div>
      </div>
      <div className={styles.conversationList} ref={listRef}>
        {messages.length === 0 && (
          <div className={`${styles.message} ${styles.assistant}`}>
            <div className={styles.messageText}>Hello. Say "Hey Jarvis" to wake me up.</div>
          </div>
        )}
        {messages.map((msg) => (
          <ConversationMessage key={msg.id} msg={msg} formatTime={formatTime} />
        ))}
      </div>
    </section>
  );
};

interface MessageProps {
  msg: ChatMessage;
  formatTime: (timestamp: number) => string;
}

const ConversationMessage: React.FC<MessageProps> = ({ msg, formatTime }) => {
  const roleClass = msg.role === 'tool' ? styles.tool : styles[msg.role];

  return (
    <div className={`${styles.message} ${roleClass || ''}`}>
      {msg.toolCall ? (
        <div className={styles.messageText}>
          <span className={styles.inlineIcon}>{getToolIcon(msg.toolCall.name)}</span>
          {formatToolMessage(msg)}
        </div>
      ) : (
        <div className={styles.messageText}>{msg.text}</div>
      )}
      <div className={styles.messageTime}>{formatTime(msg.timestamp)}</div>
    </div>
  );
};

const getToolIcon = (name: string) => {
  if (name.includes('light')) {
    return '💡';
  }
  if (name.includes('timer')) {
    return '⏱';
  }
  if (name.includes('music') || name.includes('audio')) {
    return '🎵';
  }
  if (name.includes('calendar')) {
    return '📅';
  }
  if (name.includes('recipe') || name.includes('dough')) {
    return '🍕';
  }
  return '🔧';
};

const formatToolMessage = (msg: ChatMessage) => {
  const { name, args, result, isError } = msg.toolCall || {};
  let text = name || 'tool_call';

  if (name === 'control_home_lighting' && args) {
    const deviceId = Number(args.device_id ?? -1);
    const brightness = Number(args.brightness ?? 0);
    if (deviceId === 999) {
      text = brightness === 0 ? 'All lights off' : brightness === 100 ? 'All lights on' : `All lights set to ${brightness}%`;
    } else {
      text = `Light ${deviceId} set to ${brightness}%`;
    }
  } else if (name === 'set_timer' && args) {
    const timerName = args.name ? ` (${String(args.name)})` : '';
    text = `Timer set for ${String(args.duration_minutes ?? '?')} min${timerName}`;
  } else if (name === 'cancel_timer' && args) {
    const timerName = args.name ? String(args.name) : 'latest timer';
    text = `Cancelled ${timerName}`;
  } else if (name === 'stop_music') {
    text = 'Stopped audio';
  }

  if (result) {
    text += ` - ${result}`;
  }
  if (isError) {
    text = `WARN: ${text}`;
  }

  return text;
};

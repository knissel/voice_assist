import React, { useEffect, useRef } from 'react';
import { useWebSocket } from '../../contexts/WebSocketContext';
import styles from './ConversationPanel.module.css';

export const ConversationPanel: React.FC = () => {
    const { messages, clearMessages } = useWebSocket();
    const listRef = useRef<HTMLDivElement>(null);

    // Auto-scroll to bottom
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
                        <div className={styles.messageText}>Hello! I'm Computer. Say "Computer" to wake me up.</div>
                    </div>
                )}
                {messages.map((msg) => (
                    <div key={msg.id} className={`${styles.message} ${styles[msg.role] || ''} ${msg.toolCall ? styles.tool : ''}`}>
                        {msg.toolCall ? (
                            <div className={styles.messageText} dangerouslySetInnerHTML={{
                                __html: `<span class="${styles.inlineIcon}">${getToolIcon(msg.toolCall.name)}</span> ${formatToolMessage(msg)}`
                            }} />
                        ) : (
                            <div className={styles.messageText}>{msg.text}</div>
                        )}
                        <div className={styles.messageTime}>{formatTime(msg.timestamp)}</div>
                    </div>
                ))}
            </div>
        </section>
    );
};

// Helper for tool messages
const getToolIcon = (_name: string) => {
    // Return SVG strings - simplified for now, ideally components
    // Reusing the same SVGs as legacy code
    return '🔧'; // Placeholder, legacy used SVG strings
};

const formatToolMessage = (msg: any) => {
    const { name, args, result, isError } = msg.toolCall;
    let text = name;

    if (name === 'control_home_lighting' && args) {
        if (args.device_id === 999) {
            text = args.brightness === 0 ? 'All lights off' : 'All lights on';
        } else {
            text = `Light ${args.device_id} -> ${args.brightness}%`;
        }
    } else if (name === 'set_timer' && args) {
        const label = args.name ? ` (${args.name})` : '';
        text = `Timer set for ${args.duration_minutes} min${label}`;
    } else if (name === 'stop_music') {
        text = 'Stopped audio';
    }

    if (result) text += ` - ${result}`;
    if (isError) text = `WARN: ${text}`;

    return text;
};

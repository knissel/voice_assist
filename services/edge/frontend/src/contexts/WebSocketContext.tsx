import React, { createContext, useContext, useEffect, useState, useRef, type ReactNode } from 'react';
import type { AssassinState, Card, ChatMessage, Timer } from '../types';

interface WebSocketContextType {
    isConnected: boolean;
    assistantState: AssassinState;
    messages: ChatMessage[];
    timers: Timer[];
    activeCard: Card | null;
    sendMessage: (type: string, data?: any) => void;
    sendToolCall: (tool: string, args?: any) => void;
    clearMessages: () => void;
}

const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined);

export const useWebSocket = () => {
    const context = useContext(WebSocketContext);
    if (!context) {
        throw new Error('useWebSocket must be used within a WebSocketProvider');
    }
    return context;
};

interface WebSocketProviderProps {
    children: ReactNode;
}

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({ children }) => {
    const [isConnected, setIsConnected] = useState(false);
    const [assistantState, setAssistantState] = useState<AssassinState>('idle');
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [timers, setTimers] = useState<Timer[]>([]);
    const [activeCard, setActiveCard] = useState<Card | null>(null);

    const ws = useRef<WebSocket | null>(null);
    const reconnectAttempts = useRef(0);
    const maxReconnectAttempts = 10;
    const reconnectTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);

    const connect = () => {
        const hostname = window.location.hostname;
        const wsUrl = `ws://${hostname}:8766`;

        console.log(`Connecting to ${wsUrl}...`);
        ws.current = new WebSocket(wsUrl);

        ws.current.onopen = () => {
            console.log('Connected to assistant');
            setIsConnected(true);
            reconnectAttempts.current = 0;
            setAssistantState('idle');
        };

        ws.current.onclose = () => {
            console.log('Disconnected from assistant');
            setIsConnected(false);
            scheduleReconnect();
        };

        ws.current.onerror = (error) => {
            console.error('WebSocket error:', error);
            // onClose will be called automatically
        };

        ws.current.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleEvent(data);
            } catch (e) {
                console.error('Failed to parse message:', e);
            }
        };
    };

    const scheduleReconnect = () => {
        if (reconnectAttempts.current < maxReconnectAttempts) {
            reconnectAttempts.current++;
            const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 30000);
            reconnectTimeout.current = setTimeout(connect, delay);
        }
    };

    const handleEvent = (event: any) => {
        console.log('Event:', event.type, event.data);

        switch (event.type) {
            case 'state_changed':
                setAssistantState(event.data.to_state);
                break;
            case 'wakeword_detected':
                addMessage('system', 'WAKE: Wake word detected');
                // Trigger flash effect if needed (handled by state change to listening usually)
                break;
            case 'transcript_final':
                addMessage('user', event.data.text);
                break;
            case 'assistant_text':
                addMessage('assistant', event.data.text);
                break;
            case 'tool_call':
                if (event.data.origin !== 'ui') {
                    addToolMessage(event.data.tool_name, event.data.arguments);
                }
                break;
            case 'tool_result':
                if (!event.data.success) {
                    addToolMessage(event.data.tool_name, null, event.data.result, true);
                }
                break;
            case 'ui_card':
                if (event.data && event.data.card) {
                    setActiveCard(event.data.card);
                }
                break;
            case 'timer_started':
                setTimers(prev => [...prev.filter(t => t.id !== event.data.id), event.data]);
                break;
            case 'timer_tick':
                setTimers(prev => prev.map(t => t.id === event.data.id ? event.data : t));
                break;
            case 'timer_complete':
                setTimers(prev => prev.map(t => t.id === event.data.id ? { ...t, remaining_seconds: 0 } : t)); // Ensure it shows complete
                break;
            case 'timer_cancelled':
                setTimers(prev => prev.filter(t => t.id !== event.data.id));
                break;
            case 'error':
                addMessage('assistant', `WARN: ${event.data.message}`);
                break;
        }
    };

    const addMessage = (role: 'user' | 'assistant' | 'system', text: string) => {
        const msg: ChatMessage = {
            id: Date.now().toString() + Math.random().toString(),
            role,
            text,
            timestamp: Date.now()
        };
        setMessages(prev => [...prev.slice(-19), msg]);
    };

    const addToolMessage = (name: string, args: any, result?: string, isError?: boolean) => {
        const msg: ChatMessage = {
            id: Date.now().toString() + Math.random().toString(),
            role: 'system', // Display as tool/system
            text: '', // Text handled by renderer
            timestamp: Date.now(),
            toolCall: { name, args, result, isError }
        };
        setMessages(prev => [...prev.slice(-19), msg]);
    }

    const sendMessage = (type: string, data: any = {}) => {
        if (ws.current && ws.current.readyState === WebSocket.OPEN) {
            ws.current.send(JSON.stringify({ type, ...data }));
        }
    };

    const sendToolCall = (tool: string, args: any = {}) => {
        sendMessage('tool_call', { tool, args });
    };

    const clearMessages = () => {
        setMessages([]);
    };

    useEffect(() => {
        connect();
        return () => {
            if (ws.current) {
                ws.current.close();
            }
            if (reconnectTimeout.current) {
                clearTimeout(reconnectTimeout.current);
            }
        };
    }, []);

    return (
        <WebSocketContext.Provider value={{
            isConnected,
            assistantState,
            messages,
            timers,
            activeCard,
            sendMessage,
            sendToolCall,
            clearMessages
        }}>
            {children}
        </WebSocketContext.Provider>
    );
};

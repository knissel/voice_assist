import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import type {
  AssistantEvent,
  AssistantState,
  Card,
  ChatMessage,
  ConnectionState,
  SystemState,
  Timer,
  UICommandResponse,
} from '../types';

interface WebSocketContextType {
  isConnected: boolean;
  connectionState: ConnectionState;
  assistantState: AssistantState;
  messages: ChatMessage[];
  timers: Timer[];
  activeCard: Card | null;
  systemState: SystemState;
  sendToolCall: (tool: string, args?: Record<string, unknown>) => Promise<UICommandResponse>;
  clearMessages: () => void;
}

const DEFAULT_SYSTEM_STATE: SystemState = {
  status: 'initializing',
  last_transcript: null,
  last_response: null,
  last_activity: null,
  remote_status: 'unknown',
  remote_latency_ms: null,
  end_to_final_ms: null,
};

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

const toAssistantState = (value: string): AssistantState => {
  const normalized = value as AssistantState;
  return normalized;
};

const normalizeTimer = (raw: Record<string, any>): Timer => ({
  id: String(raw.id ?? `timer_${Date.now()}`),
  name: String(raw.name ?? raw.label ?? 'Timer'),
  duration_seconds: Number(raw.duration_seconds ?? 0),
  remaining_seconds: Number(raw.remaining_seconds ?? 0),
  end_time: raw.end_time ? String(raw.end_time) : undefined,
  is_active: raw.is_active === undefined ? undefined : Boolean(raw.is_active),
  cancelled: raw.cancelled === undefined ? undefined : Boolean(raw.cancelled),
});

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({ children }) => {
  const [isConnected, setIsConnected] = useState(false);
  const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected');
  const [assistantState, setAssistantState] = useState<AssistantState>('initializing');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [timers, setTimers] = useState<Timer[]>([]);
  const [activeCard, setActiveCard] = useState<Card | null>(null);
  const [systemState, setSystemState] = useState<SystemState>(DEFAULT_SYSTEM_STATE);

  const sourceRef = useRef<EventSource | null>(null);
  const seenSignatures = useRef<string[]>([]);
  const seenSet = useRef<Set<string>>(new Set());

  const rememberEvent = useCallback((signature: string) => {
    if (seenSet.current.has(signature)) {
      return false;
    }

    seenSet.current.add(signature);
    seenSignatures.current.push(signature);

    if (seenSignatures.current.length > 400) {
      const stale = seenSignatures.current.splice(0, 100);
      for (const item of stale) {
        seenSet.current.delete(item);
      }
    }

    return true;
  }, []);

  const addMessage = useCallback((role: ChatMessage['role'], text: string) => {
    const msg: ChatMessage = {
      id: `${Date.now()}_${Math.random().toString(36).slice(2)}`,
      role,
      text,
      timestamp: Date.now(),
    };
    setMessages((prev) => [...prev.slice(-39), msg]);
  }, []);

  const addToolMessage = useCallback(
    (name: string, args: Record<string, unknown> | null, result?: string, isError?: boolean) => {
      const msg: ChatMessage = {
        id: `${Date.now()}_${Math.random().toString(36).slice(2)}`,
        role: 'tool',
        text: '',
        timestamp: Date.now(),
        toolCall: {
          name,
          args,
          result,
          isError,
        },
      };
      setMessages((prev) => [...prev.slice(-39), msg]);
    },
    [],
  );

  const upsertTimer = useCallback((nextTimer: Timer) => {
    setTimers((prev) => {
      const idx = prev.findIndex((timer) => timer.id === nextTimer.id);
      if (idx === -1) {
        return [...prev, nextTimer].sort((a, b) => a.remaining_seconds - b.remaining_seconds);
      }
      const next = [...prev];
      next[idx] = { ...next[idx], ...nextTimer };
      return next.sort((a, b) => a.remaining_seconds - b.remaining_seconds);
    });
  }, []);

  const handleEvent = useCallback(
    (event: AssistantEvent) => {
      const signature = `${event.timestamp ?? ''}|${event.correlation_id ?? ''}|${event.type}|${JSON.stringify(event.data ?? {})}`;
      if (!rememberEvent(signature)) {
        return;
      }

      switch (event.type) {
        case 'ui_connected':
          return;
        case 'state_changed': {
          const toState = String(event.data?.to_state ?? event.data?.state ?? 'idle');
          setAssistantState(toAssistantState(toState));
          break;
        }
        case 'wakeword_detected':
        case 'wake_word':
          addMessage('system', 'WAKE: Wake word detected');
          break;
        case 'transcript_final':
          if (event.data?.text) {
            addMessage('user', String(event.data.text));
          }
          break;
        case 'assistant_text': {
          const text = String(event.data?.text ?? '');
          if (text && !event.data?.is_partial) {
            addMessage('assistant', text);
          }
          break;
        }
        case 'tool_call': {
          const origin = String(event.data?.origin ?? 'assistant');
          if (origin !== 'ui') {
            addToolMessage(
              String(event.data?.tool_name ?? 'tool_call'),
              (event.data?.arguments as Record<string, unknown>) ?? null,
            );
          }
          break;
        }
        case 'tool_result':
          if (event.data?.success === false) {
            addToolMessage(
              String(event.data?.tool_name ?? 'tool_result'),
              null,
              String(event.data?.result ?? 'Tool failed'),
              true,
            );
          }
          break;
        case 'ui_card':
          if (event.data?.card) {
            setActiveCard(event.data.card as Card);
          }
          break;
        case 'timer_started':
          upsertTimer(normalizeTimer(event.data));
          break;
        case 'timer_tick':
          upsertTimer(normalizeTimer(event.data));
          break;
        case 'timer_complete': {
          const timerId = String(event.data?.id ?? '');
          const timerName = String(event.data?.name ?? 'Timer');
          upsertTimer(
            normalizeTimer({
              id: timerId,
              name: timerName,
              remaining_seconds: 0,
              is_active: false,
            }),
          );
          addMessage('assistant', `TIMER: "${timerName}" is complete.`);
          break;
        }
        case 'timer_cancelled':
          setTimers((prev) => prev.filter((timer) => timer.id !== String(event.data?.id ?? '')));
          break;
        case 'error':
          addMessage('assistant', `WARN: ${String(event.data?.message ?? 'Unknown error')}`);
          break;
        default:
          break;
      }
    },
    [addMessage, addToolMessage, rememberEvent, upsertTimer],
  );

  const connectEventStream = useCallback(() => {
    if (sourceRef.current) {
      sourceRef.current.close();
    }

    const source = new EventSource('/api/ui/events');
    sourceRef.current = source;

    setConnectionState('reconnecting');

    source.onopen = () => {
      setIsConnected(true);
      setConnectionState('connected');
    };

    source.onerror = () => {
      setIsConnected(false);
      setConnectionState('reconnecting');
    };

    source.onmessage = (event) => {
      try {
        const parsed = JSON.parse(event.data) as AssistantEvent;
        handleEvent(parsed);
      } catch (err) {
        console.error('Failed to parse SSE event', err);
      }
    };
  }, [handleEvent]);

  const fetchStatus = useCallback(async () => {
    try {
      const response = await fetch('/api/status');
      if (!response.ok) {
        throw new Error(`status ${response.status}`);
      }
      const data = (await response.json()) as SystemState;
      setSystemState(data);

      if (data.status) {
        setAssistantState(toAssistantState(data.status));
      }
    } catch (err) {
      console.error('Failed to fetch status', err);
      setConnectionState((prev) => (prev === 'connected' ? 'connected' : 'disconnected'));
    }
  }, []);

  const sendToolCall = useCallback(async (tool: string, args: Record<string, unknown> = {}) => {
    try {
      const response = await fetch('/api/ui/command', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ tool, args }),
      });

      const payload = (await response.json()) as UICommandResponse;
      if (!response.ok || !payload.ok) {
        addMessage('assistant', `WARN: ${payload.result || `Failed to run ${tool}`}`);
      }

      return payload;
    } catch (err) {
      console.error('Failed to execute UI command', err);
      const fallback = { ok: false, result: 'Command endpoint unavailable' };
      addMessage('assistant', `WARN: ${fallback.result}`);
      return fallback;
    }
  }, [addMessage]);

  const clearMessages = useCallback(() => {
    setMessages([]);
  }, []);

  useEffect(() => {
    connectEventStream();
    return () => {
      if (sourceRef.current) {
        sourceRef.current.close();
      }
    };
  }, [connectEventStream]);

  useEffect(() => {
    fetchStatus();
    const timer = window.setInterval(fetchStatus, 5000);
    return () => window.clearInterval(timer);
  }, [fetchStatus]);

  const value = useMemo(
    () => ({
      isConnected,
      connectionState,
      assistantState,
      messages,
      timers,
      activeCard,
      systemState,
      sendToolCall,
      clearMessages,
    }),
    [
      isConnected,
      connectionState,
      assistantState,
      messages,
      timers,
      activeCard,
      systemState,
      sendToolCall,
      clearMessages,
    ],
  );

  return <WebSocketContext.Provider value={value}>{children}</WebSocketContext.Provider>;
};

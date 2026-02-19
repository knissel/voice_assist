export type AssistantState =
  | 'initializing'
  | 'idle'
  | 'listening'
  | 'recording'
  | 'transcribing'
  | 'processing'
  | 'thinking'
  | 'thinking_local'
  | 'executing'
  | 'speaking'
  | 'error';

export interface Timer {
  id: string;
  name: string;
  duration_seconds: number;
  remaining_seconds: number;
  end_time?: string;
  is_active?: boolean;
  cancelled?: boolean;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system' | 'tool';
  text: string;
  timestamp: number;
  toolCall?: {
    name: string;
    args: Record<string, unknown> | null;
    result?: string;
    isError?: boolean;
  };
}

export interface RecipeStep {
  number: number;
  instruction: string;
}

export interface RecipeIngredient {
  name: string;
  grams: number;
  bakers_percent?: number;
}

export interface RecipeData {
  style?: string;
  ingredients: RecipeIngredient[];
  steps: RecipeStep[];
  ball_count?: number;
  ball_weight_g?: number;
  total_dough_g?: number;
  hydration_percent?: number;
  cold_ferment_hours?: number;
  bake_temp_f?: number;
  updated_at?: string;
}

export interface CalendarCardEvent {
  title: string;
  start: string;
  end?: string;
  location?: string;
  description?: string;
}

export interface Card {
  card_type: string;
  title: string;
  subtitle?: string;
  body?: string;
  data: Record<string, unknown>;
}

export type ConnectionState = 'connected' | 'disconnected' | 'reconnecting';

export interface SystemState {
  status: string;
  last_transcript: string | null;
  last_response: string | null;
  last_activity: string | null;
  remote_status: 'online' | 'offline' | 'unknown';
  remote_latency_ms: number | null;
  end_to_final_ms: number | null;
}

export interface AssistantEvent {
  type: string;
  data: Record<string, any>;
  correlation_id?: string;
  timestamp?: string;
}

export interface UICommandResponse {
  ok: boolean;
  result: string;
}

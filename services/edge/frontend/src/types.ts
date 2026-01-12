export interface Timer {
    id: string;
    duration_seconds: number;
    remaining_seconds: number;
    start_time: number;
    label?: string;
    is_paused: boolean;
}

export interface ChatMessage {
    id: string;
    role: 'user' | 'assistant' | 'system';
    text: string;
    timestamp: number;
    toolCall?: {
        name: string;
        args: any;
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

export interface Card {
    card_type: string;
    title: string;
    subtitle?: string;
    data: any;
}

export type ConnectionState = 'connected' | 'disconnected' | 'reconnecting';
export type AssassinState = 'idle' | 'listening' | 'transcribing' | 'thinking' | 'executing' | 'speaking';

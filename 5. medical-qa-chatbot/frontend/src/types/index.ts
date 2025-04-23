export interface Entity {
  text: string;
  type: string;
  start: number;
  end: number;
  source: string;
}

export interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  entities?: Entity[];
  source?: string;
  confidence?: number;
  isWelcome?: boolean;
  isConversational?: boolean;
  isUserMessage?: boolean;
  isLoading?: boolean;
}

export interface ChatState {
  messages: Message[];
  isLoading: boolean;
  error: string | null;
}

export interface QuestionRequest {
  question: string;
  max_results?: number;
}

export interface AnswerResponse {
  answer: string;
  source: string;
  confidence: number;
  entities?: Entity[];
}

export interface ChatResponse {
  answers: AnswerResponse[];
  entities_detected: Entity[];
}

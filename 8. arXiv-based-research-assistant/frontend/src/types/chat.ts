export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface ChatResponse {
  response: string;
  papers: PaperReference[];
}

export interface PaperReference {
  paper_id: string;
  title: string;
  authors: string;
  published_date: string;
  categories: string;
  url: string;
}

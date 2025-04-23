import axios from 'axios';
import { QuestionRequest, ChatResponse } from '../types';

const API_URL = 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const askQuestion = async (request: QuestionRequest): Promise<ChatResponse> => {
  try {
    const response = await api.post<ChatResponse>('/ask', request);
    return response.data;
  } catch (error) {
    console.error('Error asking question:', error);
    throw error;
  }
};

export const checkHealth = async (): Promise<{ status: string }> => {
  try {
    const response = await api.get<{ status: string }>('/health');
    return response.data;
  } catch (error) {
    console.error('Error checking health:', error);
    throw error;
  }
};

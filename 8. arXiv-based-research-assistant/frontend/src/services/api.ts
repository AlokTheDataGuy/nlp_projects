import axios from 'axios';
import { ChatMessage } from '../types/chat';
import { Paper } from '../types/paper';
import { Concept, ConceptRelation } from '../types/concept';

const API_URL = 'http://localhost:8000/api';

// Create axios instance
const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Chat API
export const sendChatMessage = async (message: string, conversationHistory: ChatMessage[] = []) => {
  try {
    const response = await api.post('/chat', {
      message,
      conversation_history: conversationHistory,
    });
    return response.data;
  } catch (error) {
    console.error('Error sending chat message:', error);
    throw error;
  }
};

// Papers API
export const searchPapers = async (query: string, maxResults: number = 10) => {
  try {
    const response = await api.post('/papers/search', {
      query,
      max_results: maxResults,
    });
    return response.data.papers as Paper[];
  } catch (error) {
    console.error('Error searching papers:', error);
    throw error;
  }
};

export const getPaperById = async (paperId: string) => {
  try {
    const response = await api.get(`/papers/${paperId}`);
    return response.data as Paper;
  } catch (error) {
    console.error(`Error getting paper with ID ${paperId}:`, error);
    throw error;
  }
};

// Concepts API
export const getConcepts = async () => {
  try {
    const response = await api.get('/concepts');
    return response.data as Concept[];
  } catch (error) {
    console.error('Error getting concepts:', error);
    throw error;
  }
};

export const getConceptById = async (conceptId: number) => {
  try {
    const response = await api.get(`/concepts/${conceptId}`);
    return response.data as Concept;
  } catch (error) {
    console.error(`Error getting concept with ID ${conceptId}:`, error);
    throw error;
  }
};

export const getConceptRelations = async () => {
  try {
    const response = await api.get('/concepts/relations');
    return response.data as ConceptRelation[];
  } catch (error) {
    console.error('Error getting concept relations:', error);
    throw error;
  }
};

export default api;

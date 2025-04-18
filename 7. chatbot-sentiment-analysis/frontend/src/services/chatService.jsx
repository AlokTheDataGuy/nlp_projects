import axios from 'axios';

const API_URL = 'http://localhost:8000';

/**
 * Send a message to the chatbot API
 * @param {string} content - The message content
 * @param {string} conversationId - Optional conversation ID for continuing a conversation
 * @returns {Promise<Object>} - The response from the API
 */
export const sendMessage = async (content, conversationId = null) => {
  try {
    const response = await axios.post(`${API_URL}/chat`, {
      content,
      conversation_id: conversationId
    });
    
    return response.data;
  } catch (error) {
    console.error('Error sending message:', error);
    throw error;
  }
};

/**
 * Get conversation history
 * @param {string} conversationId - The conversation ID
 * @returns {Promise<Array>} - The conversation history
 */
export const getConversationHistory = async (conversationId) => {
  try {
    const response = await axios.get(`${API_URL}/conversations/${conversationId}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching conversation history:', error);
    throw error;
  }
};

/**
 * Create a WebSocket connection for real-time chat
 * @param {string} conversationId - The conversation ID
 * @param {function} onMessage - Callback function for received messages
 * @returns {WebSocket} - The WebSocket connection
 */
export const createWebSocketConnection = (conversationId, onMessage) => {
  const ws = new WebSocket(`ws://localhost:8000/ws/${conversationId}`);
  
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    onMessage(data);
  };
  
  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
  };
  
  ws.onclose = () => {
    console.log('WebSocket connection closed');
  };
  
  return ws;
};

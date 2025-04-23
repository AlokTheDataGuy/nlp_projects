import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { FiSend, FiTrash2, FiInfo } from 'react-icons/fi';
import { BiLoaderAlt } from 'react-icons/bi';
import './App.css';

function App() {
  const [messages, setMessages] = useState([
    {
      id: 'welcome',
      role: 'system',
      content: 'Welcome to the Multilingual Chatbot! I can speak English, Hindi, Bengali, and Marathi. Feel free to type in any of these languages.'
    }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState('');
  const [languages, setLanguages] = useState([]);
  const [detectedLanguage, setDetectedLanguage] = useState('English');
  const messagesEndRef = useRef(null);

  // Create session on component mount
  useEffect(() => {
    // Create a new session ID
    setSessionId(generateSessionId());

    // Fetch languages for reference only (not for UI display)
    const fetchLanguages = async () => {
      try {
        const response = await axios.get('/api/languages');
        const languageOptions = response.data.map(lang => ({
          value: lang.code,
          label: lang.name
        }));
        setLanguages(languageOptions);
      } catch (error) {
        console.error('Error fetching languages:', error);
      }
    };

    fetchLanguages();
  }, []);

  // Scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const generateSessionId = () => {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
      const r = Math.random() * 16 | 0;
      const v = c === 'x' ? r : (r & 0x3 | 0x8);
      return v.toString(16);
    });
  };

  const addSystemMessage = (content) => {
    setMessages(prev => [...prev, { id: Date.now(), role: 'system', content }]);
  };

  const handleSendMessage = async () => {
    if (!input.trim()) return;

    // Add user message to chat
    const userMessage = { id: Date.now(), role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      // Send message to API
      const response = await axios.post('/api/chat', {
        message: input,
        session_id: sessionId,
        language: null  // Let the backend auto-detect the language
      });

      // Add bot response to chat
      const botMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: response.data.message
      };
      setMessages(prev => [...prev, botMessage]);

      // Update detected language
      setDetectedLanguage(getLanguageName(response.data.detected_language));
    } catch (error) {
      console.error('Error sending message:', error);

      // Check if we have a specific error message from the server
      let errorMessage = 'Error: Failed to get response from the chatbot.';

      if (error.response && error.response.data && error.response.data.detail) {
        errorMessage = `Error: ${error.response.data.detail}`;
      } else if (error.message && error.message.includes('timeout')) {
        errorMessage = 'Error: The server took too long to respond. The language model might be busy.';
      }

      addSystemMessage(errorMessage);

      // Add a retry suggestion
      addSystemMessage('Try again with a shorter message or wait a moment before retrying.');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Language switching is now handled automatically by the backend

  const handleClearConversation = async () => {
    try {
      // Send clear conversation request to API
      const response = await axios.post('/api/clear-conversation', {
        session_id: sessionId
      });

      if (response.data.success) {
        setMessages([{
          id: 'welcome',
          role: 'system',
          content: 'Conversation cleared. You can start a new conversation now.'
        }]);
      }
    } catch (error) {
      console.error('Error clearing conversation:', error);
      addSystemMessage('Error: Failed to clear conversation.');
    }
  };

  const getLanguageName = (code) => {
    const language = languages.find(lang => lang.value === code);
    return language ? language.label : code;
  };

  return (
    <div className="chat-container">
    <header className="chat-header">
      <h1>Multilingual Chatbot</h1>
      <div className="header-controls">
        <button
          className="clear-button"
          onClick={handleClearConversation}
          aria-label="Clear conversation"
        >
          <FiTrash2 /> Clear Chat
        </button>
      </div>
    </header>

      <div className="chat-messages">
        {messages.map(message => (
          <div key={message.id} className={`message ${message.role}`}>
            <div className="message-content">{message.content}</div>
          </div>
        ))}
        {loading && (
          <div className="message assistant loading">
            <div className="message-content">
              <div className="typing-indicator">
                <BiLoaderAlt className="spinner" />
                <span>Generating response...</span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input-container">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your message..."
          disabled={loading}
          rows={1}
          className="chat-input"
        />
        <button
          onClick={handleSendMessage}
          disabled={loading || !input.trim()}
          className="send-button"
          aria-label="Send message"
        >
          <FiSend />
        </button>
      </div>

      <footer className="chat-footer">
        <div className="chat-info">
          <div className="info-item">
            <FiInfo /> Session ID: <span>{sessionId}</span>
          </div>
          <div className="info-item">
            <FiInfo /> Detected Language: <span>{detectedLanguage}</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;

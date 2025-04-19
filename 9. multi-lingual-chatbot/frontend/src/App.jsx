import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import Select from 'react-select';
import { FiSend, FiTrash2, FiInfo } from 'react-icons/fi';
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
  const [selectedLanguage, setSelectedLanguage] = useState(null);
  const [detectedLanguage, setDetectedLanguage] = useState('English');
  const messagesEndRef = useRef(null);

  // Fetch languages and create session on component mount
  useEffect(() => {
    const fetchLanguages = async () => {
      try {
        const response = await axios.get('/api/languages');
        const languageOptions = response.data.map(lang => ({
          value: lang.code,
          label: lang.name
        }));
        setLanguages(languageOptions);
        
        // Set default language to English
        const englishOption = languageOptions.find(lang => lang.value === 'eng_Latn');
        if (englishOption) {
          setSelectedLanguage(englishOption);
        }
      } catch (error) {
        console.error('Error fetching languages:', error);
        addSystemMessage('Error loading supported languages. Please refresh the page.');
      }
    };

    // Create a new session ID
    setSessionId(generateSessionId());
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
        language: selectedLanguage?.value
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
      addSystemMessage('Error: Failed to get response from the chatbot.');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleLanguageChange = async (selectedOption) => {
    setSelectedLanguage(selectedOption);
    
    try {
      // Send language switch request to API
      const response = await axios.post('/api/switch-language', {
        language: selectedOption.value,
        session_id: sessionId
      });
      
      if (response.data.success) {
        addSystemMessage(`Language switched to ${response.data.language_name}`);
      }
    } catch (error) {
      console.error('Error switching language:', error);
      addSystemMessage('Error: Failed to switch language.');
    }
  };

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
          <div className="language-selector">
            <label htmlFor="language-select">Language:</label>
            <Select
              id="language-select"
              value={selectedLanguage}
              onChange={handleLanguageChange}
              options={languages}
              className="select-container"
              classNamePrefix="select"
              isSearchable
              placeholder="Select language"
            />
          </div>
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
        <div ref={messagesEndRef} />
      </div>
      
      <div className="chat-input-container">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
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

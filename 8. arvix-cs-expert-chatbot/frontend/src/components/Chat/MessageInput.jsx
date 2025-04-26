import React, { useState } from 'react';

/**
 * Message input component
 */
const MessageInput = ({ onSendMessage, disabled }) => {
  const [message, setMessage] = useState('');
  
  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (message.trim() && !disabled) {
      onSendMessage(message);
      setMessage('');
    }
  };
  
  return (
    <form onSubmit={handleSubmit} className="relative">
      <div className="flex items-center bg-gray-800 rounded-lg border border-gray-700 shadow-lg">
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Ask about computer science research..."
          className="flex-1 bg-transparent border-0 focus:ring-0 text-white px-4 py-3"
          disabled={disabled}
        />
        <button
          type="submit"
          className="p-2 rounded-r-lg text-blue-400 hover:text-blue-300 disabled:opacity-50 disabled:cursor-not-allowed"
          disabled={!message.trim() || disabled}
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5l7 7-7 7" />
          </svg>
        </button>
      </div>
      
      {/* Suggestions */}
      <div className="mt-3 flex flex-wrap gap-2 justify-center">
        <button
          type="button"
          onClick={() => onSendMessage("What are the latest trends in deep learning?")}
          className="text-sm bg-gray-800 hover:bg-gray-700 text-gray-300 py-1 px-3 rounded-full border border-gray-700"
          disabled={disabled}
        >
          Latest trends in deep learning
        </button>
        <button
          type="button"
          onClick={() => onSendMessage("Explain transformers in NLP")}
          className="text-sm bg-gray-800 hover:bg-gray-700 text-gray-300 py-1 px-3 rounded-full border border-gray-700"
          disabled={disabled}
        >
          Explain transformers in NLP
        </button>
        <button
          type="button"
          onClick={() => onSendMessage("Find papers about quantum computing algorithms")}
          className="text-sm bg-gray-800 hover:bg-gray-700 text-gray-300 py-1 px-3 rounded-full border border-gray-700"
          disabled={disabled}
        >
          Quantum computing papers
        </button>
      </div>
    </form>
  );
};

export default MessageInput;

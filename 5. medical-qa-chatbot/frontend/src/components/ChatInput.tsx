import React, { useState } from 'react';
import { FaArrowRight } from 'react-icons/fa';

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  isLoading: boolean;
}

const ChatInput: React.FC<ChatInputProps> = ({ onSendMessage, isLoading }) => {
  const [message, setMessage] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() && !isLoading) {
      onSendMessage(message);
      setMessage('');
    }
  };

  return (
    <form onSubmit={handleSubmit} className="mt-4 flex items-center">
      <div className="flex w-full">
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Type your message..."
          className="flex-grow py-3 px-4 text-base focus:outline-none border border-gray-200 rounded-l bg-white text-gray-800"
          disabled={isLoading}
        />
        <button
          type="submit"
          className="px-5 py-3 bg-gray-200 text-gray-600 hover:bg-gray-300 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center rounded-r border border-l-0 border-gray-200"
          disabled={isLoading || !message.trim()}
        >
          <span className="mr-1 font-medium text-sm">SEND</span>
          {isLoading ? (
            <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
          ) : (
            <FaArrowRight className="text-sm" />
          )}
        </button>
      </div>
    </form>
  );
};

export default ChatInput;

import React, { useEffect, useRef } from 'react';
import Message from './Message';
import WelcomeScreen from './WelcomeScreen';

/**
 * Message list component
 */
const MessageList = ({ messages, loading }) => {
  const messagesEndRef = useRef(null);
  
  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);
  
  return (
    <div className="flex-1 overflow-y-auto mb-4 space-y-4">
      {/* Welcome screen when no messages */}
      {(!messages || messages.length === 0) && !loading && (
        <WelcomeScreen />
      )}
      
      {/* Message list */}
      {messages && messages.map((message) => (
        <Message key={message.id} message={message} />
      ))}
      
      {/* Loading indicator */}
      {loading && (
        <div className="flex items-start">
          <div className="bg-gray-800 rounded-lg rounded-tl-none p-4 max-w-[80%] shadow-md border border-gray-700">
            <div className="flex items-center space-x-2">
              <div className="flex space-x-1">
                <div className="w-2 h-2 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '0ms' }}></div>
                <div className="w-2 h-2 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '150ms' }}></div>
                <div className="w-2 h-2 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '300ms' }}></div>
              </div>
              <span className="text-gray-400">Thinking...</span>
            </div>
          </div>
        </div>
      )}
      
      {/* Auto-scroll anchor */}
      <div ref={messagesEndRef} />
    </div>
  );
};

export default MessageList;

import React from 'react';
import MessageList from './MessageList';
import MessageInput from './MessageInput';
import { useChat } from '../../hooks/useChat';

/**
 * Main chat container component
 */
const ChatContainer = () => {
  const { messages, sendMessage, loading } = useChat();
  
  return (
    <div className="flex flex-col h-full max-w-4xl mx-auto w-full p-4">
      <MessageList messages={messages} loading={loading} />
      <MessageInput onSendMessage={sendMessage} disabled={loading} />
    </div>
  );
};

export default ChatContainer;

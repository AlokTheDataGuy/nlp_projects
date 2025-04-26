import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useChat } from '../../hooks/useChat';

/**
 * Sidebar component for conversations and navigation
 */
const Sidebar = () => {
  const location = useLocation();
  const { conversations, currentConversation, selectConversation, deleteConversation, clearChat } = useChat();
  const [isOpen, setIsOpen] = useState(true);
  
  // Toggle sidebar on mobile
  const toggleSidebar = () => {
    setIsOpen(!isOpen);
  };
  
  return (
    <>
      {/* Mobile sidebar toggle */}
      <div className="md:hidden fixed top-4 left-4 z-50">
        <button
          onClick={toggleSidebar}
          className="btn-icon bg-gray-800 text-white shadow-lg"
          aria-label={isOpen ? 'Close sidebar' : 'Open sidebar'}
        >
          {isOpen ? (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          ) : (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          )}
        </button>
      </div>
      
      {/* Sidebar */}
      <div className={`w-72 bg-gray-800 border-r border-gray-700 flex-shrink-0 flex flex-col h-full transition-all duration-300 ease-in-out ${isOpen ? 'translate-x-0' : '-translate-x-full'} md:translate-x-0 fixed md:static z-40`}>
        {/* Sidebar header */}
        <div className="p-4 border-b border-gray-700">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-bold text-white">Conversations</h2>
            <button
              onClick={toggleSidebar}
              className="md:hidden btn-icon text-gray-400 hover:text-white"
              aria-label="Close sidebar"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          
          <button
            onClick={clearChat}
            className="mt-4 w-full btn btn-primary"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
            </svg>
            New Conversation
          </button>
        </div>
        
        {/* Conversation list */}
        <div className="flex-1 overflow-y-auto p-2">
          {conversations && conversations.length > 0 ? (
            <div className="space-y-2">
              {conversations.map(conversation => (
                <div
                  key={conversation.id}
                  className={`p-3 rounded-md cursor-pointer transition-colors ${
                    currentConversation === conversation.id
                      ? 'bg-blue-600 bg-opacity-20 border-l-4 border-blue-600'
                      : 'hover:bg-gray-700'
                  }`}
                  onClick={() => selectConversation(conversation.id)}
                >
                  <div className="flex justify-between items-center">
                    <div className="truncate flex-1">
                      <p className="font-medium text-white">{conversation.title}</p>
                      <p className="text-xs text-gray-400">
                        {new Date(conversation.updated_at).toLocaleDateString()}
                      </p>
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteConversation(conversation.id);
                      }}
                      className="text-gray-400 hover:text-red-400 transition-colors"
                      aria-label="Delete conversation"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                      </svg>
                    </button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 px-4">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 mx-auto text-gray-500 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
              </svg>
              <p className="text-gray-400 mb-1">No conversations yet</p>
              <p className="text-sm text-gray-500">Start a new chat to begin</p>
            </div>
          )}
        </div>
        
        {/* Mobile navigation */}
        <div className="md:hidden p-4 border-t border-gray-700">
          <div className="grid grid-cols-3 gap-2">
            <Link
              to="/"
              className={`text-center py-2 px-3 rounded-md text-sm font-medium ${
                location.pathname === '/'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              Chat
            </Link>
            <Link
              to="/search"
              className={`text-center py-2 px-3 rounded-md text-sm font-medium ${
                location.pathname === '/search'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              Search
            </Link>
            <Link
              to="/visualization"
              className={`text-center py-2 px-3 rounded-md text-sm font-medium ${
                location.pathname === '/visualization'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              Viz
            </Link>
          </div>
        </div>
      </div>
    </>
  );
};

export default Sidebar;

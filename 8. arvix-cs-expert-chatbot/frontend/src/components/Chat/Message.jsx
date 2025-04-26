import React from 'react';

/**
 * Individual message component
 */
const Message = ({ message }) => {
  const isUser = message.role === 'user';
  
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div 
        className={`
          ${isUser 
            ? 'bg-blue-600 text-white rounded-lg rounded-tr-none' 
            : 'bg-gray-800 text-gray-100 rounded-lg rounded-tl-none border border-gray-700'
          }
          p-4 max-w-[80%] shadow-md
        `}
      >
        {/* Message content */}
        <div className="whitespace-pre-wrap">{message.content}</div>
        
        {/* Paper references */}
        {!isUser && message.papers && message.papers.length > 0 && (
          <div className="mt-3 pt-3 border-t border-gray-700">
            <div className="font-medium text-sm mb-1">Related Papers:</div>
            <ul className="space-y-1">
              {message.papers.slice(0, 3).map((paper, index) => (
                <li key={index} className="text-sm">
                  <a 
                    href={paper.pdf_url} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="text-blue-400 hover:text-blue-300 hover:underline flex items-start"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                    </svg>
                    <span className="truncate">{paper.title}</span>
                  </a>
                </li>
              ))}
              {message.papers.length > 3 && (
                <li className="text-xs text-gray-400 mt-1">
                  +{message.papers.length - 3} more papers
                </li>
              )}
            </ul>
          </div>
        )}
        
        {/* Timestamp */}
        <div className="text-xs opacity-70 mt-2 text-right">
          {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </div>
      </div>
    </div>
  );
};

export default Message;

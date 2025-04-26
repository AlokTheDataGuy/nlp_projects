import React, { useState } from 'react';

/**
 * Individual paper card component
 */
const PaperCard = ({ paper }) => {
  const [expanded, setExpanded] = useState(false);
  
  // Format date
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });
  };
  
  // Truncate abstract
  const truncateAbstract = (text, maxLength = 250) => {
    if (!text || text.length <= maxLength) return text;
    return text.slice(0, maxLength) + '...';
  };
  
  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden hover:shadow-lg transition-shadow">
      {/* Paper header */}
      <div className="p-4 border-b border-gray-700">
        <div className="flex justify-between items-start">
          <h3 className="text-lg font-medium text-blue-400 hover:text-blue-300 transition-colors">
            <a href={paper.pdf_url} target="_blank" rel="noopener noreferrer">
              {paper.title}
            </a>
          </h3>
          
          <div className="flex space-x-2">
            <a
              href={paper.pdf_url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-gray-400 hover:text-white transition-colors"
              title="Download PDF"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
            </a>
            
            <button
              onClick={() => setExpanded(!expanded)}
              className="text-gray-400 hover:text-white transition-colors"
              title={expanded ? "Show less" : "Show more"}
            >
              <svg xmlns="http://www.w3.org/2000/svg" className={`h-5 w-5 transform transition-transform ${expanded ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
          </div>
        </div>
        
        {/* Authors and date */}
        <div className="mt-2 flex flex-wrap items-center text-sm text-gray-400">
          <span className="mr-3">
            {paper.authors && paper.authors.length > 0 
              ? paper.authors.slice(0, 3).join(', ') + (paper.authors.length > 3 ? ' et al.' : '')
              : 'Unknown authors'}
          </span>
          
          <span className="mr-3">
            {paper.published_date ? formatDate(paper.published_date) : 'Unknown date'}
          </span>
          
          {paper.categories && paper.categories.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-1 md:mt-0">
              {paper.categories.slice(0, 3).map(category => (
                <span key={category} className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-900 text-blue-300">
                  {category}
                </span>
              ))}
              {paper.categories.length > 3 && (
                <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-700 text-gray-300">
                  +{paper.categories.length - 3}
                </span>
              )}
            </div>
          )}
        </div>
      </div>
      
      {/* Paper content */}
      <div className={`p-4 ${expanded ? '' : 'max-h-32 overflow-hidden'}`}>
        <p className="text-gray-300 text-sm">
          {expanded ? paper.abstract : truncateAbstract(paper.abstract)}
        </p>
        
        {!expanded && paper.abstract && paper.abstract.length > 250 && (
          <div className="absolute bottom-0 left-0 right-0 h-8 bg-gradient-to-t from-gray-800 to-transparent"></div>
        )}
      </div>
      
      {/* Paper footer */}
      {expanded && (
        <div className="px-4 py-3 bg-gray-900 border-t border-gray-700 flex justify-between items-center">
          <div className="flex space-x-4 text-sm">
            {paper.citation_count !== undefined && (
              <span className="text-gray-400 flex items-center">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z" />
                </svg>
                {paper.citation_count} citations
              </span>
            )}
            
            {paper.arxiv_id && (
              <a 
                href={`https://arxiv.org/abs/${paper.arxiv_id}`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-gray-400 hover:text-blue-400 transition-colors flex items-center"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                </svg>
                arXiv
              </a>
            )}
          </div>
          
          <button
            className="text-blue-400 hover:text-blue-300 transition-colors text-sm flex items-center"
            onClick={() => {/* Add to chat functionality */}}
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
            </svg>
            Ask about this paper
          </button>
        </div>
      )}
    </div>
  );
};

export default PaperCard;

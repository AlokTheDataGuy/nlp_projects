import React from 'react';
import PaperCard from './PaperCard';

/**
 * List of paper results
 */
const PaperList = ({ papers, loading, error }) => {
  // Loading state
  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-12">
        <div className="w-12 h-12 border-4 border-gray-600 border-t-blue-500 rounded-full animate-spin mb-4"></div>
        <p className="text-gray-400">Searching for papers...</p>
      </div>
    );
  }
  
  // Error state
  if (error) {
    return (
      <div className="bg-red-900 bg-opacity-20 border border-red-700 rounded-lg p-4 text-center">
        <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-red-500 mx-auto mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <h3 className="text-lg font-medium text-red-400 mb-1">Error Loading Papers</h3>
        <p className="text-gray-300">{error}</p>
      </div>
    );
  }
  
  // Empty state
  if (!papers || papers.length === 0) {
    return (
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-8 text-center">
        <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 text-gray-500 mx-auto mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
        <h3 className="text-lg font-medium text-white mb-2">No Papers Found</h3>
        <p className="text-gray-400 max-w-md mx-auto">
          Try adjusting your search terms or filters to find relevant papers.
        </p>
      </div>
    );
  }
  
  // Results
  return (
    <div className="space-y-4">
      <div className="text-sm text-gray-400 mb-2">
        Found {papers.length} papers
      </div>
      
      {papers.map(paper => (
        <PaperCard key={paper.id} paper={paper} />
      ))}
    </div>
  );
};

export default PaperList;

import React from 'react';

/**
 * Welcome screen shown when no messages are present
 */
const WelcomeScreen = () => {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center p-4">
      <div className="bg-blue-600 bg-opacity-10 p-6 rounded-full mb-6">
        <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
        </svg>
      </div>
      
      <h2 className="text-2xl font-bold text-white mb-2">Welcome to arXiv CS Expert</h2>
      <p className="text-lg text-gray-300 mb-6">Your AI research assistant for computer science</p>
      
      <div className="max-w-lg text-gray-400 mb-8 space-y-4">
        <p>
          I can help you explore computer science research papers, explain complex concepts, 
          and keep up with the latest developments in the field.
        </p>
        <p>
          Ask me anything about algorithms, machine learning, computer vision, 
          natural language processing, or any other CS topic!
        </p>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 w-full max-w-2xl">
        <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
          <div className="text-blue-400 mb-2">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
          </div>
          <h3 className="font-medium text-white mb-1">Research Insights</h3>
          <p className="text-sm text-gray-400">Get explanations of complex CS concepts and research papers</p>
        </div>
        
        <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
          <div className="text-purple-400 mb-2">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z" />
            </svg>
          </div>
          <h3 className="font-medium text-white mb-1">Paper Search</h3>
          <p className="text-sm text-gray-400">Find relevant papers from arXiv's CS collection</p>
        </div>
        
        <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
          <div className="text-green-400 mb-2">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
            </svg>
          </div>
          <h3 className="font-medium text-white mb-1">Visualizations</h3>
          <p className="text-sm text-gray-400">Explore relationships between concepts and papers</p>
        </div>
      </div>
    </div>
  );
};

export default WelcomeScreen;

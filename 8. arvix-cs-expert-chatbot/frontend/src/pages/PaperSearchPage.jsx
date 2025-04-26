import React, { useState } from 'react';
import Layout from '../components/Layout/Layout';
import SearchForm from '../components/PaperSearch/SearchForm';
import PaperList from '../components/PaperSearch/PaperList';
import { usePaperSearch } from '../hooks/usePaperSearch';

/**
 * Paper search page
 */
const PaperSearchPage = () => {
  const { papers, loading, error, searchPapers } = usePaperSearch();
  const [searchPerformed, setSearchPerformed] = useState(false);
  
  const handleSearch = (searchParams) => {
    searchPapers(searchParams.query, {
      categories: searchParams.categories,
      dateRange: searchParams.dateRange,
      sortBy: searchParams.sortBy
    });
    setSearchPerformed(true);
  };
  
  return (
    <Layout>
      <div className="max-w-6xl mx-auto w-full p-4">
        <div className="mb-6">
          <h1 className="text-2xl font-bold text-white mb-2">Paper Search</h1>
          <p className="text-gray-400">
            Search for computer science research papers from arXiv's extensive collection.
          </p>
        </div>
        
        <SearchForm onSearch={handleSearch} loading={loading} />
        
        {searchPerformed ? (
          <PaperList papers={papers} loading={loading} error={error} />
        ) : (
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-8 text-center">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 text-gray-600 mx-auto mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <h2 className="text-xl font-medium text-white mb-2">Search for Papers</h2>
            <p className="text-gray-400 max-w-md mx-auto mb-6">
              Enter keywords, author names, or topics to find relevant computer science papers.
              Use filters to narrow down your search.
            </p>
            <div className="flex flex-wrap justify-center gap-2">
              <button
                onClick={() => handleSearch({ query: "machine learning" })}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-md text-sm text-white transition-colors"
              >
                Machine Learning
              </button>
              <button
                onClick={() => handleSearch({ query: "natural language processing" })}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-md text-sm text-white transition-colors"
              >
                NLP
              </button>
              <button
                onClick={() => handleSearch({ query: "computer vision" })}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-md text-sm text-white transition-colors"
              >
                Computer Vision
              </button>
            </div>
          </div>
        )}
      </div>
    </Layout>
  );
};

export default PaperSearchPage;

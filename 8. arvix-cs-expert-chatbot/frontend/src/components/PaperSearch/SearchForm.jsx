import React, { useState } from 'react';

/**
 * Paper search form component
 */
const SearchForm = ({ onSearch, loading }) => {
  const [query, setQuery] = useState('');
  const [showFilters, setShowFilters] = useState(false);
  const [selectedCategories, setSelectedCategories] = useState([]);
  const [dateRange, setDateRange] = useState('all');
  const [sortBy, setSortBy] = useState('relevance');
  
  // CS categories
  const categories = [
    { id: 'cs.AI', name: 'Artificial Intelligence' },
    { id: 'cs.CL', name: 'Computation and Language' },
    { id: 'cs.CV', name: 'Computer Vision' },
    { id: 'cs.LG', name: 'Machine Learning' },
    { id: 'cs.NE', name: 'Neural and Evolutionary Computing' },
    { id: 'cs.RO', name: 'Robotics' },
    { id: 'cs.CR', name: 'Cryptography and Security' },
    { id: 'cs.DS', name: 'Data Structures and Algorithms' },
    { id: 'cs.DB', name: 'Databases' },
    { id: 'cs.DC', name: 'Distributed Computing' },
    { id: 'cs.SE', name: 'Software Engineering' },
    { id: 'cs.HC', name: 'Human-Computer Interaction' }
  ];
  
  // Toggle category selection
  const toggleCategory = (categoryId) => {
    if (selectedCategories.includes(categoryId)) {
      setSelectedCategories(selectedCategories.filter(id => id !== categoryId));
    } else {
      setSelectedCategories([...selectedCategories, categoryId]);
    }
  };
  
  // Clear all filters
  const clearFilters = () => {
    setSelectedCategories([]);
    setDateRange('all');
    setSortBy('relevance');
  };
  
  // Handle form submission
  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (query.trim()) {
      onSearch({
        query,
        categories: selectedCategories,
        dateRange,
        sortBy
      });
    }
  };
  
  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700 p-4 mb-6">
      <form onSubmit={handleSubmit}>
        {/* Search input */}
        <div className="flex flex-col md:flex-row gap-2">
          <div className="flex-1">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search for papers, authors, or topics..."
              className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-md text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              disabled={loading}
            />
          </div>
          <button
            type="submit"
            className="btn btn-primary"
            disabled={!query.trim() || loading}
          >
            {loading ? (
              <div className="flex items-center">
                <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Searching...
              </div>
            ) : (
              <div className="flex items-center">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
                Search
              </div>
            )}
          </button>
        </div>
        
        {/* Filters toggle */}
        <div className="mt-3 flex justify-between items-center">
          <button
            type="button"
            onClick={() => setShowFilters(!showFilters)}
            className="text-sm text-blue-400 hover:text-blue-300 flex items-center"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z" />
            </svg>
            {showFilters ? 'Hide Filters' : 'Show Filters'}
          </button>
          
          {selectedCategories.length > 0 || dateRange !== 'all' || sortBy !== 'relevance' ? (
            <button
              type="button"
              onClick={clearFilters}
              className="text-sm text-gray-400 hover:text-gray-300"
            >
              Clear Filters
            </button>
          ) : null}
        </div>
        
        {/* Filters */}
        {showFilters && (
          <div className="mt-4 border-t border-gray-700 pt-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Categories */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Categories</label>
                <div className="max-h-48 overflow-y-auto bg-gray-700 rounded-md p-2 space-y-1">
                  {categories.map(category => (
                    <div key={category.id} className="flex items-center">
                      <input
                        type="checkbox"
                        id={category.id}
                        checked={selectedCategories.includes(category.id)}
                        onChange={() => toggleCategory(category.id)}
                        className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-500 rounded"
                      />
                      <label htmlFor={category.id} className="ml-2 text-sm text-gray-300">
                        {category.name}
                      </label>
                    </div>
                  ))}
                </div>
              </div>
              
              {/* Date range */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Date Range</label>
                <select
                  value={dateRange}
                  onChange={(e) => setDateRange(e.target.value)}
                  className="w-full bg-gray-700 border border-gray-600 rounded-md text-white py-2 px-3 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                >
                  <option value="all">All Time</option>
                  <option value="1w">Past Week</option>
                  <option value="1m">Past Month</option>
                  <option value="3m">Past 3 Months</option>
                  <option value="1y">Past Year</option>
                  <option value="3y">Past 3 Years</option>
                </select>
              </div>
              
              {/* Sort by */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Sort By</label>
                <select
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value)}
                  className="w-full bg-gray-700 border border-gray-600 rounded-md text-white py-2 px-3 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                >
                  <option value="relevance">Relevance</option>
                  <option value="date-desc">Newest First</option>
                  <option value="date-asc">Oldest First</option>
                  <option value="citations">Most Cited</option>
                </select>
              </div>
            </div>
          </div>
        )}
      </form>
    </div>
  );
};

export default SearchForm;

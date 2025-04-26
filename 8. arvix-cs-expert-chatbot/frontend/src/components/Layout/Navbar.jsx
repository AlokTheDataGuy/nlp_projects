import React from 'react';
import { Link, useLocation } from 'react-router-dom';

/**
 * Navbar component for the top navigation
 */
const Navbar = () => {
  const location = useLocation();
  const currentPath = location.pathname;
  
  // Get the page title based on the current path
  const getPageTitle = () => {
    switch (currentPath) {
      case '/':
        return 'Chat';
      case '/search':
        return 'Paper Search';
      case '/visualization':
        return 'Concept Visualization';
      default:
        return 'arXiv CS Expert';
    }
  };
  
  return (
    <nav className="bg-gray-800 border-b border-gray-700">
      <div className="container-custom">
        <div className="flex items-center justify-between h-16">
          {/* Logo and title */}
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <span className="text-blue-500 font-bold text-xl">arXiv</span>
              <span className="text-white font-bold text-xl ml-1">CS Expert</span>
            </div>
            <div className="hidden md:block ml-6">
              <div className="flex items-center space-x-4">
                <Link
                  to="/"
                  className={`nav-link ${currentPath === '/' ? 'nav-link-active' : 'nav-link-inactive'}`}
                >
                  Chat
                </Link>
                <Link
                  to="/search"
                  className={`nav-link ${currentPath === '/search' ? 'nav-link-active' : 'nav-link-inactive'}`}
                >
                  Paper Search
                </Link>
                <Link
                  to="/visualization"
                  className={`nav-link ${currentPath === '/visualization' ? 'nav-link-active' : 'nav-link-inactive'}`}
                >
                  Visualization
                </Link>
              </div>
            </div>
          </div>
          
          {/* Page title (mobile) */}
          <div className="md:hidden">
            <h1 className="text-white font-medium">{getPageTitle()}</h1>
          </div>
          
          {/* Right side actions */}
          <div className="flex items-center">
            <button
              className="btn-icon text-gray-300 hover:text-white"
              aria-label="Toggle theme"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;

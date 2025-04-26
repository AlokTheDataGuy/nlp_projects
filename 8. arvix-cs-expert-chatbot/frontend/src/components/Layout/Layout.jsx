import React from 'react';
import Navbar from './Navbar';
import Sidebar from './Sidebar';

/**
 * Main layout component that wraps all pages
 */
const Layout = ({ children }) => {
  return (
    <div className="flex h-full bg-gray-900">
      {/* Sidebar */}
      <Sidebar />
      
      {/* Main content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <Navbar />
        <main className="flex-1 overflow-auto bg-gray-900">
          {children}
        </main>
      </div>
    </div>
  );
};

export default Layout;

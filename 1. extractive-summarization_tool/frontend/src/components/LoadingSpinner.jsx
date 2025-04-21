import React from 'react';

const LoadingSpinner = () => {
  return (
    <div className="loading-container">
      <div className="spinner">
        <div className="spinner-inner"></div>
      </div>
      <p className="loading-text">Generating summary...</p>
      <p className="loading-subtext">This may take a moment as we process and analyze your document</p>
    </div>
  );
};

export default LoadingSpinner;
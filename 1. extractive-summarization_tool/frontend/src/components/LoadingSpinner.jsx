import React from 'react';

const LoadingSpinner = () => {
  return (
    <div className="loading-container">
      <div className="spinner">
        <div className="spinner-inner"></div>
      </div>
      <p className="loading-text">Generating summary...</p>
      <p className="loading-subtext">This may take a moment as we analyze your text</p>
    </div>
  );
};

export default LoadingSpinner;
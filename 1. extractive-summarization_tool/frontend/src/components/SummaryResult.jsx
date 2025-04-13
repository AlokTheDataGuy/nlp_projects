import React, { useState } from 'react';

const SummaryResult = ({ summary, onDownload }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(summary);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="summary-result-section">
      <h2 className="section-title">Summary Result</h2>
      
      <div className="result-container">
        <div className="result-header">
          <div className="result-stats">
            <span className="stat">
              <strong>{summary.split(' ').length}</strong> words
            </span>
            <span className="stat">
              <strong>{summary.split(/[.!?]+/).filter(Boolean).length}</strong> sentences
            </span>
          </div>
          
          <div className="result-actions">
            <button 
              className="action-button copy-button"
              onClick={handleCopy}
            >
              {copied ? 'Copied!' : 'Copy'}
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
              </svg>
            </button>
            
            <button 
              className="action-button download-button"
              onClick={onDownload}
            >
              Download
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="7 10 12 15 17 10"></polyline>
                <line x1="12" y1="15" x2="12" y2="3"></line>
              </svg>
            </button>
          </div>
        </div>
        
        <div className="result-content">
          {summary}
        </div>
      </div>
    </div>
  );
};

export default SummaryResult;
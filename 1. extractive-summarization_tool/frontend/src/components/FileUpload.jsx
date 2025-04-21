import React, { useRef } from 'react';

const FileUpload = ({ onFileChange, fileName }) => {
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.add('dragover');
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.remove('dragover');
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.remove('dragover');

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      const fileType = file.type;
      const fileExtension = file.name.split('.').pop().toLowerCase();

      if (fileType === 'text/plain' || fileType === 'application/pdf' ||
          fileExtension === 'txt' || fileExtension === 'pdf') {
        onFileChange(file);
      } else {
        alert('Please upload a .txt or .pdf file');
      }
    }
  };

  const handleFileInputChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      onFileChange(e.target.files[0]);
    }
  };

  const handleButtonClick = () => {
    fileInputRef.current.click();
  };

  return (
    <div className="file-upload-section">
      <h2 className="section-title">Upload Document</h2>

      <div
        className="drop-area"
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleButtonClick}
      >
        <input
          type="file"
          accept=".txt,.pdf,application/pdf,text/plain"
          ref={fileInputRef}
          onChange={handleFileInputChange}
          style={{ display: 'none' }}
        />

        <div className="drop-content">
          <div className="upload-icon">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="17 8 12 3 7 8"></polyline>
              <line x1="12" y1="3" x2="12" y2="15"></line>
            </svg>
          </div>

          {fileName ? (
            <p className="file-name">{fileName}</p>
          ) : (
            <>
              <p className="drop-text">Drag & drop your document here</p>
              <p className="drop-subtext">- or -</p>
              <button className="browse-button">Browse files</button>
              <p className="file-type-hint">.txt and .pdf files are supported</p>
            </>
          )}
        </div>
      </div>

      {fileName && (
        <button
          className="clear-button"
          onClick={() => onFileChange(null)}
        >
          Clear selection
        </button>
      )}
    </div>
  );
};

export default FileUpload;
import React, { useState } from 'react';
import './App.css';
import FileUpload from './components/FileUpload';
import SummaryControls from './components/SummaryControls';
import SummaryResult from './components/SummaryResult';
import LoadingSpinner from './components/LoadingSpinner';
import ErrorMessage from './components/ErrorMessage';

function App() {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState('');
  const [summary, setSummary] = useState('');
  const [loading, setLoading] = useState(false);
  const [downloadInfo, setDownloadInfo] = useState(null);
  const [error, setError] = useState('');
  const [settings, setSettings] = useState({
    ratio: 0.3,
    min: 3,
    max: 10
  });

  const handleFileChange = (selectedFile) => {
    setFile(selectedFile);
    setFileName(selectedFile ? selectedFile.name : '');
    setSummary('');
    setDownloadInfo(null);
    setError('');
  };

  const handleSettingsChange = (newSettings) => {
    setSettings(newSettings);
  };

  const handleSubmit = async () => {
    if (!file) return;

    setLoading(true);
    setError('');
    const formData = new FormData();
    formData.append('file', file);
    formData.append('ratio', settings.ratio);
    formData.append('min', settings.min);
    formData.append('max', settings.max);

    try {
      const response = await fetch('http://localhost:5000/api/summarize', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (data.success) {
        setSummary(data.summary);
        setDownloadInfo({
          filePath: data.file_path,
          downloadName: data.download_name
        });
      } else {
        setError(data.error || 'Error generating summary');
      }
    } catch (error) {
      console.error('Error:', error);
      setError('Failed to connect to the server. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    if (!downloadInfo) return;

    window.location.href = `http://localhost:5000/api/download/${encodeURIComponent(downloadInfo.filePath)}/${encodeURIComponent(downloadInfo.downloadName)}`;
  };

  return (
    <div className="app-container">
      <main className="main-content">
        <div className="container">
          <h1 className="app-title">AI Text Summarizer</h1>
          <p className="app-description">
            Upload your text or PDF file and get an intelligent summary powered by BERT
          </p>

          <div className="card">
            <FileUpload
              onFileChange={handleFileChange}
              fileName={fileName}
            />

            <SummaryControls
              settings={settings}
              onSettingsChange={handleSettingsChange}
              onSubmit={handleSubmit}
              disabled={!file || loading}
            />

            {error && (
              <ErrorMessage
                message={error}
                onDismiss={() => setError('')}
              />
            )}

            {loading ? (
              <LoadingSpinner />
            ) : (
              summary && (
                <SummaryResult
                  summary={summary}
                  onDownload={handleDownload}
                />
              )
            )}
          </div>
        </div>
      </main>
      <footer className="footer">
        <p>© {new Date().getFullYear()} AI Text Summarizer. All rights reserved.</p>
      </footer>
    </div>
  );
}

export default App;
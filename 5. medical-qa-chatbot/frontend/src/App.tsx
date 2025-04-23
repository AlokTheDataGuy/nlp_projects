import { useState, useEffect } from 'react';
import ChatContainer from './components/ChatContainer';
import { checkHealth } from './services/api';

function App() {
  const [apiStatus, setApiStatus] = useState<'loading' | 'online' | 'offline'>('loading');

  useEffect(() => {
    const checkApiStatus = async () => {
      try {
        await checkHealth();
        setApiStatus('online');
      } catch (error) {
        console.error('API is offline:', error);
        setApiStatus('offline');
      }
    };

    checkApiStatus();
    // Check API status every 30 seconds
    const interval = setInterval(checkApiStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-gray-100 py-6 px-4 flex items-center justify-center">
      {apiStatus === 'loading' && (
        <div className="text-center">
          <p className="text-lg">Connecting to the API...</p>
          <div className="mt-4 flex justify-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          </div>
        </div>
      )}

      {apiStatus === 'offline' && (
        <div className="max-w-md mx-auto bg-white p-6 rounded-lg shadow-md text-center">
          <h1 className="text-xl font-bold text-red-600 mb-4">API Connection Error</h1>
          <p className="mb-4">
            Unable to connect to the API. Please make sure the backend server is running.
          </p>
          <button
            onClick={() => setApiStatus('loading')}
            className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
          >
            Retry Connection
          </button>
        </div>
      )}

      {apiStatus === 'online' && <ChatContainer />}
    </div>
  );
}

export default App;

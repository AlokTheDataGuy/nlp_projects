import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ChatProvider } from './context/ChatContext';
import { PaperSearchProvider } from './context/PaperSearchContext';
import ChatPage from './pages/ChatPage';
import PaperSearchPage from './pages/PaperSearchPage';
import VisualizationPage from './pages/VisualizationPage';

/**
 * Main App component with routing
 */
const App = () => {
  return (
    <Router>
      <ChatProvider>
        <PaperSearchProvider>
          <Routes>
            <Route path="/" element={<ChatPage />} />
            <Route path="/search" element={<PaperSearchPage />} />
            <Route path="/visualization" element={<VisualizationPage />} />
          </Routes>
        </PaperSearchProvider>
      </ChatProvider>
    </Router>
  );
};

export default App;
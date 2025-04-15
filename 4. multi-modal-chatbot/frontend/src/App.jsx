import { useState } from 'react'
import ChatInterface from './components/ChatInterface'
import MessageList from './components/MessageList'
import './App.css'

function App() {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const addMessage = (message) => {
    setMessages(prevMessages => [...prevMessages, message]);
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Multi-Modal Chatbot</h1>
        <p>Chat with text and images using AI</p>
      </header>

      <main className="app-main">
        <MessageList messages={messages} loading={loading} />
        <ChatInterface
          onSendMessage={addMessage}
          setLoading={setLoading}
          messages={messages}
        />
      </main>

      <footer className="app-footer">
        <p>Powered by llama3.2-vision:11b for image understanding and qwen2.5:7b for text processing</p>
      </footer>
    </div>
  )
}

export default App
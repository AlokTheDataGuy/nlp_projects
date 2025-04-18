const App = () => {
  const [messages, setMessages] = React.useState([
    {
      type: 'assistant',
      text: 'Hello! I\'m a medical Q&A chatbot. Ask me any medical questions, and I\'ll try to provide information from trusted medical sources.',
      timestamp: new Date(),
    }
  ]);
  const [isLoading, setIsLoading] = React.useState(false);
  const [isDarkMode, setIsDarkMode] = React.useState(
    window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches
  );

  // Apply dark mode class to body
  React.useEffect(() => {
    if (isDarkMode) {
      document.body.classList.add('dark-theme');
    } else {
      document.body.classList.remove('dark-theme');
    }
  }, [isDarkMode]);

  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode);
  };

  const handleSendMessage = async (text) => {
    // Add user message to chat
    const userMessage = {
      type: 'user',
      text,
      timestamp: new Date(),
    };

    setMessages(prevMessages => [...prevMessages, userMessage]);
    setIsLoading(true);

    try {
      // Send request to backend
      const response = await fetch('/api/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: text }),
      });

      if (!response.ok) {
        throw new Error('Failed to get response');
      }

      const data = await response.json();

      // Get the best answer
      const bestAnswer = data.answers && data.answers.length > 0
        ? data.answers[0]
        : null;

      // Create assistant message
      const assistantMessage = {
        type: 'assistant',
        text: bestAnswer
          ? bestAnswer.answer
          : "I'm sorry, I couldn't find a good answer to your question.",
        entities: data.entities,
        entity_labels: data.entity_labels,
        abbreviations: data.abbreviations,
        source: bestAnswer ? bestAnswer.source : null,
        timestamp: new Date(),
      };

      // Add assistant message to chat
      setMessages(prevMessages => [...prevMessages, assistantMessage]);
    } catch (error) {
      console.error('Error:', error);

      // Add error message
      const errorMessage = {
        type: 'assistant',
        text: 'Sorry, there was an error processing your request. Please try again.',
        timestamp: new Date(),
      };

      setMessages(prevMessages => [...prevMessages, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <Header toggleTheme={toggleTheme} isDarkMode={isDarkMode} />
      <MessageList messages={messages} isLoading={isLoading} />
      <ChatInterface onSendMessage={handleSendMessage} isLoading={isLoading} />
    </div>
  );
};

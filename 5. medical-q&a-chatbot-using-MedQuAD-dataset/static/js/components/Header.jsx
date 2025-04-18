const Header = ({ toggleTheme, isDarkMode }) => {
  return (
    <header className="header">
      <button className="theme-toggle" onClick={toggleTheme}>
        {isDarkMode ? '☀️' : '🌙'}
      </button>
      <h1>Medical Q&A Chatbot</h1>
      <p>Ask medical questions and get answers from trusted sources</p>
    </header>
  );
};

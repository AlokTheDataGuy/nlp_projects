import React, { useState, useRef, useEffect } from 'react';
import { 
  Box, 
  TextField, 
  Button, 
  Typography, 
  Paper, 
  List, 
  ListItem, 
  ListItemText, 
  Divider, 
  Link,
  CircularProgress
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import ReactMarkdown from 'react-markdown';
import { sendChatMessage } from '../services/api';
import { ChatMessage, PaperReference } from '../types/chat';

const ChatPage: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [paperReferences, setPaperReferences] = useState<PaperReference[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom of messages
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  const handleSendMessage = async () => {
    if (!input.trim()) return;

    // Add user message
    const userMessage: ChatMessage = {
      role: 'user',
      content: input,
    };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      // Send message to API
      const response = await sendChatMessage(input, messages);
      
      // Add assistant message
      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: response.response,
      };
      setMessages((prev) => [...prev, assistantMessage]);
      
      // Update paper references
      setPaperReferences(response.papers);
    } catch (error) {
      console.error('Error sending message:', error);
      
      // Add error message
      const errorMessage: ChatMessage = {
        role: 'assistant',
        content: 'Sorry, I encountered an error while processing your request. Please try again.',
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <Box className="chat-container">
      <Box className="chat-messages">
        {messages.map((message, index) => (
          <Paper
            key={index}
            className={`message ${message.role === 'user' ? 'user-message' : 'assistant-message'}`}
            elevation={1}
          >
            {message.role === 'assistant' ? (
              <ReactMarkdown>{message.content}</ReactMarkdown>
            ) : (
              <Typography>{message.content}</Typography>
            )}
          </Paper>
        ))}
        {loading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
            <CircularProgress size={24} />
          </Box>
        )}
        <div ref={messagesEndRef} />
        
        <Box className="chat-input">
          <TextField
            fullWidth
            multiline
            maxRows={4}
            variant="outlined"
            placeholder="Ask about computer science research papers..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={loading}
          />
          <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 1 }}>
            <Button
              variant="contained"
              color="primary"
              endIcon={<SendIcon />}
              onClick={handleSendMessage}
              disabled={!input.trim() || loading}
            >
              Send
            </Button>
          </Box>
        </Box>
      </Box>
      
      <Box className="paper-references">
        <Typography variant="h6" gutterBottom>
          Paper References
        </Typography>
        <Divider />
        {paperReferences.length > 0 ? (
          <List>
            {paperReferences.map((paper, index) => (
              <React.Fragment key={paper.paper_id}>
                <ListItem alignItems="flex-start">
                  <ListItemText
                    primary={
                      <Link href={paper.url} target="_blank" rel="noopener">
                        {paper.title}
                      </Link>
                    }
                    secondary={
                      <>
                        <Typography component="span" variant="body2" color="text.primary">
                          {paper.authors}
                        </Typography>
                        <br />
                        {paper.published_date} • {paper.categories}
                      </>
                    }
                  />
                </ListItem>
                {index < paperReferences.length - 1 && <Divider component="li" />}
              </React.Fragment>
            ))}
          </List>
        ) : (
          <Typography variant="body2" color="text.secondary" sx={{ mt: 2, textAlign: 'center' }}>
            No paper references yet. Start a conversation to see relevant papers.
          </Typography>
        )}
      </Box>
    </Box>
  );
};

export default ChatPage;

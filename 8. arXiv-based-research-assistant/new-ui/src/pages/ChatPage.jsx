import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  List,
  ListItem,
  ListItemText,
  CircularProgress
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';

const ChatPage = () => {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hello! I\'m your Computer Science research assistant. Ask me anything about CS research papers from arXiv.' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSend = () => {
    if (!input.trim()) return;

    // Add user message
    setMessages([...messages, { role: 'user', content: input }]);
    
    // Clear input and set loading
    setInput('');
    setLoading(true);
    
    // Simulate API call
    setTimeout(() => {
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: 'This is a simulated response. The backend API is not connected yet.' 
      }]);
      setLoading(false);
    }, 1500);
  };

  return (
    <Box sx={{ maxWidth: 800, mx: 'auto' }}>
      <Typography variant="h4" gutterBottom>
        Chat with arXiv Research Assistant
      </Typography>
      
      <Paper sx={{ p: 2, mb: 2, maxHeight: 400, overflow: 'auto' }}>
        <List>
          {messages.map((message, index) => (
            <ListItem key={index} sx={{ 
              justifyContent: message.role === 'user' ? 'flex-end' : 'flex-start',
              mb: 1
            }}>
              <Paper sx={{ 
                p: 2, 
                maxWidth: '80%',
                bgcolor: message.role === 'user' ? 'primary.main' : 'grey.100',
                color: message.role === 'user' ? 'white' : 'text.primary'
              }}>
                <ListItemText primary={message.content} />
              </Paper>
            </ListItem>
          ))}
          {loading && (
            <ListItem sx={{ justifyContent: 'flex-start' }}>
              <Paper sx={{ p: 2, bgcolor: 'grey.100' }}>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <CircularProgress size={20} sx={{ mr: 1 }} />
                  <Typography>Thinking...</Typography>
                </Box>
              </Paper>
            </ListItem>
          )}
        </List>
      </Paper>
      
      <Box sx={{ display: 'flex' }}>
        <TextField
          fullWidth
          variant="outlined"
          placeholder="Ask a question..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          sx={{ mr: 1 }}
        />
        <Button 
          variant="contained" 
          color="primary" 
          endIcon={<SendIcon />}
          onClick={handleSend}
          disabled={!input.trim() || loading}
        >
          Send
        </Button>
      </Box>
    </Box>
  );
};

export default ChatPage;
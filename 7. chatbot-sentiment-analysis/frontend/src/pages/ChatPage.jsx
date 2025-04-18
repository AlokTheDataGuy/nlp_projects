import React, { useState, useEffect, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import Alert from '@mui/material/Alert';
import Snackbar from '@mui/material/Snackbar';
import CircularProgress from '@mui/material/CircularProgress';
import useMediaQuery from '@mui/material/useMediaQuery';
import { useTheme } from '@mui/material/styles';

import ChatMessage from '../components/ChatMessage.jsx';
import ChatInput from '../components/ChatInput.jsx';
import SentimentIndicator from '../components/SentimentIndicator.jsx';
import SentimentTrend from '../components/SentimentTrend.jsx';
import { sendMessage } from '../services/chatService.jsx';

const ChatPage = () => {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [conversationId, setConversationId] = useState(uuidv4());
  const [currentSentiment, setCurrentSentiment] = useState(null);
  const [sentimentHistory, setSentimentHistory] = useState([]);
  const [escalated, setEscalated] = useState(false);

  const messagesEndRef = useRef(null);
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  // Scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (content) => {
    try {
      setLoading(true);
      setError(null);

      // Add user message to chat
      const userMessage = {
        id: uuidv4(),
        role: 'user',
        content,
        timestamp: new Date().toISOString()
      };

      setMessages(prev => [...prev, userMessage]);

      // Send message to API
      const response = await sendMessage(content, conversationId);

      // Update conversation ID if it's a new conversation
      if (!conversationId) {
        setConversationId(response.conversation_id);
      }

      // Add bot response to chat
      const botMessage = {
        id: uuidv4(),
        role: 'bot',
        content: response.response,
        timestamp: new Date().toISOString()
      };

      setMessages(prev => [...prev, botMessage]);

      // Update current sentiment
      setCurrentSentiment(response.sentiment);

      // Update sentiment history
      setSentimentHistory(prev => [
        ...prev,
        {
          timestamp: new Date().toISOString(),
          sentiment: response.sentiment
        }
      ]);

      // Check if conversation should be escalated
      if (response.escalate && !escalated) {
        setEscalated(true);
      }

    } catch (err) {
      console.error('Error sending message:', err);
      setError('Failed to send message. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ maxWidth: 800, mx: 'auto', mt: 4 }}>

      {currentSentiment && (
        <SentimentIndicator sentiment={currentSentiment} />
      )}

      {escalated && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          This conversation has been flagged for escalation due to negative sentiment.
          A customer support representative will be notified.
        </Alert>
      )}

      <Paper
        elevation={3}
        sx={{
          height: isMobile ? 'calc(100vh - 300px)' : 400,
          p: 2,
          mb: 2,
          display: 'flex',
          flexDirection: 'column'
        }}
      >
        <Box
          sx={{
            flex: 1,
            overflowY: 'auto',
            mb: 2,
            display: 'flex',
            flexDirection: 'column'
          }}
        >
          {messages.length === 0 ? (
            <Typography
              variant="body1"
              color="text.secondary"
              sx={{ textAlign: 'center', mt: 10 }}
            >
              Start a conversation by sending a message!
            </Typography>
          ) : (
            messages.map(message => (
              <ChatMessage
                key={message.id}
                message={message}
                isUser={message.role === 'user'}
              />
            ))
          )}
          <div ref={messagesEndRef} />
        </Box>

        <ChatInput onSendMessage={handleSendMessage} disabled={loading} />

        {loading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
            <CircularProgress size={24} />
          </Box>
        )}
      </Paper>

      {/* {sentimentHistory.length > 1 && (
        <Paper elevation={2} sx={{ p: 2, mb: 2 }}>
          <Typography variant="h6" gutterBottom>
            Sentiment Trend
          </Typography>
          <SentimentTrend sentimentHistory={sentimentHistory} />
        </Paper>
      )} */}

      <Snackbar
        open={!!error}
        autoHideDuration={6000}
        onClose={() => setError(null)}
        message={error}
      />
    </Box>
  );
};

export default ChatPage;

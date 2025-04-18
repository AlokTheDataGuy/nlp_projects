import React from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Paper from '@mui/material/Paper';
import Tooltip from '@mui/material/Tooltip';

const ChatMessage = ({ message, isUser }) => {
  // Determine sentiment color if available
  const getSentimentColor = () => {
    if (!message.sentiment) return null;

    const label = message.sentiment.label;
    if (label === 'positive') return '#4caf50';
    if (label === 'negative') return '#f44336';
    return '#ffeb3b'; // neutral
  };

  const sentimentColor = getSentimentColor();
  const confidencePercent = message.sentiment ? Math.round(message.sentiment.confidence * 100) : 0;

  return (
    <Box
      sx={{
        display: 'flex',
        justifyContent: isUser ? 'flex-end' : 'flex-start',
        mb: 2
      }}
      className="message-bubble-container"
    >
      <Paper
        elevation={1}
        sx={{
          p: 2,
          maxWidth: '70%',
          backgroundColor: isUser ? 'primary.main' : 'grey.100',
          color: isUser ? 'white' : 'text.primary',
          borderRadius: 2,
          position: 'relative',
          '&:hover': {
            boxShadow: 2
          }
        }}
        className={`message-bubble ${isUser ? 'user-message' : 'bot-message'}`}
      >
        {sentimentColor && message.sentiment && (
          <Tooltip
            title={`Sentiment: ${message.sentiment.label} (${confidencePercent}% confidence)`}
            arrow
          >
            <Box
              sx={{
                display: 'inline-block',
                width: 12,
                height: 12,
                borderRadius: '50%',
                backgroundColor: sentimentColor,
                mr: 1
              }}
            />
          </Tooltip>
        )}
        <Typography variant="body1">{message.content}</Typography>
        <Typography variant="caption" sx={{ display: 'block', mt: 1, opacity: 0.7 }}>
          {new Date(message.timestamp).toLocaleTimeString()}
        </Typography>
      </Paper>
    </Box>
  );
};

export default ChatMessage;

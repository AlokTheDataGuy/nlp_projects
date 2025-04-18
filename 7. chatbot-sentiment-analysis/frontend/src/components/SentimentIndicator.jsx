import React from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Paper from '@mui/material/Paper';
import LinearProgress from '@mui/material/LinearProgress';

const SentimentIndicator = ({ sentiment }) => {
  if (!sentiment) return null;

  // Get colors for each sentiment
  const getColor = (label) => {
    switch (label) {
      case 'positive':
        return '#4caf50';
      case 'negative':
        return '#f44336';
      case 'neutral':
      default:
        return '#ffeb3b';
    }
  };

  // Get the current sentiment color
  const currentColor = getColor(sentiment.label);

  return (
    <Paper elevation={1} sx={{ p: 2, mb: 2 }}>
      <Typography variant="h6" gutterBottom>
        Current Sentiment: {sentiment.label.charAt(0).toUpperCase() + sentiment.label.slice(1)}
      </Typography>

      <Box sx={{ mb: 2 }}>
        <Typography variant="body2" gutterBottom>
          Confidence: {Math.round(sentiment.confidence * 100)}%
        </Typography>
        <LinearProgress
          variant="determinate"
          value={sentiment.confidence * 100}
          sx={{
            height: 10,
            borderRadius: 5,
            backgroundColor: 'grey.300',
            '& .MuiLinearProgress-bar': {
              backgroundColor: currentColor
            }
          }}
        />
      </Box>

      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
        {Object.entries(sentiment.scores).map(([label, score]) => (
          <Box key={label} sx={{ textAlign: 'center', flex: 1 }}>
            <Box
              sx={{
                width: 20,
                height: 20,
                borderRadius: '50%',
                backgroundColor: getColor(label),
                margin: '0 auto',
                mb: 1
              }}
            />
            <Typography variant="body2">{label}</Typography>
            <Typography variant="body2">{Math.round(score * 100)}%</Typography>
          </Box>
        ))}
      </Box>
    </Paper>
  );
};

export default SentimentIndicator;

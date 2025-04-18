import React from 'react';
import { Line } from 'react-chartjs-2';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const SentimentTrend = ({ sentimentHistory }) => {
  if (!sentimentHistory || sentimentHistory.length < 2) {
    return (
      <Box sx={{ textAlign: 'center', py: 2 }}>
        <Typography variant="body2" color="text.secondary">
          Not enough data to display sentiment trend
        </Typography>
      </Box>
    );
  }

  // Format timestamps for display
  const formatTime = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  // Extract sentiment scores and timestamps
  const labels = sentimentHistory.map(item => formatTime(item.timestamp));
  const positiveScores = sentimentHistory.map(item => item.sentiment.scores.positive * 100);
  const neutralScores = sentimentHistory.map(item => item.sentiment.scores.neutral * 100);
  const negativeScores = sentimentHistory.map(item => item.sentiment.scores.negative * 100);

  const data = {
    labels,
    datasets: [
      {
        label: 'Positive',
        data: positiveScores,
        borderColor: '#4caf50',
        backgroundColor: 'rgba(76, 175, 80, 0.1)',
        tension: 0.4,
        fill: true
      },
      {
        label: 'Neutral',
        data: neutralScores,
        borderColor: '#ffeb3b',
        backgroundColor: 'rgba(255, 235, 59, 0.1)',
        tension: 0.4,
        fill: true
      },
      {
        label: 'Negative',
        data: negativeScores,
        borderColor: '#f44336',
        backgroundColor: 'rgba(244, 67, 54, 0.1)',
        tension: 0.4,
        fill: true
      }
    ]
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        title: {
          display: true,
          text: 'Score (%)'
        }
      },
      x: {
        grid: {
          display: false
        },
        title: {
          display: true,
          text: 'Time'
        }
      }
    },
    plugins: {
      legend: {
        position: 'top',
        labels: {
          boxWidth: 12,
          usePointStyle: true
        }
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        callbacks: {
          label: function(context) {
            return `${context.dataset.label}: ${context.parsed.y.toFixed(1)}%`;
          }
        }
      }
    },
    interaction: {
      mode: 'nearest',
      axis: 'x',
      intersect: false
    }
  };

  return (
    <Box sx={{ height: 200 }}>
      <Line data={data} options={options} />
    </Box>
  );
};

export default SentimentTrend;

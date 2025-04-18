import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  CircularProgress,
  Container,
  Divider,
  useMediaQuery,
  useTheme,
  IconButton,
  Tooltip
} from '@mui/material';
import { Bar, Pie, Line, Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip as ChartTooltip,
  Legend,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  PointElement,
  LineElement,
  Filler
} from 'chart.js';
import RefreshIcon from '@mui/icons-material/Refresh';
import { getMetrics } from '../services/metricsService.jsx';

// Register ChartJS components
ChartJS.register(
  ArcElement,
  ChartTooltip,
  Legend,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  PointElement,
  LineElement,
  Filler
);

const DashboardPage = () => {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [refreshing, setRefreshing] = useState(false);

  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  const fetchMetrics = async () => {
    try {
      setRefreshing(true);
      const data = await getMetrics();
      setMetrics(data);
      setError(null);
    } catch (err) {
      console.error('Error fetching metrics:', err);
      setError('Failed to load metrics data');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchMetrics();

    // Refresh metrics every 30 seconds
    const intervalId = setInterval(fetchMetrics, 30000);

    return () => clearInterval(intervalId);
  }, []);

  const handleRefresh = () => {
    fetchMetrics();
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '50vh' }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ mt: 4, textAlign: 'center' }}>
        <Typography color="error" variant="h6">{error}</Typography>
        <Box sx={{ mt: 2 }}>
          <IconButton onClick={handleRefresh} color="primary">
            <RefreshIcon />
          </IconButton>
        </Box>
      </Box>
    );
  }

  if (!metrics) {
    return (
      <Box sx={{ mt: 4, textAlign: 'center' }}>
        <Typography variant="h6">No metrics data available</Typography>
        <Box sx={{ mt: 2 }}>
          <IconButton onClick={handleRefresh} color="primary">
            <RefreshIcon />
          </IconButton>
        </Box>
      </Box>
    );
  }

  // Prepare sentiment distribution data for pie chart
  const sentimentData = {
    labels: ['Positive', 'Neutral', 'Negative'],
    datasets: [
      {
        data: [
          metrics.sentiment_distribution.positive,
          metrics.sentiment_distribution.neutral,
          metrics.sentiment_distribution.negative
        ],
        backgroundColor: ['rgba(76, 175, 80, 0.8)', 'rgba(255, 235, 59, 0.8)', 'rgba(244, 67, 54, 0.8)'],
        borderColor: ['#388e3c', '#fdd835', '#d32f2f'],
        borderWidth: 1,
      },
    ],
  };

  // Calculate escalation rate
  const escalationRate = metrics.total_conversations > 0
    ? (metrics.escalations / metrics.total_conversations * 100).toFixed(2)
    : 0;

  // Calculate sentiment percentages
  const totalSentiments =
    metrics.sentiment_distribution.positive +
    metrics.sentiment_distribution.neutral +
    metrics.sentiment_distribution.negative;

  const sentimentPercentages = {
    positive: totalSentiments > 0 ? (metrics.sentiment_distribution.positive / totalSentiments * 100).toFixed(1) : 0,
    neutral: totalSentiments > 0 ? (metrics.sentiment_distribution.neutral / totalSentiments * 100).toFixed(1) : 0,
    negative: totalSentiments > 0 ? (metrics.sentiment_distribution.negative / totalSentiments * 100).toFixed(1) : 0
  };

  // Calculate customer satisfaction score (CSAT)
  // Simple formula: (positive / total) * 100
  const csatScore = totalSentiments > 0
    ? (metrics.sentiment_distribution.positive / totalSentiments * 100).toFixed(1)
    : 0;

  // Calculate net sentiment score
  // Formula: (positive - negative) / total * 100
  const netSentimentScore = totalSentiments > 0
    ? ((metrics.sentiment_distribution.positive - metrics.sentiment_distribution.negative) / totalSentiments * 100).toFixed(1)
    : 0;

  // Simulated time-based data for sentiment trend
  // In a real app, this would come from the backend
  const sentimentTrendData = {
    labels: ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'],
    datasets: [
      {
        label: 'Positive',
        data: [12, 15, 18, 14, 20, 17, metrics.sentiment_distribution.positive],
        borderColor: '#4caf50',
        backgroundColor: 'rgba(76, 175, 80, 0.1)',
        tension: 0.4,
        fill: true
      },
      {
        label: 'Neutral',
        data: [8, 10, 6, 11, 8, 12, metrics.sentiment_distribution.neutral],
        borderColor: '#ffeb3b',
        backgroundColor: 'rgba(255, 235, 59, 0.1)',
        tension: 0.4,
        fill: true
      },
      {
        label: 'Negative',
        data: [5, 3, 7, 4, 6, 2, metrics.sentiment_distribution.negative],
        borderColor: '#f44336',
        backgroundColor: 'rgba(244, 67, 54, 0.1)',
        tension: 0.4,
        fill: true
      }
    ]
  };

  return (
    <Container maxWidth="xl" className="dashboard-container">
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">
          Performance Dashboard
        </Typography>
        <Tooltip title="Refresh data">
          <IconButton onClick={handleRefresh} disabled={refreshing}>
            {refreshing ? <CircularProgress size={24} /> : <RefreshIcon />}
          </IconButton>
        </Tooltip>
      </Box>

      <Grid container spacing={3}>
        {/* Summary Cards - First Row */}
        <Grid item xs={6} sm={6} md={3}>
          <Card className="dashboard-card" sx={{ bgcolor: 'primary.light', color: 'white' }}>
            <CardContent>
              <Typography variant={isMobile ? "subtitle2" : "subtitle1"} gutterBottom>
                Total Conversations
              </Typography>
              <Typography variant={isMobile ? "h5" : "h4"}>
                {metrics.total_conversations}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={6} sm={6} md={3}>
          <Card className="dashboard-card" sx={{ bgcolor: 'info.light', color: 'white' }}>
            <CardContent>
              <Typography variant={isMobile ? "subtitle2" : "subtitle1"} gutterBottom>
                Total Messages
              </Typography>
              <Typography variant={isMobile ? "h5" : "h4"}>
                {metrics.total_messages}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={6} sm={6} md={3}>
          <Card className="dashboard-card" sx={{ bgcolor: 'warning.light', color: 'white' }}>
            <CardContent>
              <Typography variant={isMobile ? "subtitle2" : "subtitle1"} gutterBottom>
                Escalations
              </Typography>
              <Typography variant={isMobile ? "h5" : "h4"}>
                {metrics.escalations}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={6} sm={6} md={3}>
          <Card className="dashboard-card" sx={{ bgcolor: 'error.light', color: 'white' }}>
            <CardContent>
              <Typography variant={isMobile ? "subtitle2" : "subtitle1"} gutterBottom>
                Escalation Rate
              </Typography>
              <Typography variant={isMobile ? "h5" : "h4"}>
                {escalationRate}%
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Visualization Cards - 2x2 Grid */}
        <Grid container spacing={3} sx={{ mt: 1 }}>
          {/* Customer Satisfaction Metrics */}
          <Grid item xs={12} md={6}>
            <Paper elevation={2} sx={{ p: 3, height: '100%' }}>
              <Typography variant="h6" gutterBottom>
                Customer Satisfaction
              </Typography>
              <Divider sx={{ mb: 2 }} />
              <Box sx={{ display: 'flex', justifyContent: 'center', mb: 2 }}>
                <Box sx={{
                  width: 180,
                  height: 180,
                  position: 'relative',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  <Doughnut
                    data={{
                      labels: ['CSAT Score', 'Remaining'],
                      datasets: [{
                        data: [csatScore, 100 - csatScore],
                        backgroundColor: [
                          csatScore >= 70 ? '#4caf50' : csatScore >= 40 ? '#ffeb3b' : '#f44336',
                          '#e0e0e0'
                        ],
                        borderWidth: 0,
                        cutout: '75%'
                      }]
                    }}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: {
                        legend: {
                          display: false
                        },
                        tooltip: {
                          enabled: false
                        }
                      }
                    }}
                  />
                  <Typography
                    variant="h4"
                    sx={{
                      position: 'absolute',
                      color: csatScore >= 70 ? '#4caf50' : csatScore >= 40 ? '#ffeb3b' : '#f44336',
                    }}
                  >
                    {csatScore}%
                  </Typography>
                </Box>
              </Box>
              <Typography variant="subtitle2" align="center" gutterBottom>
                CSAT Score
              </Typography>
              <Box sx={{ mt: 3 }}>
                <Typography variant="body2" gutterBottom>
                  Net Sentiment Score: {netSentimentScore}%
                </Typography>
                <Typography variant="body2" gutterBottom>
                  Positive Sentiment: {sentimentPercentages.positive}%
                </Typography>
                <Typography variant="body2" gutterBottom>
                  Neutral Sentiment: {sentimentPercentages.neutral}%
                </Typography>
                <Typography variant="body2" gutterBottom>
                  Negative Sentiment: {sentimentPercentages.negative}%
                </Typography>
              </Box>
            </Paper>
          </Grid>

          {/* Sentiment Distribution Chart */}
          <Grid item xs={12} md={6}>
            <Paper elevation={2} sx={{ p: 3, height: '100%' }}>
              <Typography variant="h6" gutterBottom>
                Sentiment Distribution
              </Typography>
              <Divider sx={{ mb: 2 }} />
              <Box sx={{ height: isMobile ? 200 : 300 }}>
                <Pie
                  data={sentimentData}
                  options={{
                    maintainAspectRatio: false,
                    plugins: {
                      legend: {
                        position: 'bottom'
                      }
                    }
                  }}
                />
              </Box>
            </Paper>
          </Grid>

          {/* Sentiment Trend Over Time */}
          <Grid item xs={12} md={6}>
            <Paper elevation={2} sx={{ p: 3, height: '100%' }}>
              <Typography variant="h6" gutterBottom>
                Sentiment Trend (Last 7 Days)
              </Typography>
              <Divider sx={{ mb: 2 }} />
              <Box sx={{ height: isMobile ? 200 : 300 }}>
                <Line
                  data={sentimentTrendData}
                  options={{
                    maintainAspectRatio: false,
                    scales: {
                      y: {
                        beginAtZero: true,
                        stacked: false
                      },
                      x: {
                        grid: {
                          display: false
                        }
                      }
                    },
                    plugins: {
                      legend: {
                        position: 'top'
                      }
                    },
                    interaction: {
                      mode: 'index',
                      intersect: false
                    }
                  }}
                />
              </Box>
            </Paper>
          </Grid>

          {/* Conversation Stats */}
          <Grid item xs={12} md={6}>
            <Paper elevation={2} sx={{ p: 3, height: '100%' }}>
              <Typography variant="h6" gutterBottom>
                Conversation Stats
              </Typography>
              <Divider sx={{ mb: 2 }} />

              <Grid container spacing={2} sx={{ mb: 3 }}>
                <Grid item xs={6}>
                  <Box sx={{ p: 2, bgcolor: 'background.default', borderRadius: 1, textAlign: 'center' }}>
                    <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                      Messages per Conversation
                    </Typography>
                    <Typography variant="h4" color="primary.main">
                      {metrics.total_conversations > 0
                        ? (metrics.total_messages / metrics.total_conversations).toFixed(1)
                        : '0'}
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={6}>
                  <Box sx={{ p: 2, bgcolor: 'background.default', borderRadius: 1, textAlign: 'center' }}>
                    <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                      Avg. Response Time
                    </Typography>
                    <Typography variant="h4" color="primary.main">
                      {/* Simulated data */}
                      15s
                    </Typography>
                  </Box>
                </Grid>
              </Grid>

              <Typography variant="subtitle2" gutterBottom>
                Messages by Sentiment
              </Typography>
              <Box sx={{ height: isMobile ? 180 : 220 }}>
                <Bar
                  data={{
                    labels: ['Positive', 'Neutral', 'Negative'],
                    datasets: [
                      {
                        label: 'Number of Messages',
                        data: [
                          metrics.sentiment_distribution.positive,
                          metrics.sentiment_distribution.neutral,
                          metrics.sentiment_distribution.negative
                        ],
                        backgroundColor: ['rgba(76, 175, 80, 0.8)', 'rgba(255, 235, 59, 0.8)', 'rgba(244, 67, 54, 0.8)'],
                      }
                    ]
                  }}
                  options={{
                    maintainAspectRatio: false,
                    scales: {
                      y: {
                        beginAtZero: true
                      }
                    },
                    plugins: {
                      legend: {
                        display: false
                      }
                    }
                  }}
                />
              </Box>
            </Paper>
          </Grid>
        </Grid>
      </Grid>
    </Container>
  );
};

export default DashboardPage;

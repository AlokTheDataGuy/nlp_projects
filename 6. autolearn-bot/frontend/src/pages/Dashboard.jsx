import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import Container from '@mui/material/Container'
import Grid from '@mui/material/Grid'
import Paper from '@mui/material/Paper'
import Typography from '@mui/material/Typography'
import Box from '@mui/material/Box'
import Card from '@mui/material/Card'
import CardContent from '@mui/material/CardContent'
import CardActions from '@mui/material/CardActions'
import Button from '@mui/material/Button'
import Divider from '@mui/material/Divider'
import SubscriptionsIcon from '@mui/icons-material/Subscriptions'
import VideoLibraryIcon from '@mui/icons-material/VideoLibrary'
import LightbulbIcon from '@mui/icons-material/Lightbulb'
import ChatIcon from '@mui/icons-material/Chat'

import { getChannels, getVideos, getInsights } from '../services/api'
import InsightCard from '../components/InsightCard'

export default function Dashboard() {
  const [stats, setStats] = useState({
    channels: 0,
    videos: 0,
    insights: 0,
  })
  const [latestInsights, setLatestInsights] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        
        // Fetch channels count
        const channelsResponse = await getChannels()
        
        // Fetch videos count
        const videosResponse = await getVideos({ limit: 1 })
        
        // Fetch insights count and latest insights
        const insightsResponse = await getInsights({ limit: 3 })
        
        setStats({
          channels: channelsResponse.data.length || 0,
          videos: videosResponse.data.length || 0,
          insights: insightsResponse.data.length || 0,
        })
        
        setLatestInsights(insightsResponse.data || [])
      } catch (error) {
        console.error('Error fetching dashboard data:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [])

  const statCards = [
    {
      title: 'Channels',
      value: stats.channels,
      icon: <SubscriptionsIcon fontSize="large" color="primary" />,
      link: '/channels',
    },
    {
      title: 'Videos',
      value: stats.videos,
      icon: <VideoLibraryIcon fontSize="large" color="primary" />,
      link: '/videos',
    },
    {
      title: 'Insights',
      value: stats.insights,
      icon: <LightbulbIcon fontSize="large" color="primary" />,
      link: '/insights',
    },
  ]

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom>
        Dashboard
      </Typography>
      
      <Grid container spacing={3}>
        {/* Stats Cards */}
        {statCards.map((card) => (
          <Grid item xs={12} sm={4} key={card.title}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <Typography variant="h6" color="text.secondary">
                      {card.title}
                    </Typography>
                    <Typography variant="h3">
                      {loading ? '...' : card.value}
                    </Typography>
                  </div>
                  {card.icon}
                </Box>
              </CardContent>
              <Divider />
              <CardActions>
                <Button size="small" component={Link} to={card.link}>
                  View All
                </Button>
              </CardActions>
            </Card>
          </Grid>
        ))}
        
        {/* Chat Card */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="h6" color="text.secondary">
                  Ask about the latest videos
                </Typography>
                <ChatIcon fontSize="large" color="primary" />
              </Box>
              <Typography variant="body1" sx={{ mt: 2 }}>
                Try asking: "What are the key takeaways from the latest videos?"
              </Typography>
            </CardContent>
            <Divider />
            <CardActions>
              <Button size="small" component={Link} to="/chat">
                Go to Chat
              </Button>
            </CardActions>
          </Card>
        </Grid>
        
        {/* Latest Insights */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Latest Insights
            </Typography>
            
            {loading ? (
              <Typography>Loading latest insights...</Typography>
            ) : latestInsights.length > 0 ? (
              <Grid container spacing={3}>
                {latestInsights.map((insight) => (
                  <Grid item xs={12} md={4} key={insight.id}>
                    <InsightCard insight={insight} />
                  </Grid>
                ))}
              </Grid>
            ) : (
              <Typography>No insights available yet.</Typography>
            )}
            
            <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end' }}>
              <Button component={Link} to="/insights">
                View All Insights
              </Button>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Container>
  )
}

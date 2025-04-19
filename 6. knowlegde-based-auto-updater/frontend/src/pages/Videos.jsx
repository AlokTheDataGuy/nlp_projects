import { useState, useEffect } from 'react'
import Container from '@mui/material/Container'
import Grid from '@mui/material/Grid'
import Paper from '@mui/material/Paper'
import Typography from '@mui/material/Typography'
import Card from '@mui/material/Card'
import CardContent from '@mui/material/CardContent'
import CardMedia from '@mui/material/CardMedia'
import CardActions from '@mui/material/CardActions'
import Button from '@mui/material/Button'
import Chip from '@mui/material/Chip'
import Box from '@mui/material/Box'
import FormControl from '@mui/material/FormControl'
import InputLabel from '@mui/material/InputLabel'
import Select from '@mui/material/Select'
import MenuItem from '@mui/material/MenuItem'
import Pagination from '@mui/material/Pagination'
import { formatDistanceToNow } from 'date-fns'

import { getVideos, getChannels, getInsightsByVideo } from '../services/api'

export default function Videos() {
  const [videos, setVideos] = useState([])
  const [channels, setChannels] = useState([])
  const [loading, setLoading] = useState(true)
  const [selectedChannel, setSelectedChannel] = useState('')
  const [page, setPage] = useState(1)
  const [totalPages, setTotalPages] = useState(1)
  const limit = 9

  const fetchVideos = async () => {
    try {
      setLoading(true)
      const params = {
        limit,
        skip: (page - 1) * limit,
      }
      
      if (selectedChannel) {
        params.channel_id = selectedChannel
      }
      
      const response = await getVideos(params)
      setVideos(response.data)
      
      // In a real app, we would get the total count from the API
      // For now, we'll just assume there are more pages if we get a full page
      setTotalPages(Math.ceil(response.data.length / limit) || 1)
    } catch (error) {
      console.error('Error fetching videos:', error)
    } finally {
      setLoading(false)
    }
  }

  const fetchChannels = async () => {
    try {
      const response = await getChannels()
      setChannels(response.data)
    } catch (error) {
      console.error('Error fetching channels:', error)
    }
  }

  useEffect(() => {
    fetchChannels()
  }, [])

  useEffect(() => {
    fetchVideos()
  }, [selectedChannel, page])

  const handleChannelChange = (event) => {
    setSelectedChannel(event.target.value)
    setPage(1)
  }

  const handlePageChange = (event, value) => {
    setPage(value)
  }

  // Function to get YouTube thumbnail URL
  const getYouTubeThumbnail = (videoId) => {
    return `https://img.youtube.com/vi/${videoId}/mqdefault.jpg`
  }

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Grid container spacing={3}>
        <Grid item xs={12} sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="h4">Videos</Typography>
          
          <FormControl sx={{ minWidth: 200 }}>
            <InputLabel id="channel-select-label">Filter by Channel</InputLabel>
            <Select
              labelId="channel-select-label"
              id="channel-select"
              value={selectedChannel}
              label="Filter by Channel"
              onChange={handleChannelChange}
            >
              <MenuItem value="">
                <em>All Channels</em>
              </MenuItem>
              {channels.map((channel) => (
                <MenuItem key={channel.id} value={channel.id}>
                  {channel.name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>
        
        {loading ? (
          <Grid item xs={12}>
            <Typography>Loading videos...</Typography>
          </Grid>
        ) : videos.length > 0 ? (
          <>
            {videos.map((video) => (
              <Grid item xs={12} sm={6} md={4} key={video.id}>
                <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                  <CardMedia
                    component="img"
                    height="140"
                    image={getYouTubeThumbnail(video.youtube_id)}
                    alt={video.title}
                  />
                  <CardContent sx={{ flexGrow: 1 }}>
                    <Typography gutterBottom variant="h6" component="div" noWrap>
                      {video.title}
                    </Typography>
                    
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                      {formatDistanceToNow(new Date(video.published_at), { addSuffix: true })}
                    </Typography>
                    
                    <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mt: 1 }}>
                      {video.transcript_processed && (
                        <Chip
                          label="Transcript"
                          size="small"
                          color="primary"
                          variant="outlined"
                        />
                      )}
                      {video.insights_extracted && (
                        <Chip
                          label="Insights"
                          size="small"
                          color="secondary"
                          variant="outlined"
                        />
                      )}
                    </Box>
                  </CardContent>
                  <CardActions>
                    <Button
                      size="small"
                      href={`https://www.youtube.com/watch?v=${video.youtube_id}`}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      Watch on YouTube
                    </Button>
                    {video.insights_extracted && (
                      <Button
                        size="small"
                        color="secondary"
                        onClick={() => {
                          // In a real app, we would navigate to a video insights page
                          // or open a dialog showing the insights
                          getInsightsByVideo(video.id)
                            .then((response) => console.log('Insights:', response.data))
                            .catch((error) => console.error('Error fetching insights:', error))
                        }}
                      >
                        View Insights
                      </Button>
                    )}
                  </CardActions>
                </Card>
              </Grid>
            ))}
            
            <Grid item xs={12} sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
              <Pagination
                count={totalPages}
                page={page}
                onChange={handlePageChange}
                color="primary"
              />
            </Grid>
          </>
        ) : (
          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Typography>No videos found.</Typography>
            </Paper>
          </Grid>
        )}
      </Grid>
    </Container>
  )
}

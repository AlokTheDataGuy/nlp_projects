import { useState, useEffect } from 'react'
import Container from '@mui/material/Container'
import Grid from '@mui/material/Grid'
import Paper from '@mui/material/Paper'
import Typography from '@mui/material/Typography'
import Box from '@mui/material/Box'
import FormControl from '@mui/material/FormControl'
import InputLabel from '@mui/material/InputLabel'
import Select from '@mui/material/Select'
import MenuItem from '@mui/material/MenuItem'
import TextField from '@mui/material/TextField'
import Pagination from '@mui/material/Pagination'
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns'
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider'
import { DatePicker } from '@mui/x-date-pickers/DatePicker'
import { subDays } from 'date-fns'

import { getInsights, getChannels } from '../services/api'
import InsightCard from '../components/InsightCard'

export default function Insights() {
  const [insights, setInsights] = useState([])
  const [channels, setChannels] = useState([])
  const [loading, setLoading] = useState(true)
  const [filters, setFilters] = useState({
    channel_id: '',
    topic: '',
    from_date: subDays(new Date(), 30),
    to_date: new Date(),
  })
  const [page, setPage] = useState(1)
  const [totalPages, setTotalPages] = useState(1)
  const limit = 9

  const fetchInsights = async () => {
    try {
      setLoading(true)
      
      const params = {
        limit,
        skip: (page - 1) * limit,
      }
      
      if (filters.channel_id) {
        params.channel_id = filters.channel_id
      }
      
      if (filters.topic) {
        params.topic = filters.topic
      }
      
      if (filters.from_date) {
        params.from_date = filters.from_date.toISOString()
      }
      
      if (filters.to_date) {
        params.to_date = filters.to_date.toISOString()
      }
      
      const response = await getInsights(params)
      setInsights(response.data)
      
      // In a real app, we would get the total count from the API
      // For now, we'll just assume there are more pages if we get a full page
      setTotalPages(Math.ceil(response.data.length / limit) || 1)
    } catch (error) {
      console.error('Error fetching insights:', error)
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
    fetchInsights()
  }, [filters, page])

  const handleFilterChange = (name, value) => {
    setFilters({
      ...filters,
      [name]: value,
    })
    setPage(1)
  }

  const handlePageChange = (event, value) => {
    setPage(value)
  }

  // Get unique topics from insights for the filter dropdown
  const uniqueTopics = [...new Set(insights.flatMap(insight => insight.topics || []))]

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Typography variant="h4" gutterBottom>
            Insights
          </Typography>
          
          <Paper sx={{ p: 2, mb: 3 }}>
            <Grid container spacing={2} alignItems="center">
              <Grid item xs={12} sm={3}>
                <FormControl fullWidth>
                  <InputLabel id="channel-select-label">Channel</InputLabel>
                  <Select
                    labelId="channel-select-label"
                    id="channel-select"
                    value={filters.channel_id}
                    label="Channel"
                    onChange={(e) => handleFilterChange('channel_id', e.target.value)}
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
              
              <Grid item xs={12} sm={3}>
                <FormControl fullWidth>
                  <InputLabel id="topic-select-label">Topic</InputLabel>
                  <Select
                    labelId="topic-select-label"
                    id="topic-select"
                    value={filters.topic}
                    label="Topic"
                    onChange={(e) => handleFilterChange('topic', e.target.value)}
                  >
                    <MenuItem value="">
                      <em>All Topics</em>
                    </MenuItem>
                    {uniqueTopics.map((topic) => (
                      <MenuItem key={topic} value={topic}>
                        {topic}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              
              <Grid item xs={12} sm={3}>
                <LocalizationProvider dateAdapter={AdapterDateFns}>
                  <DatePicker
                    label="From Date"
                    value={filters.from_date}
                    onChange={(date) => handleFilterChange('from_date', date)}
                    slotProps={{ textField: { fullWidth: true } }}
                  />
                </LocalizationProvider>
              </Grid>
              
              <Grid item xs={12} sm={3}>
                <LocalizationProvider dateAdapter={AdapterDateFns}>
                  <DatePicker
                    label="To Date"
                    value={filters.to_date}
                    onChange={(date) => handleFilterChange('to_date', date)}
                    slotProps={{ textField: { fullWidth: true } }}
                  />
                </LocalizationProvider>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
        
        {loading ? (
          <Grid item xs={12}>
            <Typography>Loading insights...</Typography>
          </Grid>
        ) : insights.length > 0 ? (
          <>
            {insights.map((insight) => (
              <Grid item xs={12} sm={6} md={4} key={insight.id}>
                <InsightCard insight={insight} />
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
              <Typography>No insights found matching your filters.</Typography>
            </Paper>
          </Grid>
        )}
      </Grid>
    </Container>
  )
}

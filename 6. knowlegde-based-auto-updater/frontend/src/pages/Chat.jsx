import { useState, useEffect, useRef } from 'react'
import Container from '@mui/material/Container'
import Grid from '@mui/material/Grid'
import Paper from '@mui/material/Paper'
import Typography from '@mui/material/Typography'
import Box from '@mui/material/Box'
import TextField from '@mui/material/TextField'
import Button from '@mui/material/Button'
import IconButton from '@mui/material/IconButton'
import SendIcon from '@mui/icons-material/Send'
import FilterListIcon from '@mui/icons-material/FilterList'
import Drawer from '@mui/material/Drawer'
import FormControl from '@mui/material/FormControl'
import InputLabel from '@mui/material/InputLabel'
import Select from '@mui/material/Select'
import MenuItem from '@mui/material/MenuItem'
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns'
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider'
import { DatePicker } from '@mui/x-date-pickers/DatePicker'
import { subDays } from 'date-fns'

import { sendChatMessage, getChannels } from '../services/api'
import ChatMessage from '../components/ChatMessage'

export default function Chat() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [channels, setChannels] = useState([])
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [filters, setFilters] = useState({
    channel_id: '',
    topic: '',
    from_date: subDays(new Date(), 30),
    to_date: new Date(),
  })
  
  const messagesEndRef = useRef(null)

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
    
    // Add a welcome message
    setMessages([
      {
        response: "Hello! I'm your YouTube content curator assistant. Ask me about insights from the videos I've analyzed.",
        sources: []
      }
    ])
  }, [])

  useEffect(() => {
    // Scroll to bottom when messages change
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSendMessage = async () => {
    if (!input.trim()) return
    
    const userMessage = {
      query: input,
      ...filters
    }
    
    // Add user message to chat
    setMessages([...messages, userMessage])
    setInput('')
    setLoading(true)
    
    try {
      const response = await sendChatMessage(userMessage)
      
      // Add bot response to chat
      setMessages((prevMessages) => [...prevMessages, response.data])
    } catch (error) {
      console.error('Error sending message:', error)
      
      // Add error message
      setMessages((prevMessages) => [
        ...prevMessages,
        {
          response: "Sorry, I encountered an error processing your request. Please try again.",
          sources: []
        }
      ])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const toggleDrawer = () => {
    setDrawerOpen(!drawerOpen)
  }

  const handleFilterChange = (name, value) => {
    setFilters({
      ...filters,
      [name]: value,
    })
  }

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4, height: 'calc(100vh - 100px)', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h4">Chat</Typography>
        <Button
          variant="outlined"
          startIcon={<FilterListIcon />}
          onClick={toggleDrawer}
        >
          Filters
        </Button>
      </Box>
      
      <Paper sx={{ p: 2, flexGrow: 1, display: 'flex', flexDirection: 'column', mb: 2, overflow: 'hidden' }}>
        <Box sx={{ flexGrow: 1, overflow: 'auto', mb: 2 }}>
          {messages.map((message, index) => (
            <ChatMessage
              key={index}
              message={message}
              isUser={!message.response}
            />
          ))}
          <div ref={messagesEndRef} />
        </Box>
        
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <TextField
            fullWidth
            variant="outlined"
            placeholder="Ask about insights from YouTube videos..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={loading}
            multiline
            maxRows={3}
          />
          <IconButton
            color="primary"
            onClick={handleSendMessage}
            disabled={!input.trim() || loading}
            sx={{ ml: 1 }}
          >
            <SendIcon />
          </IconButton>
        </Box>
      </Paper>
      
      {/* Filters Drawer */}
      <Drawer
        anchor="right"
        open={drawerOpen}
        onClose={toggleDrawer}
      >
        <Box sx={{ width: 300, p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Filter Results
          </Typography>
          
          <FormControl fullWidth sx={{ mt: 2 }}>
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
          
          <FormControl fullWidth sx={{ mt: 2 }}>
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
              <MenuItem value="AI">AI</MenuItem>
              <MenuItem value="Machine Learning">Machine Learning</MenuItem>
              <MenuItem value="Deep Learning">Deep Learning</MenuItem>
              <MenuItem value="Computer Vision">Computer Vision</MenuItem>
              <MenuItem value="NLP">NLP</MenuItem>
              <MenuItem value="Robotics">Robotics</MenuItem>
            </Select>
          </FormControl>
          
          <LocalizationProvider dateAdapter={AdapterDateFns}>
            <DatePicker
              label="From Date"
              value={filters.from_date}
              onChange={(date) => handleFilterChange('from_date', date)}
              slotProps={{ textField: { fullWidth: true, sx: { mt: 2 } } }}
            />
          </LocalizationProvider>
          
          <LocalizationProvider dateAdapter={AdapterDateFns}>
            <DatePicker
              label="To Date"
              value={filters.to_date}
              onChange={(date) => handleFilterChange('to_date', date)}
              slotProps={{ textField: { fullWidth: true, sx: { mt: 2 } } }}
            />
          </LocalizationProvider>
          
          <Button
            variant="contained"
            fullWidth
            sx={{ mt: 3 }}
            onClick={toggleDrawer}
          >
            Apply Filters
          </Button>
        </Box>
      </Drawer>
    </Container>
  )
}

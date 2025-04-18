import axios from 'axios'

const API_URL = 'http://localhost:8000/api'

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Channels
export const getChannels = () => api.get('/channels')
export const addChannel = (channel) => api.post('/channels', channel)
export const getChannel = (id) => api.get(`/channels/${id}`)
export const deleteChannel = (id) => api.delete(`/channels/${id}`)

// Videos
export const getVideos = (params) => api.get('/videos', { params })
export const getVideo = (id) => api.get(`/videos/${id}`)
export const getVideoTranscript = (id) => api.get(`/videos/${id}/transcript`)

// Insights
export const getInsights = (params) => api.get('/insights', { params })
export const getInsight = (id) => api.get(`/insights/${id}`)
export const getInsightsByVideo = (videoId) => api.get(`/insights/video/${videoId}`)

// Chat
export const sendChatMessage = (message) => api.post('/chat', message)

export default api

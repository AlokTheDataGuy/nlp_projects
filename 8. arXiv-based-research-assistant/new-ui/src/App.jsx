import { Routes, Route } from 'react-router-dom'
import { ThemeProvider } from '@mui/material/styles'
import CssBaseline from '@mui/material/CssBaseline'
import Layout from './components/Layout'
import ChatPage from './pages/ChatPage'
import SearchPage from './pages/SearchPage'
import VisualizationPage from './pages/VisualizationPage'
import NotFoundPage from './pages/NotFoundPage'
import theme from './theme'
import './App.css'

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Layout>
        <Routes>
          <Route path="/" element={<ChatPage />} />
          <Route path="/search" element={<SearchPage />} />
          <Route path="/visualization" element={<VisualizationPage />} />
          <Route path="*" element={<NotFoundPage />} />
        </Routes>
      </Layout>
    </ThemeProvider>
  )
}

export default App

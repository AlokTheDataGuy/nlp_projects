import { useState } from 'react'
import { styled } from '@mui/material/styles'
import Paper from '@mui/material/Paper'
import Typography from '@mui/material/Typography'
import Box from '@mui/material/Box'
import Collapse from '@mui/material/Collapse'
import Button from '@mui/material/Button'
import List from '@mui/material/List'
import ListItem from '@mui/material/ListItem'
import ListItemText from '@mui/material/ListItemText'
import ListItemIcon from '@mui/material/ListItemIcon'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import ExpandLessIcon from '@mui/icons-material/ExpandLess'
import VideoLibraryIcon from '@mui/icons-material/VideoLibrary'
import { formatDistanceToNow } from 'date-fns'

const MessageContainer = styled(Paper)(({ theme, isuser }) => ({
  padding: theme.spacing(2),
  marginBottom: theme.spacing(2),
  maxWidth: '80%',
  borderRadius: 16,
  ...(isuser === 'true'
    ? {
        marginLeft: 'auto',
        backgroundColor: theme.palette.primary.main,
        color: theme.palette.primary.contrastText,
      }
    : {
        marginRight: 'auto',
        backgroundColor: theme.palette.background.paper,
      }),
}))

export default function ChatMessage({ message, isUser }) {
  const [showSources, setShowSources] = useState(false)

  const toggleSources = () => {
    setShowSources(!showSources)
  }

  // Format the message with emoji and line breaks
  const formatMessage = (text) => {
    return text.split('\n').map((line, i) => (
      <Typography key={i} variant="body1" component="p">
        {line}
      </Typography>
    ))
  }

  return (
    <MessageContainer isuser={isUser ? 'true' : 'false'}>
      {isUser ? (
        <Typography variant="body1">{message.query}</Typography>
      ) : (
        <Box>
          {formatMessage(message.response)}
          
          {message.sources && message.sources.length > 0 && (
            <Box sx={{ mt: 2 }}>
              <Button
                size="small"
                onClick={toggleSources}
                endIcon={showSources ? <ExpandLessIcon /> : <ExpandMoreIcon />}
              >
                {showSources ? 'Hide Sources' : 'Show Sources'}
              </Button>
              
              <Collapse in={showSources}>
                <List dense>
                  {message.sources.map((source, index) => (
                    <ListItem key={index}>
                      <ListItemIcon>
                        <VideoLibraryIcon />
                      </ListItemIcon>
                      <ListItemText
                        primary={source.video_title}
                        secondary={`${source.channel_name} • ${formatDistanceToNow(new Date(source.published_at), { addSuffix: true })}`}
                      />
                    </ListItem>
                  ))}
                </List>
              </Collapse>
            </Box>
          )}
        </Box>
      )}
    </MessageContainer>
  )
}

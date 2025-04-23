import Card from '@mui/material/Card'
import CardContent from '@mui/material/CardContent'
import Typography from '@mui/material/Typography'
import Chip from '@mui/material/Chip'
import Box from '@mui/material/Box'
import Stack from '@mui/material/Stack'
import { formatDistanceToNow } from 'date-fns'

export default function InsightCard({ insight }) {
  return (
    <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <CardContent sx={{ flexGrow: 1 }}>
        <Typography variant="body1" component="div" gutterBottom>
          {insight.content}
        </Typography>
        
        <Box sx={{ mt: 2 }}>
          <Typography variant="caption" color="text.secondary">
            From: {insight.channel_name} • {insight.video_title}
          </Typography>
        </Box>
        
        <Box sx={{ mt: 1 }}>
          <Typography variant="caption" color="text.secondary">
            {formatDistanceToNow(new Date(insight.video_published_at), { addSuffix: true })}
          </Typography>
        </Box>
        
        {insight.topics && insight.topics.length > 0 && (
          <Stack direction="row" spacing={1} sx={{ mt: 2, flexWrap: 'wrap', gap: 1 }}>
            {insight.topics.map((topic, index) => (
              <Chip key={index} label={topic} size="small" />
            ))}
          </Stack>
        )}
      </CardContent>
    </Card>
  )
}

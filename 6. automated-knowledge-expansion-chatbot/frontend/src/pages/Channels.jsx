import { useState, useEffect } from 'react'
import Container from '@mui/material/Container'
import Grid from '@mui/material/Grid'
import Paper from '@mui/material/Paper'
import Typography from '@mui/material/Typography'
import Button from '@mui/material/Button'
import TextField from '@mui/material/TextField'
import Dialog from '@mui/material/Dialog'
import DialogActions from '@mui/material/DialogActions'
import DialogContent from '@mui/material/DialogContent'
import DialogContentText from '@mui/material/DialogContentText'
import DialogTitle from '@mui/material/DialogTitle'
import Table from '@mui/material/Table'
import TableBody from '@mui/material/TableBody'
import TableCell from '@mui/material/TableCell'
import TableContainer from '@mui/material/TableContainer'
import TableHead from '@mui/material/TableHead'
import TableRow from '@mui/material/TableRow'
import IconButton from '@mui/material/IconButton'
import DeleteIcon from '@mui/icons-material/Delete'
import AddIcon from '@mui/icons-material/Add'
import { formatDistanceToNow } from 'date-fns'

import { getChannels, addChannel, deleteChannel } from '../services/api'

export default function Channels() {
  const [channels, setChannels] = useState([])
  const [loading, setLoading] = useState(true)
  const [openDialog, setOpenDialog] = useState(false)
  const [newChannel, setNewChannel] = useState({
    youtube_id: '',
    name: '',
    description: '',
  })
  const [error, setError] = useState('')

  const fetchChannels = async () => {
    try {
      setLoading(true)
      const response = await getChannels()
      setChannels(response.data)
    } catch (error) {
      console.error('Error fetching channels:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchChannels()
  }, [])

  const handleOpenDialog = () => {
    setOpenDialog(true)
    setError('')
  }

  const handleCloseDialog = () => {
    setOpenDialog(false)
    setNewChannel({
      youtube_id: '',
      name: '',
      description: '',
    })
  }

  const handleInputChange = (e) => {
    const { name, value } = e.target
    setNewChannel({
      ...newChannel,
      [name]: value,
    })
  }

  const handleAddChannel = async () => {
    try {
      if (!newChannel.youtube_id) {
        setError('YouTube Channel ID is required')
        return
      }
      
      await addChannel(newChannel)
      handleCloseDialog()
      fetchChannels()
    } catch (error) {
      console.error('Error adding channel:', error)
      setError(error.response?.data?.detail || 'Error adding channel')
    }
  }

  const handleDeleteChannel = async (id) => {
    if (window.confirm('Are you sure you want to delete this channel?')) {
      try {
        await deleteChannel(id)
        fetchChannels()
      } catch (error) {
        console.error('Error deleting channel:', error)
      }
    }
  }

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Grid container spacing={3}>
        <Grid item xs={12} sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="h4">Monitored Channels</Typography>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={handleOpenDialog}
          >
            Add Channel
          </Button>
        </Grid>
        
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            {loading ? (
              <Typography>Loading channels...</Typography>
            ) : channels.length > 0 ? (
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Name</TableCell>
                      <TableCell>Description</TableCell>
                      <TableCell>YouTube ID</TableCell>
                      <TableCell>Added</TableCell>
                      <TableCell>Last Checked</TableCell>
                      <TableCell>Actions</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {channels.map((channel) => (
                      <TableRow key={channel.id}>
                        <TableCell>{channel.name}</TableCell>
                        <TableCell>{channel.description || 'N/A'}</TableCell>
                        <TableCell>{channel.youtube_id}</TableCell>
                        <TableCell>
                          {formatDistanceToNow(new Date(channel.created_at), { addSuffix: true })}
                        </TableCell>
                        <TableCell>
                          {channel.last_checked
                            ? formatDistanceToNow(new Date(channel.last_checked), { addSuffix: true })
                            : 'Never'}
                        </TableCell>
                        <TableCell>
                          <IconButton
                            color="error"
                            onClick={() => handleDeleteChannel(channel.id)}
                          >
                            <DeleteIcon />
                          </IconButton>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            ) : (
              <Typography>No channels added yet. Add a channel to start monitoring.</Typography>
            )}
          </Paper>
        </Grid>
      </Grid>
      
      {/* Add Channel Dialog */}
      <Dialog open={openDialog} onClose={handleCloseDialog}>
        <DialogTitle>Add YouTube Channel</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Enter the YouTube channel ID and details to start monitoring it for new videos.
          </DialogContentText>
          
          {error && (
            <Typography color="error" sx={{ mt: 2 }}>
              {error}
            </Typography>
          )}
          
          <TextField
            autoFocus
            margin="dense"
            name="youtube_id"
            label="YouTube Channel ID"
            type="text"
            fullWidth
            variant="outlined"
            value={newChannel.youtube_id}
            onChange={handleInputChange}
            required
            sx={{ mt: 2 }}
          />
          
          <TextField
            margin="dense"
            name="name"
            label="Channel Name"
            type="text"
            fullWidth
            variant="outlined"
            value={newChannel.name}
            onChange={handleInputChange}
            required
          />
          
          <TextField
            margin="dense"
            name="description"
            label="Description"
            type="text"
            fullWidth
            variant="outlined"
            value={newChannel.description}
            onChange={handleInputChange}
            multiline
            rows={3}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog}>Cancel</Button>
          <Button onClick={handleAddChannel} variant="contained">Add</Button>
        </DialogActions>
      </Dialog>
    </Container>
  )
}

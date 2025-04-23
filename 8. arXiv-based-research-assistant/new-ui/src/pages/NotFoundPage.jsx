import React from 'react';
import { Box, Typography, Button, Paper } from '@mui/material';
import { Link as RouterLink } from 'react-router-dom';

const NotFoundPage = () => {
  return (
    <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '70vh' }}>
      <Paper sx={{ p: 4, textAlign: 'center', maxWidth: 500 }}>
        <Typography variant="h1" color="primary" sx={{ fontWeight: 'bold' }}>
          404
        </Typography>
        <Typography variant="h5" gutterBottom>
          Page Not Found
        </Typography>
        <Typography variant="body1" paragraph>
          The page you are looking for doesn't exist or has been moved.
        </Typography>
        <Button variant="contained" component={RouterLink} to="/">
          Go to Home
        </Button>
      </Paper>
    </Box>
  );
};

export default NotFoundPage;


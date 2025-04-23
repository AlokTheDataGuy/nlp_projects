import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Card,
  CardContent
} from '@mui/material';

const VisualizationPage = () => {
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Knowledge Graph Visualization
      </Typography>
      
      <Paper sx={{ p: 3, mb: 4 }}>
        <Typography variant="body1" paragraph>
          This page will display a visualization of the knowledge graph connecting papers, authors, and concepts.
        </Typography>
        <Typography variant="body2" color="text.secondary">
          The visualization feature is currently under development.
        </Typography>
      </Paper>
      
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
        <Card sx={{ minWidth: 275, flex: 1 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Papers
            </Typography>
            <Typography variant="body2">
              The knowledge graph will include papers from arXiv, showing connections between related research.
            </Typography>
          </CardContent>
        </Card>
        
        <Card sx={{ minWidth: 275, flex: 1 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Authors
            </Typography>
            <Typography variant="body2">
              Authors will be connected to their papers, showing collaboration networks and research communities.
            </Typography>
          </CardContent>
        </Card>
        
        <Card sx={{ minWidth: 275, flex: 1 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Concepts
            </Typography>
            <Typography variant="body2">
              Key concepts and topics will be extracted from papers, showing the relationships between different research areas.
            </Typography>
          </CardContent>
        </Card>
      </Box>
    </Box>
  );
};

export default VisualizationPage;
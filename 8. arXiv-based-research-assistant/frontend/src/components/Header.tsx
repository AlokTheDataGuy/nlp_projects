import React from 'react';
import { AppBar, Toolbar, Typography, Button, Box } from '@mui/material';
import { Link as RouterLink } from 'react-router-dom';
import ChatIcon from '@mui/icons-material/Chat';
import ArticleIcon from '@mui/icons-material/Article';
import BubbleChartIcon from '@mui/icons-material/BubbleChart';

const Header: React.FC = () => {
  return (
    <AppBar position="static">
      <Toolbar>
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          ArXiv Expert Chatbot
        </Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button
            color="inherit"
            component={RouterLink}
            to="/"
            startIcon={<ChatIcon />}
          >
            Chat
          </Button>
          <Button
            color="inherit"
            component={RouterLink}
            to="/papers"
            startIcon={<ArticleIcon />}
          >
            Papers
          </Button>
          <Button
            color="inherit"
            component={RouterLink}
            to="/concepts"
            startIcon={<BubbleChartIcon />}
          >
            Concepts
          </Button>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Header;

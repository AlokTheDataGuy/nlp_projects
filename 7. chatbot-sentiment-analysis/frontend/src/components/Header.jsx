import React from 'react';
import { Link as RouterLink } from 'react-router-dom';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import Box from '@mui/material/Box';
import useMediaQuery from '@mui/material/useMediaQuery';
import { useTheme } from '@mui/material/styles';
import Container from '@mui/material/Container';

const Header = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  return (
    <Box sx={{ mt: 2, mx: 2 }}>
      <AppBar 
        position="static" 
        sx={{ 
          maxWidth: '65%', 
          mx: 'auto',
          borderRadius: 2,
          boxShadow: 3
        }}
      >
        <Container maxWidth="lg">
          <Toolbar>
            <Typography
              variant={isMobile ? "subtitle1" : "h6"}
              component="div"
              sx={{
                flexGrow: 1,
                textAlign: 'left'
              }}
            >
              Sentiment-Based Chatbot
            </Typography>
            <Box>
              <Button
                color="inherit"
                component={RouterLink}
                to="/"
                sx={{ mr: 1 }}
                size={isMobile ? "small" : "medium"}
              >
                Chat
              </Button>
              <Button
                color="inherit"
                component={RouterLink}
                to="/dashboard"
                size={isMobile ? "small" : "medium"}
              >
                Dashboard
              </Button>
            </Box>
          </Toolbar>
        </Container>
      </AppBar>
    </Box>
  );
};

export default Header;
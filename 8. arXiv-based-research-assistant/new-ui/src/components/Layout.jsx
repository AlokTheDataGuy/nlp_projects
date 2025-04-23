import React from 'react';
import {
  AppBar,
  Box,
  Toolbar,
  Typography,
  Button,
  Container
} from '@mui/material';
import { Link } from 'react-router-dom';

const Layout = ({ children }) => {
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            arXiv Research Assistant
          </Typography>
          <Button color="inherit" component={Link} to="/">Chat</Button>
          <Button color="inherit" component={Link} to="/search">Search</Button>
          <Button color="inherit" component={Link} to="/visualization">Visualization</Button>
        </Toolbar>
      </AppBar>
      <Container sx={{ mt: 4, mb: 4, flexGrow: 1 }}>
        {children}
      </Container>
    </Box>
  );
};

export default Layout;

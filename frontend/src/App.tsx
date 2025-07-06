import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { Box, AppBar, Toolbar, Typography, Container } from '@mui/material';
import PhotoLibraryIcon from '@mui/icons-material/PhotoLibrary';

import Sidebar from './components/Sidebar';
import PhotoGrid from './components/PhotoGrid';
import PhotoUpload from './components/PhotoUpload';
import PhotoDetail from './components/PhotoDetail';
import Search from './components/Search';
import Analytics from './components/Analytics';

function App() {
  return (
    <Box sx={{ display: 'flex' }}>
      <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
        <Toolbar>
          <PhotoLibraryIcon sx={{ mr: 2 }} />
          <Typography variant="h6" noWrap component="div">
            Photo Analyzer
          </Typography>
        </Toolbar>
      </AppBar>
      
      <Sidebar />
      
      <Box component="main" sx={{ flexGrow: 1, p: 3, mt: 8 }}>
        <Container maxWidth="xl">
          <Routes>
            <Route path="/" element={<PhotoGrid />} />
            <Route path="/upload" element={<PhotoUpload />} />
            <Route path="/photo/:id" element={<PhotoDetail />} />
            <Route path="/search" element={<Search />} />
            <Route path="/analytics" element={<Analytics />} />
          </Routes>
        </Container>
      </Box>
    </Box>
  );
}

export default App;
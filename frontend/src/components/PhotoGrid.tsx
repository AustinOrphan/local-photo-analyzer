import React, { useState } from 'react';
import { useQuery } from 'react-query';
import {
  Grid,
  Card,
  CardMedia,
  CardContent,
  Typography,
  Chip,
  Box,
  CircularProgress,
  Alert,
  Pagination,
  TextField,
  InputAdornment,
} from '@mui/material';
import { Search as SearchIcon } from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { format } from 'date-fns';

import { apiClient } from '../services/api';
import { Photo } from '../types';

function PhotoGrid() {
  const navigate = useNavigate();
  const [page, setPage] = useState(1);
  const [searchTerm, setSearchTerm] = useState('');
  const limit = 20;

  const { data, isLoading, error } = useQuery(
    ['photos', page, searchTerm],
    () => apiClient.getPhotos({ 
      limit, 
      offset: (page - 1) * limit,
      search: searchTerm || undefined 
    }),
    { keepPreviousData: true }
  );

  const photos = data?.photos || [];
  const totalPages = Math.ceil((data?.total || 0) / limit);

  const handlePhotoClick = (photo: Photo) => {
    navigate(`/photo/${photo.id}`);
  };

  const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(event.target.value);
    setPage(1); // Reset to first page when searching
  };

  if (isLoading && !data) {
    return (
      <Box display="flex" justifyContent="center" p={4}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mt: 2 }}>
        Failed to load photos. Please try again.
      </Alert>
    );
  }

  return (
    <Box>
      {/* Search Bar */}
      <TextField
        fullWidth
        variant="outlined"
        placeholder="Search photos by content, tags, or filename..."
        value={searchTerm}
        onChange={handleSearchChange}
        sx={{ mb: 3 }}
        InputProps={{
          startAdornment: (
            <InputAdornment position="start">
              <SearchIcon />
            </InputAdornment>
          ),
        }}
      />

      {/* Photo Grid */}
      {photos.length === 0 ? (
        <Box textAlign="center" py={8}>
          <Typography variant="h6" color="text.secondary">
            {searchTerm ? 'No photos found matching your search.' : 'No photos yet. Upload some to get started!'}
          </Typography>
        </Box>
      ) : (
        <>
          <Grid container spacing={2}>
            {photos.map((photo) => (
              <Grid item xs={12} sm={6} md={4} lg={3} key={photo.id}>
                <Card 
                  className="photo-card"
                  sx={{ cursor: 'pointer', height: '100%' }}
                  onClick={() => handlePhotoClick(photo)}
                >
                  <CardMedia
                    component="img"
                    height="200"
                    image={`/api/photos/${photo.id}/thumbnail`}
                    alt={photo.filename}
                    sx={{ objectFit: 'cover' }}
                    onError={(e) => {
                      // Fallback to a placeholder or the original image
                      (e.target as HTMLImageElement).src = '/placeholder-image.png';
                    }}
                  />
                  <CardContent>
                    <Typography variant="subtitle2" noWrap>
                      {photo.filename}
                    </Typography>
                    
                    {photo.date_taken && (
                      <Typography variant="caption" color="text.secondary" display="block">
                        {format(new Date(photo.date_taken), 'MMM dd, yyyy')}
                      </Typography>
                    )}
                    
                    {photo.description && (
                      <Typography 
                        variant="body2" 
                        color="text.secondary" 
                        sx={{ 
                          mt: 1, 
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          display: '-webkit-box',
                          WebkitLineClamp: 2,
                          WebkitBoxOrient: 'vertical',
                        }}
                      >
                        {photo.description}
                      </Typography>
                    )}
                    
                    {photo.tags.length > 0 && (
                      <Box className="tags-container" sx={{ mt: 1 }}>
                        {photo.tags.slice(0, 3).map((tag) => (
                          <Chip
                            key={tag}
                            label={tag}
                            size="small"
                            variant="outlined"
                            className="tag-chip"
                          />
                        ))}
                        {photo.tags.length > 3 && (
                          <Chip
                            label={`+${photo.tags.length - 3}`}
                            size="small"
                            variant="outlined"
                            className="tag-chip"
                          />
                        )}
                      </Box>
                    )}
                    
                    <Box display="flex" justifyContent="space-between" alignItems="center" mt={1}>
                      <Box display="flex" gap={1}>
                        {photo.analyzed && (
                          <Chip label="Analyzed" size="small" color="success" />
                        )}
                        {photo.organized && (
                          <Chip label="Organized" size="small" color="primary" />
                        )}
                      </Box>
                      {photo.confidence_score && (
                        <Typography variant="caption" color="text.secondary">
                          {Math.round(photo.confidence_score * 100)}%
                        </Typography>
                      )}
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>

          {/* Pagination */}
          {totalPages > 1 && (
            <Box display="flex" justifyContent="center" mt={4}>
              <Pagination
                count={totalPages}
                page={page}
                onChange={(_, newPage) => setPage(newPage)}
                color="primary"
              />
            </Box>
          )}
        </>
      )}
      
      {isLoading && (
        <Box display="flex" justifyContent="center" mt={2}>
          <CircularProgress size={24} />
        </Box>
      )}
    </Box>
  );
}

export default PhotoGrid;
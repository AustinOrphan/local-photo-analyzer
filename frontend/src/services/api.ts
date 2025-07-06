import axios from 'axios';
import { Photo, AnalysisResult, OrganizationResult } from '../types';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    console.log(`API Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('API Response Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

export interface GetPhotosParams {
  limit?: number;
  offset?: number;
  tag?: string;
  date_from?: string;
  date_to?: string;
  search?: string;
}

export interface GetPhotosResponse {
  photos: Photo[];
  total: number;
}

export interface UploadPhotoParams {
  file: File;
  auto_analyze?: boolean;
  auto_organize?: boolean;
}

export interface AnalyzePhotoParams {
  model?: string;
  include_tags?: boolean;
  include_description?: boolean;
  include_filename_suggestion?: boolean;
  custom_prompt?: string;
}

export interface OrganizePhotoParams {
  target_structure?: string;
  create_symlinks?: boolean;
  backup_original?: boolean;
  custom_path?: string;
}

export interface SearchParams {
  query: string;
  search_type?: string;
  limit?: number;
  date_from?: string;
  date_to?: string;
}

export const apiClient = {
  // Health checks
  async healthCheck(): Promise<{ status: string; version: string }> {
    const response = await api.get('/health');
    return response.data;
  },

  async checkOllamaHealth(): Promise<{ ollama_connected: boolean }> {
    const response = await api.get('/health/ollama');
    return response.data;
  },

  // Photo management
  async getPhotos(params: GetPhotosParams = {}): Promise<GetPhotosResponse> {
    const response = await api.get('/api/photos', { params });
    return {
      photos: response.data,
      total: response.headers['x-total-count'] ? parseInt(response.headers['x-total-count']) : response.data.length,
    };
  },

  async getPhoto(photoId: string): Promise<Photo> {
    const response = await api.get(`/api/photos/${photoId}`);
    return response.data;
  },

  async uploadPhoto(params: UploadPhotoParams): Promise<any> {
    const formData = new FormData();
    formData.append('file', params.file);
    
    if (params.auto_analyze !== undefined) {
      formData.append('auto_analyze', params.auto_analyze.toString());
    }
    if (params.auto_organize !== undefined) {
      formData.append('auto_organize', params.auto_organize.toString());
    }

    const response = await api.post('/api/photos/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  // Photo analysis
  async analyzePhoto(photoId: string, params: AnalyzePhotoParams = {}): Promise<AnalysisResult> {
    const response = await api.post(`/api/photos/${photoId}/analyze`, params);
    return response.data;
  },

  async analyzeBatch(photoIds: string[], model?: string): Promise<any> {
    const response = await api.post('/api/photos/analyze/batch', {
      photo_ids: photoIds,
      model,
    });
    return response.data;
  },

  // Photo organization
  async organizePhoto(photoId: string, params: OrganizePhotoParams = {}): Promise<OrganizationResult> {
    const response = await api.post(`/api/photos/${photoId}/organize`, params);
    return response.data;
  },

  // Search
  async searchPhotos(params: SearchParams): Promise<GetPhotosResponse> {
    const response = await api.get('/api/search', { params });
    return {
      photos: response.data.results || [],
      total: response.data.total || 0,
    };
  },

  // Statistics
  async getStats(): Promise<any> {
    const response = await api.get('/api/stats');
    return response.data;
  },

  // Tags
  async getTags(): Promise<string[]> {
    const response = await api.get('/api/tags');
    return response.data;
  },

  // Utility functions
  getPhotoThumbnailUrl(photoId: string): string {
    return `${API_BASE_URL}/api/photos/${photoId}/thumbnail`;
  },

  getPhotoUrl(photoId: string): string {
    return `${API_BASE_URL}/api/photos/${photoId}/file`;
  },
};

export default apiClient;
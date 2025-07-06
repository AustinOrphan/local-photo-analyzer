export interface Photo {
  id: string;
  filename: string;
  original_path: string;
  current_path: string;
  file_size: number;
  width?: number;
  height?: number;
  format?: string;
  date_taken?: string;
  date_modified: string;
  analyzed: boolean;
  organized: boolean;
  tags: string[];
  description?: string;
  suggested_filename?: string;
  confidence_score?: number;
  created_at: string;
  updated_at: string;
}

export interface AnalysisResult {
  photo_id: string;
  model_used: string;
  description?: string;
  tags: string[];
  suggested_filename?: string;
  confidence_score: number;
  analysis_time: number;
  timestamp: string;
  raw_response?: any;
}

export interface OrganizationResult {
  photo_id: string;
  old_path: string;
  new_path: string;
  symlinks_created: string[];
  backup_path?: string;
  organization_time: number;
  timestamp: string;
}

export interface SearchResult {
  query: string;
  search_type: string;
  total_results: number;
  photos: Photo[];
  search_time: number;
  timestamp: string;
}

export interface Stats {
  total_photos: number;
  analyzed_photos: number;
  organized_photos: number;
  total_tags: number;
  storage_used: number;
  last_analysis?: string;
  last_organization?: string;
}

export interface HealthStatus {
  status: string;
  version: string;
  ollama_connected: boolean;
  database_connected: boolean;
  disk_space_available?: number;
  timestamp: string;
}

export interface ApiError {
  error: string;
  detail?: string;
  error_code?: string;
  timestamp: string;
}
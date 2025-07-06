"""Pydantic schemas for the web API."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class PhotoBase(BaseModel):
    """Base photo schema."""
    filename: str
    file_size: int
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None


class PhotoResponse(PhotoBase):
    """Photo response schema."""
    id: str
    original_path: str
    current_path: str
    date_taken: Optional[datetime] = None
    date_modified: datetime
    analyzed: bool = False
    organized: bool = False
    tags: List[str] = []
    description: Optional[str] = None
    suggested_filename: Optional[str] = None
    confidence_score: Optional[float] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class AnalysisRequest(BaseModel):
    """Photo analysis request schema."""
    model: Optional[str] = None
    include_tags: bool = True
    include_description: bool = True
    include_filename_suggestion: bool = True
    custom_prompt: Optional[str] = None


class AnalysisResponse(BaseModel):
    """Photo analysis response schema."""
    photo_id: str
    model_used: str
    description: Optional[str] = None
    tags: List[str] = []
    suggested_filename: Optional[str] = None
    confidence_score: float
    analysis_time: float
    timestamp: datetime
    raw_response: Optional[Dict[str, Any]] = None


class OrganizationRequest(BaseModel):
    """Photo organization request schema."""
    target_structure: str = Field(default="date", description="Organization structure (date, tags, manual)")
    create_symlinks: bool = True
    backup_original: bool = True
    custom_path: Optional[str] = None


class OrganizationResponse(BaseModel):
    """Photo organization response schema."""
    photo_id: str
    old_path: str
    new_path: str
    symlinks_created: List[str] = []
    backup_path: Optional[str] = None
    organization_time: float
    timestamp: datetime


class BatchRequest(BaseModel):
    """Batch operation request schema."""
    photo_ids: List[str] = Field(..., max_items=100)
    model: Optional[str] = None
    operation: str = Field(default="analyze", description="Operation to perform (analyze, organize)")


class BatchResponse(BaseModel):
    """Batch operation response schema."""
    total_requested: int
    successful: int
    failed: int
    results: List[Dict[str, Any]]
    processing_time: float
    timestamp: datetime


class SearchRequest(BaseModel):
    """Search request schema."""
    query: str = Field(..., min_length=2)
    search_type: str = Field(default="all", description="Search type (all, tags, description, filename)")
    limit: int = Field(default=20, le=100)
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None


class SearchResponse(BaseModel):
    """Search response schema."""
    query: str
    search_type: str
    total_results: int
    photos: List[PhotoResponse]
    search_time: float
    timestamp: datetime


class TagResponse(BaseModel):
    """Tag response schema."""
    name: str
    count: int
    photos: Optional[List[str]] = None  # Photo IDs


class StatsResponse(BaseModel):
    """Statistics response schema."""
    total_photos: int
    analyzed_photos: int
    organized_photos: int
    total_tags: int
    storage_used: int  # bytes
    last_analysis: Optional[datetime] = None
    last_organization: Optional[datetime] = None


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str
    detail: Optional[str] = None
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    version: str
    ollama_connected: bool = False
    database_connected: bool = False
    disk_space_available: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class ConfigResponse(BaseModel):
    """Configuration response schema."""
    app_name: str
    version: str
    features: Dict[str, bool]
    models_available: List[str]
    max_upload_size: int
    supported_formats: List[str]


class UploadResponse(BaseModel):
    """File upload response schema."""
    message: str
    photo_id: str
    filename: str
    file_size: int
    auto_analyze: bool
    upload_time: float
    timestamp: datetime = Field(default_factory=datetime.now)
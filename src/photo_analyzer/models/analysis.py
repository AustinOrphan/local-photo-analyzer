"""Analysis result models."""

from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from photo_analyzer.models.base import Base


class AnalysisSession(Base):
    """Analysis session model for tracking batch analysis operations."""
    
    __tablename__ = 'analysis_sessions'
    
    name: Mapped[str] = mapped_column(
        String(200), 
        nullable=False, 
        comment="Session name or description"
    )
    model_name: Mapped[str] = mapped_column(
        String(100), 
        nullable=False, 
        comment="LLM model used for analysis"
    )
    model_version: Mapped[Optional[str]] = mapped_column(
        String(50), 
        nullable=True, 
        comment="Model version"
    )
    total_photos: Mapped[int] = mapped_column(
        Integer, 
        default=0, 
        nullable=False, 
        comment="Total number of photos in session"
    )
    processed_photos: Mapped[int] = mapped_column(
        Integer, 
        default=0, 
        nullable=False, 
        comment="Number of photos processed"
    )
    failed_photos: Mapped[int] = mapped_column(
        Integer, 
        default=0, 
        nullable=False, 
        comment="Number of photos that failed analysis"
    )
    
    # Timing
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), 
        nullable=True, 
        comment="When analysis started"
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), 
        nullable=True, 
        comment="When analysis completed"
    )
    
    # Configuration
    settings: Mapped[Optional[Dict]] = mapped_column(
        JSON, 
        nullable=True, 
        comment="Analysis settings used"
    )
    
    # Relationships
    analysis_results: Mapped[List["AnalysisResult"]] = relationship(
        "AnalysisResult",
        back_populates="session",
        cascade="all, delete-orphan"
    )
    
    # Indexes
    __table_args__ = (
        Index('idx_analysis_sessions_model', 'model_name'),
        Index('idx_analysis_sessions_dates', 'started_at', 'completed_at'),
    )
    
    @property
    def is_completed(self) -> bool:
        """Check if session is completed."""
        return self.completed_at is not None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.processed_photos == 0:
            return 0.0
        return ((self.processed_photos - self.failed_photos) / self.processed_photos) * 100
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Get duration in seconds if completed."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class AnalysisResult(Base):
    """Analysis result model for storing AI analysis of photos."""
    
    __tablename__ = 'analysis_results'
    
    # Foreign keys
    photo_id: Mapped[str] = mapped_column(
        String(36), 
        ForeignKey('photos.id'), 
        nullable=False, 
        index=True
    )
    session_id: Mapped[Optional[str]] = mapped_column(
        String(36), 
        ForeignKey('analysis_sessions.id'), 
        nullable=True, 
        index=True
    )
    
    # Model information
    model_name: Mapped[str] = mapped_column(
        String(100), 
        nullable=False, 
        comment="LLM model used"
    )
    model_version: Mapped[Optional[str]] = mapped_column(
        String(50), 
        nullable=True, 
        comment="Model version"
    )
    
    # Analysis results
    description: Mapped[Optional[str]] = mapped_column(
        Text, 
        nullable=True, 
        comment="Generated description of photo content"
    )
    tags_detected: Mapped[Optional[List[str]]] = mapped_column(
        JSON, 
        nullable=True, 
        comment="List of detected tags"
    )
    confidence_scores: Mapped[Optional[Dict[str, float]]] = mapped_column(
        JSON, 
        nullable=True, 
        comment="Confidence scores for each detected element"
    )
    
    # Content analysis
    objects_detected: Mapped[Optional[List[Dict]]] = mapped_column(
        JSON, 
        nullable=True, 
        comment="Detected objects with bounding boxes and confidence"
    )
    scene_classification: Mapped[Optional[Dict]] = mapped_column(
        JSON, 
        nullable=True, 
        comment="Scene classification results"
    )
    colors_detected: Mapped[Optional[List[str]]] = mapped_column(
        JSON, 
        nullable=True, 
        comment="Dominant colors in the image"
    )
    
    # Metrics
    analysis_duration: Mapped[Optional[float]] = mapped_column(
        Float, 
        nullable=True, 
        comment="Analysis duration in seconds"
    )
    overall_confidence: Mapped[Optional[float]] = mapped_column(
        Float, 
        nullable=True, 
        comment="Overall confidence score"
    )
    
    # Status
    status: Mapped[str] = mapped_column(
        String(20), 
        default='completed', 
        nullable=False, 
        comment="Analysis status (completed, failed, pending)"
    )
    error_message: Mapped[Optional[str]] = mapped_column(
        Text, 
        nullable=True, 
        comment="Error message if analysis failed"
    )
    
    # Suggested filename and organization
    suggested_filename: Mapped[Optional[str]] = mapped_column(
        String(255), 
        nullable=True, 
        comment="AI-suggested filename"
    )
    suggested_tags: Mapped[Optional[List[str]]] = mapped_column(
        JSON, 
        nullable=True, 
        comment="AI-suggested tags for organization"
    )
    
    # Raw response data
    raw_response: Mapped[Optional[Dict]] = mapped_column(
        JSON, 
        nullable=True, 
        comment="Complete raw response from LLM"
    )
    
    # Relationships
    photo: Mapped["Photo"] = relationship(
        "Photo", 
        back_populates="analysis_results"
    )
    session: Mapped[Optional["AnalysisSession"]] = relationship(
        "AnalysisSession", 
        back_populates="analysis_results"
    )
    
    # Indexes
    __table_args__ = (
        Index('idx_analysis_results_photo_id', 'photo_id'),
        Index('idx_analysis_results_session_id', 'session_id'),
        Index('idx_analysis_results_model', 'model_name', 'model_version'),
        Index('idx_analysis_results_status', 'status'),
        Index('idx_analysis_results_confidence', 'overall_confidence'),
    )
    
    @property
    def is_successful(self) -> bool:
        """Check if analysis was successful."""
        return self.status == 'completed' and self.error_message is None
    
    @property
    def tag_count(self) -> int:
        """Get number of detected tags."""
        return len(self.tags_detected) if self.tags_detected else 0
    
    @property
    def object_count(self) -> int:
        """Get number of detected objects."""
        return len(self.objects_detected) if self.objects_detected else 0
    
    def get_high_confidence_tags(self, threshold: float = 0.7) -> List[str]:
        """Get tags with confidence above threshold."""
        if not self.tags_detected or not self.confidence_scores:
            return []
        
        return [
            tag for tag in self.tags_detected
            if self.confidence_scores.get(tag, 0.0) >= threshold
        ]
    
    def get_primary_objects(self, min_confidence: float = 0.5) -> List[Dict]:
        """Get objects with confidence above minimum threshold."""
        if not self.objects_detected:
            return []
        
        return [
            obj for obj in self.objects_detected
            if obj.get('confidence', 0.0) >= min_confidence
        ]
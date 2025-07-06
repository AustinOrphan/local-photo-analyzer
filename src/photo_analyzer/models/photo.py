"""Photo and tag models."""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey, Index, Integer, JSON, String, Table, Text
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from photo_analyzer.models.base import Base


# Association table for many-to-many relationship between photos and tags
photo_tag_association = Table(
    'photo_tags',
    Base.metadata,
    mapped_column('photo_id', String(36), ForeignKey('photos.id'), primary_key=True),
    mapped_column('tag_id', String(36), ForeignKey('tags.id'), primary_key=True),
    mapped_column('confidence', Float, nullable=True, comment="Tag confidence score"),
    mapped_column('created_at', DateTime(timezone=True), server_default='now()'),
)


class Photo(Base):
    """Photo model representing an image file and its metadata."""
    
    __tablename__ = 'photos'
    
    # File information
    original_path: Mapped[str] = mapped_column(
        String(1000), 
        nullable=False, 
        comment="Original file path"
    )
    current_path: Mapped[str] = mapped_column(
        String(1000), 
        nullable=False, 
        comment="Current file path"
    )
    filename: Mapped[str] = mapped_column(
        String(255), 
        nullable=False, 
        comment="Current filename"
    )
    original_filename: Mapped[str] = mapped_column(
        String(255), 
        nullable=False, 
        comment="Original filename"
    )
    file_extension: Mapped[str] = mapped_column(
        String(10), 
        nullable=False, 
        comment="File extension"
    )
    file_size: Mapped[int] = mapped_column(
        Integer, 
        nullable=False, 
        comment="File size in bytes"
    )
    file_hash: Mapped[Optional[str]] = mapped_column(
        String(64), 
        nullable=True, 
        index=True, 
        comment="SHA-256 hash of file content"
    )
    
    # Image metadata
    width: Mapped[Optional[int]] = mapped_column(
        Integer, 
        nullable=True, 
        comment="Image width in pixels"
    )
    height: Mapped[Optional[int]] = mapped_column(
        Integer, 
        nullable=True, 
        comment="Image height in pixels"
    )
    color_mode: Mapped[Optional[str]] = mapped_column(
        String(20), 
        nullable=True, 
        comment="Color mode (RGB, RGBA, etc.)"
    )
    format: Mapped[Optional[str]] = mapped_column(
        String(20), 
        nullable=True, 
        comment="Image format (JPEG, PNG, etc.)"
    )
    
    # EXIF data
    date_taken: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), 
        nullable=True, 
        index=True, 
        comment="Date photo was taken from EXIF"
    )
    camera_make: Mapped[Optional[str]] = mapped_column(
        String(100), 
        nullable=True, 
        comment="Camera manufacturer"
    )
    camera_model: Mapped[Optional[str]] = mapped_column(
        String(100), 
        nullable=True, 
        comment="Camera model"
    )
    gps_latitude: Mapped[Optional[float]] = mapped_column(
        Float, 
        nullable=True, 
        comment="GPS latitude"
    )
    gps_longitude: Mapped[Optional[float]] = mapped_column(
        Float, 
        nullable=True, 
        comment="GPS longitude"
    )
    gps_altitude: Mapped[Optional[float]] = mapped_column(
        Float, 
        nullable=True, 
        comment="GPS altitude in meters"
    )
    exif_data: Mapped[Optional[Dict]] = mapped_column(
        JSON, 
        nullable=True, 
        comment="Complete EXIF data as JSON"
    )
    
    # Analysis status
    is_analyzed: Mapped[bool] = mapped_column(
        Boolean, 
        default=False, 
        nullable=False, 
        comment="Whether photo has been analyzed"
    )
    analysis_version: Mapped[Optional[str]] = mapped_column(
        String(20), 
        nullable=True, 
        comment="Version of analysis performed"
    )
    
    # Organization status
    is_organized: Mapped[bool] = mapped_column(
        Boolean, 
        default=False, 
        nullable=False, 
        comment="Whether photo has been organized"
    )
    date_organized: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), 
        nullable=True, 
        comment="When photo was organized"
    )
    
    # Content description
    ai_description: Mapped[Optional[str]] = mapped_column(
        Text, 
        nullable=True, 
        comment="AI-generated description of photo content"
    )
    user_description: Mapped[Optional[str]] = mapped_column(
        Text, 
        nullable=True, 
        comment="User-provided description"
    )
    
    # Relationships
    tags: Mapped[List["Tag"]] = relationship(
        "Tag",
        secondary=photo_tag_association,
        back_populates="photos",
        lazy="selectin"
    )
    
    analysis_results: Mapped[List["AnalysisResult"]] = relationship(
        "AnalysisResult",
        back_populates="photo",
        cascade="all, delete-orphan"
    )
    
    organizations: Mapped[List["Organization"]] = relationship(
        "Organization",
        back_populates="photo",
        cascade="all, delete-orphan"
    )
    
    # Indexes
    __table_args__ = (
        Index('idx_photos_original_path', 'original_path'),
        Index('idx_photos_current_path', 'current_path'),
        Index('idx_photos_date_taken', 'date_taken'),
        Index('idx_photos_file_hash', 'file_hash'),
        Index('idx_photos_analysis_status', 'is_analyzed', 'analysis_version'),
        Index('idx_photos_organization_status', 'is_organized'),
    )
    
    @property
    def current_file_path(self) -> Path:
        """Get current file path as Path object."""
        return Path(self.current_path)
    
    @property
    def original_file_path(self) -> Path:
        """Get original file path as Path object."""
        return Path(self.original_path)
    
    @property
    def has_gps_data(self) -> bool:
        """Check if photo has GPS coordinates."""
        return self.gps_latitude is not None and self.gps_longitude is not None
    
    @property
    def tag_names(self) -> List[str]:
        """Get list of tag names."""
        return [tag.name for tag in self.tags]
    
    def get_tag_confidence(self, tag_name: str) -> Optional[float]:
        """Get confidence score for a specific tag."""
        # Would need to query the association table for this
        # This is a simplified version
        for tag in self.tags:
            if tag.name == tag_name:
                return getattr(tag, '_confidence', None)
        return None


class Tag(Base):
    """Tag model for categorizing photos."""
    
    __tablename__ = 'tags'
    
    name: Mapped[str] = mapped_column(
        String(100), 
        nullable=False, 
        unique=True, 
        index=True, 
        comment="Tag name"
    )
    category: Mapped[Optional[str]] = mapped_column(
        String(50), 
        nullable=True, 
        index=True, 
        comment="Tag category (object, scene, color, etc.)"
    )
    description: Mapped[Optional[str]] = mapped_column(
        Text, 
        nullable=True, 
        comment="Tag description"
    )
    color: Mapped[Optional[str]] = mapped_column(
        String(7), 
        nullable=True, 
        comment="Display color (hex code)"
    )
    
    # Usage statistics
    usage_count: Mapped[int] = mapped_column(
        Integer, 
        default=0, 
        nullable=False, 
        comment="Number of photos with this tag"
    )
    
    # Auto-generated vs manual
    is_auto_generated: Mapped[bool] = mapped_column(
        Boolean, 
        default=True, 
        nullable=False, 
        comment="Whether tag was auto-generated by AI"
    )
    
    # Relationships
    photos: Mapped[List["Photo"]] = relationship(
        "Photo",
        secondary=photo_tag_association,
        back_populates="tags",
        lazy="selectin"
    )
    
    # Indexes
    __table_args__ = (
        Index('idx_tags_name', 'name'),
        Index('idx_tags_category', 'category'),
        Index('idx_tags_usage_count', 'usage_count'),
    )
    
    def __str__(self) -> str:
        """String representation of tag."""
        return self.name


class PhotoTag(Base):
    """Association model for photo-tag relationships with additional metadata."""
    
    __tablename__ = 'photo_tag_details'
    
    photo_id: Mapped[str] = mapped_column(
        String(36), 
        ForeignKey('photos.id'), 
        nullable=False
    )
    tag_id: Mapped[str] = mapped_column(
        String(36), 
        ForeignKey('tags.id'), 
        nullable=False
    )
    confidence: Mapped[Optional[float]] = mapped_column(
        Float, 
        nullable=True, 
        comment="Confidence score for this tag on this photo"
    )
    source: Mapped[str] = mapped_column(
        String(20), 
        default='ai', 
        nullable=False, 
        comment="Source of tag (ai, user, import)"
    )
    is_verified: Mapped[bool] = mapped_column(
        Boolean, 
        default=False, 
        nullable=False, 
        comment="Whether tag has been verified by user"
    )
    
    # Relationships
    photo: Mapped["Photo"] = relationship("Photo")
    tag: Mapped["Tag"] = relationship("Tag")
    
    # Indexes
    __table_args__ = (
        Index('idx_photo_tag_photo_id', 'photo_id'),
        Index('idx_photo_tag_tag_id', 'tag_id'),
        Index('idx_photo_tag_confidence', 'confidence'),
    )
"""Organization and file management models."""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from photo_analyzer.models.base import Base


class Organization(Base):
    """Organization model for tracking photo file organization operations."""
    
    __tablename__ = 'organizations'
    
    # Foreign key
    photo_id: Mapped[str] = mapped_column(
        String(36), 
        ForeignKey('photos.id'), 
        nullable=False, 
        index=True
    )
    
    # Operation details
    operation_type: Mapped[str] = mapped_column(
        String(50), 
        nullable=False, 
        comment="Type of organization operation (move, copy, rename, etc.)"
    )
    source_path: Mapped[str] = mapped_column(
        String(1000), 
        nullable=False, 
        comment="Original file path before organization"
    )
    destination_path: Mapped[str] = mapped_column(
        String(1000), 
        nullable=False, 
        comment="New file path after organization"
    )
    
    # Backup information
    backup_path: Mapped[Optional[str]] = mapped_column(
        String(1000), 
        nullable=True, 
        comment="Path to backup file if created"
    )
    backup_created: Mapped[bool] = mapped_column(
        Boolean, 
        default=False, 
        nullable=False, 
        comment="Whether backup was created"
    )
    
    # Organization strategy
    organization_strategy: Mapped[str] = mapped_column(
        String(50), 
        nullable=False, 
        comment="Strategy used (date_based, tag_based, manual, etc.)"
    )
    date_folder_structure: Mapped[Optional[str]] = mapped_column(
        String(100), 
        nullable=True, 
        comment="Date folder structure used (YYYY/MM/DD, etc.)"
    )
    
    # Status
    status: Mapped[str] = mapped_column(
        String(20), 
        default='pending', 
        nullable=False, 
        comment="Organization status (pending, completed, failed, reverted)"
    )
    error_message: Mapped[Optional[str]] = mapped_column(
        Text, 
        nullable=True, 
        comment="Error message if organization failed"
    )
    
    # Timing
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), 
        nullable=True, 
        comment="When organization started"
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), 
        nullable=True, 
        comment="When organization completed"
    )
    
    # Reversal information
    is_reversible: Mapped[bool] = mapped_column(
        Boolean, 
        default=True, 
        nullable=False, 
        comment="Whether this operation can be reversed"
    )
    reverted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), 
        nullable=True, 
        comment="When operation was reverted"
    )
    reverted_by: Mapped[Optional[str]] = mapped_column(
        String(36), 
        nullable=True, 
        comment="ID of reversal operation"
    )
    
    # Additional metadata
    metadata: Mapped[Optional[Dict]] = mapped_column(
        JSON, 
        nullable=True, 
        comment="Additional operation metadata"
    )
    
    # Relationships
    photo: Mapped["Photo"] = relationship(
        "Photo", 
        back_populates="organizations"
    )
    
    symbolic_links: Mapped[list["SymbolicLink"]] = relationship(
        "SymbolicLink",
        back_populates="organization",
        cascade="all, delete-orphan"
    )
    
    # Indexes
    __table_args__ = (
        Index('idx_organizations_photo_id', 'photo_id'),
        Index('idx_organizations_operation_type', 'operation_type'),
        Index('idx_organizations_status', 'status'),
        Index('idx_organizations_strategy', 'organization_strategy'),
        Index('idx_organizations_dates', 'started_at', 'completed_at'),
    )
    
    @property
    def is_completed(self) -> bool:
        """Check if organization is completed."""
        return self.status == 'completed'
    
    @property
    def is_failed(self) -> bool:
        """Check if organization failed."""
        return self.status == 'failed'
    
    @property
    def is_reverted(self) -> bool:
        """Check if organization was reverted."""
        return self.reverted_at is not None
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Get operation duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def source_file_path(self) -> Path:
        """Get source path as Path object."""
        return Path(self.source_path)
    
    @property
    def destination_file_path(self) -> Path:
        """Get destination path as Path object."""
        return Path(self.destination_path)
    
    @property
    def backup_file_path(self) -> Optional[Path]:
        """Get backup path as Path object."""
        return Path(self.backup_path) if self.backup_path else None


class SymbolicLink(Base):
    """Symbolic link model for tracking tag-based categorical organization."""
    
    __tablename__ = 'symbolic_links'
    
    # Foreign key to organization
    organization_id: Mapped[Optional[str]] = mapped_column(
        String(36), 
        ForeignKey('organizations.id'), 
        nullable=True, 
        index=True
    )
    
    # Photo reference (for direct links not tied to specific organization)
    photo_id: Mapped[str] = mapped_column(
        String(36), 
        ForeignKey('photos.id'), 
        nullable=False, 
        index=True
    )
    
    # Link details
    link_path: Mapped[str] = mapped_column(
        String(1000), 
        nullable=False, 
        comment="Path to the symbolic link"
    )
    target_path: Mapped[str] = mapped_column(
        String(1000), 
        nullable=False, 
        comment="Path that the symbolic link points to"
    )
    
    # Categorization
    link_type: Mapped[str] = mapped_column(
        String(50), 
        nullable=False, 
        comment="Type of link (tag_based, date_based, collection, etc.)"
    )
    category: Mapped[Optional[str]] = mapped_column(
        String(100), 
        nullable=True, 
        comment="Category or tag name for the link"
    )
    subcategory: Mapped[Optional[str]] = mapped_column(
        String(100), 
        nullable=True, 
        comment="Subcategory for hierarchical organization"
    )
    
    # Status
    is_valid: Mapped[bool] = mapped_column(
        Boolean, 
        default=True, 
        nullable=False, 
        comment="Whether the symbolic link is valid"
    )
    last_verified: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), 
        nullable=True, 
        comment="When link was last verified"
    )
    
    # Creation details
    created_by: Mapped[str] = mapped_column(
        String(50), 
        default='system', 
        nullable=False, 
        comment="Who/what created the link (system, user, import)"
    )
    auto_created: Mapped[bool] = mapped_column(
        Boolean, 
        default=True, 
        nullable=False, 
        comment="Whether link was automatically created"
    )
    
    # Relationships
    organization: Mapped[Optional["Organization"]] = relationship(
        "Organization", 
        back_populates="symbolic_links"
    )
    photo: Mapped["Photo"] = relationship("Photo")
    
    # Indexes
    __table_args__ = (
        Index('idx_symbolic_links_organization_id', 'organization_id'),
        Index('idx_symbolic_links_photo_id', 'photo_id'),
        Index('idx_symbolic_links_link_path', 'link_path'),
        Index('idx_symbolic_links_type_category', 'link_type', 'category'),
        Index('idx_symbolic_links_validity', 'is_valid', 'last_verified'),
    )
    
    @property
    def link_file_path(self) -> Path:
        """Get link path as Path object."""
        return Path(self.link_path)
    
    @property
    def target_file_path(self) -> Path:
        """Get target path as Path object."""
        return Path(self.target_path)
    
    @property
    def exists(self) -> bool:
        """Check if the symbolic link file exists."""
        return self.link_file_path.exists()
    
    @property
    def target_exists(self) -> bool:
        """Check if the target file exists."""
        return self.target_file_path.exists()
    
    @property
    def is_broken(self) -> bool:
        """Check if the symbolic link is broken."""
        return self.exists and not self.target_exists
    
    def verify_link(self) -> bool:
        """Verify that the symbolic link is valid and update status."""
        is_valid = self.exists and self.target_exists and self.link_file_path.is_symlink()
        
        if is_valid != self.is_valid:
            self.is_valid = is_valid
            self.last_verified = datetime.utcnow()
        
        return is_valid
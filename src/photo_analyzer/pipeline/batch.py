"""Batch processing with progress tracking and error handling."""

import asyncio
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from photo_analyzer.core.config import get_config
from photo_analyzer.core.logger import get_logger, audit_log
from photo_analyzer.models.photo import Photo
from photo_analyzer.pipeline.analyzer import PhotoAnalyzer
from photo_analyzer.pipeline.processor import PhotoProcessor
from photo_analyzer.pipeline.organizer import PhotoOrganizer
from photo_analyzer.analyzer.duplicates import DuplicateDetector

logger = get_logger(__name__)


class BatchStatus(Enum):
    """Batch processing status."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchItem:
    """Individual item in a batch operation."""
    id: str
    photo_id: str
    status: str = "pending"
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class BatchOperation:
    """Batch operation tracking."""
    id: str
    operation_type: str
    total_items: int
    completed_items: int = 0
    failed_items: int = 0
    status: BatchStatus = BatchStatus.PENDING
    progress: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    items: List[BatchItem] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_summary: Optional[str] = None


@dataclass
class BatchConfig:
    """Configuration for batch operations."""
    max_concurrent: int = 3
    retry_attempts: int = 2
    retry_delay: float = 5.0
    progress_callback: Optional[Callable] = None
    error_callback: Optional[Callable] = None
    checkpoint_interval: int = 10  # Save progress every N items


class BatchProcessor:
    """Advanced batch processor with progress tracking and resilience."""
    
    def __init__(self, config=None):
        """Initialize batch processor."""
        self.config = config or get_config()
        self.active_batches: Dict[str, BatchOperation] = {}
        self.analyzer = PhotoAnalyzer()
        self.processor = PhotoProcessor()
        self.organizer = PhotoOrganizer()
        self.duplicate_detector = DuplicateDetector()
        
        logger.info("Initialized batch processor")
    
    async def analyze_photos_batch(
        self,
        photo_ids: List[str],
        model: Optional[str] = None,
        batch_config: Optional[BatchConfig] = None
    ) -> str:
        """Analyze multiple photos in batch with progress tracking."""
        batch_id = str(uuid.uuid4())
        batch_config = batch_config or BatchConfig()
        
        # Create batch operation
        batch_op = BatchOperation(
            id=batch_id,
            operation_type="analyze",
            total_items=len(photo_ids),
            metadata={"model": model}
        )
        
        # Create batch items
        for photo_id in photo_ids:
            item = BatchItem(
                id=str(uuid.uuid4()),
                photo_id=photo_id
            )
            batch_op.items.append(item)
        
        self.active_batches[batch_id] = batch_op
        
        # Start batch processing
        asyncio.create_task(
            self._process_analysis_batch(batch_op, batch_config, model)
        )
        
        logger.info(f"Started batch analysis {batch_id} for {len(photo_ids)} photos")
        audit_log("BATCH_ANALYSIS_STARTED", batch_id=batch_id, count=len(photo_ids))
        
        return batch_id
    
    async def organize_photos_batch(
        self,
        photo_ids: List[str],
        organization_config: Dict[str, Any],
        batch_config: Optional[BatchConfig] = None
    ) -> str:
        """Organize multiple photos in batch."""
        batch_id = str(uuid.uuid4())
        batch_config = batch_config or BatchConfig()
        
        # Create batch operation
        batch_op = BatchOperation(
            id=batch_id,
            operation_type="organize",
            total_items=len(photo_ids),
            metadata=organization_config
        )
        
        # Create batch items
        for photo_id in photo_ids:
            item = BatchItem(
                id=str(uuid.uuid4()),
                photo_id=photo_id
            )
            batch_op.items.append(item)
        
        self.active_batches[batch_id] = batch_op
        
        # Start batch processing
        asyncio.create_task(
            self._process_organization_batch(batch_op, batch_config, organization_config)
        )
        
        logger.info(f"Started batch organization {batch_id} for {len(photo_ids)} photos")
        audit_log("BATCH_ORGANIZATION_STARTED", batch_id=batch_id, count=len(photo_ids))
        
        return batch_id
    
    async def detect_duplicates_batch(
        self,
        photo_ids: List[str],
        detection_types: List[str] = None,
        batch_config: Optional[BatchConfig] = None
    ) -> str:
        """Detect duplicates in batch."""
        batch_id = str(uuid.uuid4())
        batch_config = batch_config or BatchConfig()
        detection_types = detection_types or ['exact', 'near', 'similar']
        
        # Create batch operation
        batch_op = BatchOperation(
            id=batch_id,
            operation_type="duplicate_detection",
            total_items=1,  # Single operation for all photos
            metadata={"detection_types": detection_types, "photo_count": len(photo_ids)}
        )
        
        # Create single batch item for the entire operation
        item = BatchItem(
            id=str(uuid.uuid4()),
            photo_id="all",  # Special ID for batch operations
        )
        batch_op.items.append(item)
        
        self.active_batches[batch_id] = batch_op
        
        # Start duplicate detection
        asyncio.create_task(
            self._process_duplicate_detection_batch(
                batch_op, batch_config, photo_ids, detection_types
            )
        )
        
        logger.info(f"Started duplicate detection {batch_id} for {len(photo_ids)} photos")
        audit_log("BATCH_DUPLICATE_DETECTION_STARTED", batch_id=batch_id, count=len(photo_ids))
        
        return batch_id
    
    async def _process_analysis_batch(
        self,
        batch_op: BatchOperation,
        batch_config: BatchConfig,
        model: Optional[str]
    ):
        """Process analysis batch operation."""
        batch_op.status = BatchStatus.RUNNING
        batch_op.started_at = datetime.now()
        
        try:
            # Create semaphore for concurrency control
            semaphore = asyncio.Semaphore(batch_config.max_concurrent)
            
            # Process items with controlled concurrency
            tasks = []
            for item in batch_op.items:
                task = self._analyze_single_item(
                    item, batch_op, batch_config, semaphore, model
                )
                tasks.append(task)
            
            # Wait for all tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Update final status
            if batch_op.failed_items == 0:
                batch_op.status = BatchStatus.COMPLETED
            else:
                batch_op.status = BatchStatus.FAILED
                batch_op.error_summary = f"{batch_op.failed_items}/{batch_op.total_items} items failed"
            
            batch_op.completed_at = datetime.now()
            batch_op.progress = 100.0
            
            # Call completion callback
            if batch_config.progress_callback:
                await batch_config.progress_callback(batch_op)
            
            logger.info(
                f"Batch analysis {batch_op.id} completed: "
                f"{batch_op.completed_items} successful, {batch_op.failed_items} failed"
            )
            
            audit_log(
                "BATCH_ANALYSIS_COMPLETED",
                batch_id=batch_op.id,
                successful=batch_op.completed_items,
                failed=batch_op.failed_items
            )
            
        except Exception as e:
            batch_op.status = BatchStatus.FAILED
            batch_op.error_summary = str(e)
            batch_op.completed_at = datetime.now()
            
            logger.error(f"Batch analysis {batch_op.id} failed: {e}")
            audit_log("BATCH_ANALYSIS_FAILED", batch_id=batch_op.id, error=str(e))
            
            if batch_config.error_callback:
                await batch_config.error_callback(batch_op, e)
    
    async def _analyze_single_item(
        self,
        item: BatchItem,
        batch_op: BatchOperation,
        batch_config: BatchConfig,
        semaphore: asyncio.Semaphore,
        model: Optional[str]
    ):
        """Analyze a single photo item with retry logic."""
        async with semaphore:
            item.started_at = datetime.now()
            item.status = "running"
            
            for attempt in range(batch_config.retry_attempts + 1):
                try:
                    # TODO: Get photo from database
                    photo_path = f"/path/to/photo/{item.photo_id}"  # Placeholder
                    
                    # Perform analysis
                    result = await self.analyzer.analyze_photo(
                        photo_path,
                        model=model
                    )
                    
                    # Update item
                    item.status = "completed"
                    item.progress = 100.0
                    item.result = result
                    item.completed_at = datetime.now()
                    
                    # Update batch
                    batch_op.completed_items += 1
                    batch_op.progress = (batch_op.completed_items / batch_op.total_items) * 100
                    
                    # Call progress callback
                    if batch_config.progress_callback:
                        await batch_config.progress_callback(batch_op)
                    
                    break
                    
                except Exception as e:
                    if attempt == batch_config.retry_attempts:
                        # Final attempt failed
                        item.status = "failed"
                        item.error = str(e)
                        item.completed_at = datetime.now()
                        
                        batch_op.failed_items += 1
                        batch_op.progress = ((batch_op.completed_items + batch_op.failed_items) / batch_op.total_items) * 100
                        
                        logger.error(f"Item {item.id} failed after {attempt + 1} attempts: {e}")
                    else:
                        # Retry after delay
                        logger.warning(f"Item {item.id} attempt {attempt + 1} failed, retrying: {e}")
                        await asyncio.sleep(batch_config.retry_delay)
    
    async def _process_organization_batch(
        self,
        batch_op: BatchOperation,
        batch_config: BatchConfig,
        organization_config: Dict[str, Any]
    ):
        """Process organization batch operation."""
        batch_op.status = BatchStatus.RUNNING
        batch_op.started_at = datetime.now()
        
        try:
            semaphore = asyncio.Semaphore(batch_config.max_concurrent)
            
            tasks = []
            for item in batch_op.items:
                task = self._organize_single_item(
                    item, batch_op, batch_config, semaphore, organization_config
                )
                tasks.append(task)
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Update final status
            if batch_op.failed_items == 0:
                batch_op.status = BatchStatus.COMPLETED
            else:
                batch_op.status = BatchStatus.FAILED
                batch_op.error_summary = f"{batch_op.failed_items}/{batch_op.total_items} items failed"
            
            batch_op.completed_at = datetime.now()
            batch_op.progress = 100.0
            
            logger.info(
                f"Batch organization {batch_op.id} completed: "
                f"{batch_op.completed_items} successful, {batch_op.failed_items} failed"
            )
            
        except Exception as e:
            batch_op.status = BatchStatus.FAILED
            batch_op.error_summary = str(e)
            batch_op.completed_at = datetime.now()
            logger.error(f"Batch organization {batch_op.id} failed: {e}")
    
    async def _organize_single_item(
        self,
        item: BatchItem,
        batch_op: BatchOperation,
        batch_config: BatchConfig,
        semaphore: asyncio.Semaphore,
        organization_config: Dict[str, Any]
    ):
        """Organize a single photo item."""
        async with semaphore:
            item.started_at = datetime.now()
            item.status = "running"
            
            try:
                # TODO: Get photo from database and organize
                # photo = get_photo_by_id(item.photo_id)
                # result = await self.organizer.organize_photo(photo, **organization_config)
                
                # Placeholder result
                result = {"organized": True, "new_path": f"/organized/{item.photo_id}"}
                
                item.status = "completed"
                item.progress = 100.0
                item.result = result
                item.completed_at = datetime.now()
                
                batch_op.completed_items += 1
                batch_op.progress = (batch_op.completed_items / batch_op.total_items) * 100
                
            except Exception as e:
                item.status = "failed"
                item.error = str(e)
                item.completed_at = datetime.now()
                
                batch_op.failed_items += 1
                batch_op.progress = ((batch_op.completed_items + batch_op.failed_items) / batch_op.total_items) * 100
                
                logger.error(f"Organization item {item.id} failed: {e}")
    
    async def _process_duplicate_detection_batch(
        self,
        batch_op: BatchOperation,
        batch_config: BatchConfig,
        photo_ids: List[str],
        detection_types: List[str]
    ):
        """Process duplicate detection batch operation."""
        batch_op.status = BatchStatus.RUNNING
        batch_op.started_at = datetime.now()
        
        item = batch_op.items[0]
        item.started_at = datetime.now()
        item.status = "running"
        
        try:
            # TODO: Get photos from database
            photos = []  # Placeholder: get_photos_by_ids(photo_ids)
            
            # Detect duplicates
            duplicate_groups = await self.duplicate_detector.detect_duplicates(
                photos, detection_types
            )
            
            # Process results
            result = {
                "duplicate_groups": len(duplicate_groups),
                "total_duplicates": sum(len(group.photo_ids) for group in duplicate_groups),
                "groups": [
                    {
                        "representative_id": group.representative_id,
                        "photo_ids": group.photo_ids,
                        "similarity_score": group.similarity_score,
                        "duplicate_type": group.duplicate_type
                    }
                    for group in duplicate_groups
                ]
            }
            
            item.status = "completed"
            item.progress = 100.0
            item.result = result
            item.completed_at = datetime.now()
            
            batch_op.completed_items = 1
            batch_op.progress = 100.0
            batch_op.status = BatchStatus.COMPLETED
            batch_op.completed_at = datetime.now()
            
            logger.info(
                f"Duplicate detection {batch_op.id} completed: "
                f"{len(duplicate_groups)} groups found"
            )
            
            audit_log(
                "BATCH_DUPLICATE_DETECTION_COMPLETED",
                batch_id=batch_op.id,
                groups_found=len(duplicate_groups)
            )
            
        except Exception as e:
            item.status = "failed"
            item.error = str(e)
            item.completed_at = datetime.now()
            
            batch_op.failed_items = 1
            batch_op.status = BatchStatus.FAILED
            batch_op.error_summary = str(e)
            batch_op.completed_at = datetime.now()
            
            logger.error(f"Duplicate detection {batch_op.id} failed: {e}")
            audit_log("BATCH_DUPLICATE_DETECTION_FAILED", batch_id=batch_op.id, error=str(e))
    
    def get_batch_status(self, batch_id: str) -> Optional[BatchOperation]:
        """Get status of a batch operation."""
        return self.active_batches.get(batch_id)
    
    def list_active_batches(self) -> List[BatchOperation]:
        """List all active batch operations."""
        return list(self.active_batches.values())
    
    async def cancel_batch(self, batch_id: str) -> bool:
        """Cancel a batch operation."""
        batch_op = self.active_batches.get(batch_id)
        if not batch_op:
            return False
        
        if batch_op.status in [BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED]:
            return False
        
        batch_op.status = BatchStatus.CANCELLED
        batch_op.completed_at = datetime.now()
        
        logger.info(f"Cancelled batch operation {batch_id}")
        audit_log("BATCH_CANCELLED", batch_id=batch_id)
        
        return True
    
    async def pause_batch(self, batch_id: str) -> bool:
        """Pause a batch operation."""
        batch_op = self.active_batches.get(batch_id)
        if not batch_op or batch_op.status != BatchStatus.RUNNING:
            return False
        
        batch_op.status = BatchStatus.PAUSED
        logger.info(f"Paused batch operation {batch_id}")
        return True
    
    async def resume_batch(self, batch_id: str) -> bool:
        """Resume a paused batch operation."""
        batch_op = self.active_batches.get(batch_id)
        if not batch_op or batch_op.status != BatchStatus.PAUSED:
            return False
        
        batch_op.status = BatchStatus.RUNNING
        logger.info(f"Resumed batch operation {batch_id}")
        return True
    
    def cleanup_completed_batches(self, max_age_hours: int = 24):
        """Clean up old completed batch operations."""
        current_time = datetime.now()
        to_remove = []
        
        for batch_id, batch_op in self.active_batches.items():
            if batch_op.status in [BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED]:
                if batch_op.completed_at:
                    age_hours = (current_time - batch_op.completed_at).total_seconds() / 3600
                    if age_hours > max_age_hours:
                        to_remove.append(batch_id)
        
        for batch_id in to_remove:
            del self.active_batches[batch_id]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old batch operations")
    
    async def get_batch_progress_stream(self, batch_id: str) -> AsyncIterator[BatchOperation]:
        """Get real-time progress updates for a batch operation."""
        batch_op = self.active_batches.get(batch_id)
        if not batch_op:
            return
        
        last_progress = -1
        
        while batch_op.status in [BatchStatus.PENDING, BatchStatus.RUNNING, BatchStatus.PAUSED]:
            if batch_op.progress != last_progress:
                yield batch_op
                last_progress = batch_op.progress
            
            await asyncio.sleep(1)  # Update every second
        
        # Send final update
        yield batch_op
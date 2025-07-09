"""Photo processing pipeline for file operations and transformations."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
import uuid
import shutil

from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import get_config
from ..core.logger import get_logger
from ..database.session import get_async_db_session
from ..models.photo import Photo
from ..models.organization import Organization, SymbolicLink
from ..utils.image import ImageProcessor
from ..utils.exif import ExifExtractor
from ..utils.date_utils import DateUtils
from ..utils.file_utils import FileUtils

logger = get_logger(__name__)


class PhotoProcessor:
    """Handles photo file operations and transformations."""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.image_processor = ImageProcessor()
        self.exif_extractor = ExifExtractor()
        self.logger = logger
    
    async def rename_photo(
        self,
        photo_id: str,
        new_filename: str,
        session: Optional[AsyncSession] = None,
        preserve_extension: bool = True,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Rename a photo file and update database."""
        if session is None:
            async with get_async_db_session() as session:
                return await self._rename_photo_impl(
                    photo_id, new_filename, session, preserve_extension, dry_run
                )
        else:
            return await self._rename_photo_impl(
                photo_id, new_filename, session, preserve_extension, dry_run
            )
    
    async def _rename_photo_impl(
        self,
        photo_id: str,
        new_filename: str,
        session: AsyncSession,
        preserve_extension: bool,
        dry_run: bool
    ) -> Dict[str, Any]:
        """Internal rename implementation."""
        from sqlalchemy import select
        
        # Get photo record
        stmt = select(Photo).where(Photo.id == photo_id)
        result = await session.execute(stmt)
        photo = result.scalar_one_or_none()
        
        if not photo:
            raise ValueError(f"Photo not found: {photo_id}")
        
        current_path = Path(photo.current_path)
        if not current_path.exists():
            raise FileNotFoundError(f"Photo file not found: {current_path}")
        
        # Process new filename
        if preserve_extension and not Path(new_filename).suffix:
            new_filename += current_path.suffix
        
        # Ensure filename is valid
        new_filename = self._sanitize_filename(new_filename)
        new_path = current_path.parent / new_filename
        
        # Check if target exists
        if new_path.exists() and new_path != current_path:
            raise FileExistsError(f"Target file already exists: {new_path}")
        
        if dry_run:
            return {
                'photo_id': photo_id,
                'current_path': str(current_path),
                'new_path': str(new_path),
                'operation': 'rename',
                'dry_run': True,
                'success': True
            }
        
        try:
            # Record operation
            operation = await self._create_operation_record(
                session, photo_id, 'rename',
                {'old_path': str(current_path), 'new_path': str(new_path)}
            )
            
            # Perform rename
            current_path.rename(new_path)
            
            # Update photo record
            photo.current_path = str(new_path)
            photo.filename = new_filename
            photo.updated_at = datetime.utcnow()
            
            # Update operation status
            operation.status = 'completed'
            operation.completed_at = datetime.utcnow()
            
            await session.commit()
            
            logger.info(f"Renamed photo {photo_id}: {current_path.name} -> {new_filename}")
            
            return {
                'photo_id': photo_id,
                'current_path': str(current_path),
                'new_path': str(new_path),
                'operation': 'rename',
                'operation_id': operation.id,
                'dry_run': False,
                'success': True
            }
            
        except Exception as e:
            if 'operation' in locals():
                operation.status = 'failed'
                operation.error_message = str(e)
                operation.completed_at = datetime.utcnow()
                await session.commit()
            
            logger.error(f"Failed to rename photo {photo_id}: {e}")
            raise
    
    async def move_photo(
        self,
        photo_id: str,
        destination_dir: Union[str, Path],
        session: Optional[AsyncSession] = None,
        create_dirs: bool = True,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Move photo to new directory."""
        if session is None:
            async with get_async_db_session() as session:
                return await self._move_photo_impl(
                    photo_id, destination_dir, session, create_dirs, dry_run
                )
        else:
            return await self._move_photo_impl(
                photo_id, destination_dir, session, create_dirs, dry_run
            )
    
    async def _move_photo_impl(
        self,
        photo_id: str,
        destination_dir: Union[str, Path],
        session: AsyncSession,
        create_dirs: bool,
        dry_run: bool
    ) -> Dict[str, Any]:
        """Internal move implementation."""
        from sqlalchemy import select
        
        # Get photo record
        stmt = select(Photo).where(Photo.id == photo_id)
        result = await session.execute(stmt)
        photo = result.scalar_one_or_none()
        
        if not photo:
            raise ValueError(f"Photo not found: {photo_id}")
        
        current_path = Path(photo.current_path)
        if not current_path.exists():
            raise FileNotFoundError(f"Photo file not found: {current_path}")
        
        destination_dir = Path(destination_dir)
        new_path = destination_dir / current_path.name
        
        # Check if target directory exists
        if not destination_dir.exists():
            if create_dirs:
                if not dry_run:
                    destination_dir.mkdir(parents=True, exist_ok=True)
            else:
                raise FileNotFoundError(f"Destination directory not found: {destination_dir}")
        
        # Check if target file exists
        if new_path.exists() and new_path != current_path:
            raise FileExistsError(f"Target file already exists: {new_path}")
        
        if dry_run:
            return {
                'photo_id': photo_id,
                'current_path': str(current_path),
                'new_path': str(new_path),
                'operation': 'move',
                'dry_run': True,
                'success': True
            }
        
        try:
            # Record operation
            operation = await self._create_operation_record(
                session, photo_id, 'move',
                {'old_path': str(current_path), 'new_path': str(new_path)}
            )
            
            # Perform move
            shutil.move(str(current_path), str(new_path))
            
            # Update photo record
            photo.current_path = str(new_path)
            photo.updated_at = datetime.utcnow()
            
            # Update operation status
            operation.status = 'completed'
            operation.completed_at = datetime.utcnow()
            
            await session.commit()
            
            logger.info(f"Moved photo {photo_id}: {current_path} -> {new_path}")
            
            return {
                'photo_id': photo_id,
                'current_path': str(current_path),
                'new_path': str(new_path),
                'operation': 'move',
                'operation_id': operation.id,
                'dry_run': False,
                'success': True
            }
            
        except Exception as e:
            if 'operation' in locals():
                operation.status = 'failed'
                operation.error_message = str(e)
                operation.completed_at = datetime.utcnow()
                await session.commit()
            
            logger.error(f"Failed to move photo {photo_id}: {e}")
            raise
    
    async def create_thumbnail(
        self,
        photo_id: str,
        thumbnail_dir: Union[str, Path],
        size: Tuple[int, int] = (200, 200),
        quality: int = 85,
        session: Optional[AsyncSession] = None
    ) -> Dict[str, Any]:
        """Create thumbnail for photo."""
        if session is None:
            async with get_async_db_session() as session:
                return await self._create_thumbnail_impl(
                    photo_id, thumbnail_dir, size, quality, session
                )
        else:
            return await self._create_thumbnail_impl(
                photo_id, thumbnail_dir, size, quality, session
            )
    
    async def _create_thumbnail_impl(
        self,
        photo_id: str,
        thumbnail_dir: Union[str, Path],
        size: Tuple[int, int],
        quality: int,
        session: AsyncSession
    ) -> Dict[str, Any]:
        """Internal thumbnail creation implementation."""
        from sqlalchemy import select
        
        # Get photo record
        stmt = select(Photo).where(Photo.id == photo_id)
        result = await session.execute(stmt)
        photo = result.scalar_one_or_none()
        
        if not photo:
            raise ValueError(f"Photo not found: {photo_id}")
        
        current_path = Path(photo.current_path)
        if not current_path.exists():
            raise FileNotFoundError(f"Photo file not found: {current_path}")
        
        # Create thumbnail path
        thumbnail_dir = Path(thumbnail_dir)
        thumbnail_dir.mkdir(parents=True, exist_ok=True)
        
        thumb_filename = f"thumb_{size[0]}x{size[1]}_{current_path.stem}.jpg"
        thumbnail_path = thumbnail_dir / thumb_filename
        
        try:
            # Create thumbnail
            success = self.image_processor.create_thumbnail(
                current_path, thumbnail_path, size, quality
            )
            
            if not success:
                raise RuntimeError("Thumbnail creation failed")
            
            # Update photo record with thumbnail path
            if not photo.thumbnail_path:
                photo.thumbnail_path = str(thumbnail_path)
                photo.updated_at = datetime.utcnow()
                await session.commit()
            
            logger.info(f"Created thumbnail for photo {photo_id}: {thumbnail_path}")
            
            return {
                'photo_id': photo_id,
                'thumbnail_path': str(thumbnail_path),
                'size': size,
                'quality': quality,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Failed to create thumbnail for photo {photo_id}: {e}")
            raise
    
    async def process_batch_operations(
        self,
        operations: List[Dict[str, Any]],
        max_concurrent: int = 5,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """Process multiple photo operations concurrently."""
        results = []
        total_ops = len(operations)
        
        logger.info(f"Starting batch processing of {total_ops} operations")
        
        # Process in batches
        for i in range(0, total_ops, max_concurrent):
            batch = operations[i:i + max_concurrent]
            
            # Create tasks for batch
            batch_tasks = []
            for op in batch:
                if op['operation'] == 'rename':
                    task = self.rename_photo(
                        op['photo_id'], 
                        op['new_filename'],
                        dry_run=op.get('dry_run', False)
                    )
                elif op['operation'] == 'move':
                    task = self.move_photo(
                        op['photo_id'], 
                        op['destination_dir'],
                        dry_run=op.get('dry_run', False)
                    )
                elif op['operation'] == 'thumbnail':
                    task = self.create_thumbnail(
                        op['photo_id'], 
                        op['thumbnail_dir'],
                        size=op.get('size', (200, 200)),
                        quality=op.get('quality', 85)
                    )
                else:
                    # Unknown operation
                    results.append({
                        'operation': op,
                        'error': f"Unknown operation: {op['operation']}",
                        'success': False
                    })
                    continue
                
                batch_tasks.append(task)
            
            # Execute batch
            try:
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Operation failed: {result}")
                        results.append({
                            'operation': batch[j] if j < len(batch) else None,
                            'error': str(result),
                            'success': False
                        })
                    else:
                        result['operation'] = batch[j] if j < len(batch) else None
                        results.append(result)
                
                # Progress callback
                if progress_callback:
                    progress_callback(len(results), total_ops)
                
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                raise
        
        success_count = sum(1 for r in results if r.get('success', False))
        logger.info(f"Batch processing complete: {success_count}/{total_ops} successful")
        
        return results
    
    def generate_smart_filename(
        self,
        description: str,
        tags: List[str],
        date_taken: Optional[datetime] = None,
        camera_info: Optional[Dict[str, Any]] = None,
        max_length: int = 100
    ) -> str:
        """Generate intelligent filename based on analysis results."""
        parts = []
        
        # Add date prefix if available
        if date_taken:
            date_str = date_taken.strftime("%Y%m%d")
            parts.append(date_str)
        
        # Process description
        if description:
            # Extract key words from description
            desc_words = self._extract_key_words(description, max_words=3)
            if desc_words:
                parts.extend(desc_words)
        
        # Add most relevant tags
        if tags:
            # Sort tags by relevance/category
            relevant_tags = self._select_relevant_tags(tags, max_tags=2)
            parts.extend(relevant_tags)
        
        # Add camera info if distinctive
        if camera_info and camera_info.get('model'):
            model = camera_info['model'].lower()
            # Only add if it's a distinctive camera (professional models)
            if any(brand in model for brand in ['canon', 'nikon', 'sony', 'fuji']):
                # Use abbreviated form
                model_abbr = self._abbreviate_camera_model(model)
                if model_abbr:
                    parts.append(model_abbr)
        
        # Join parts and sanitize
        if not parts:
            parts = ['photo']
        
        filename = '_'.join(parts)
        filename = self._sanitize_filename(filename)
        
        # Ensure length limit
        if len(filename) > max_length:
            filename = filename[:max_length]
        
        return filename
    
    def _extract_key_words(self, description: str, max_words: int = 3) -> List[str]:
        """Extract key words from description."""
        if not description:
            return []
        
        # Simple keyword extraction
        # Remove common words and extract meaningful terms
        stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'this', 'that', 'is', 'are', 'was', 'were',
            'photo', 'image', 'picture', 'showing', 'shows', 'depicts'
        }
        
        words = description.lower().split()
        key_words = []
        
        for word in words:
            # Clean word
            word = ''.join(c for c in word if c.isalnum())
            
            if (len(word) > 2 and 
                word not in stop_words and 
                len(key_words) < max_words):
                key_words.append(word)
        
        return key_words
    
    def _select_relevant_tags(self, tags: List[str], max_tags: int = 2) -> List[str]:
        """Select most relevant tags for filename."""
        if not tags:
            return []
        
        # Priority order for tag categories
        priority_categories = [
            'nature', 'landscape', 'portrait', 'architecture', 
            'wildlife', 'sports', 'event', 'travel'
        ]
        
        # Sort tags by priority and length
        sorted_tags = sorted(tags, key=lambda t: (
            -self._get_tag_priority(t), len(t)
        ))
        
        return sorted_tags[:max_tags]
    
    def _get_tag_priority(self, tag: str) -> int:
        """Get priority score for tag."""
        high_priority = ['landscape', 'portrait', 'wildlife', 'sunset', 'mountain']
        medium_priority = ['nature', 'travel', 'architecture', 'beach', 'forest']
        
        if tag in high_priority:
            return 3
        elif tag in medium_priority:
            return 2
        else:
            return 1
    
    def _abbreviate_camera_model(self, model: str) -> Optional[str]:
        """Create abbreviated camera model name."""
        abbreviations = {
            'canon': 'can',
            'nikon': 'nik',
            'sony': 'son',
            'fujifilm': 'fuji',
            'olympus': 'oly',
            'panasonic': 'pan'
        }
        
        model_lower = model.lower()
        for brand, abbr in abbreviations.items():
            if brand in model_lower:
                return abbr
        
        return None
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for filesystem compatibility."""
        # Replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Replace spaces with underscores
        filename = filename.replace(' ', '_')
        
        # Remove multiple consecutive underscores
        while '__' in filename:
            filename = filename.replace('__', '_')
        
        # Remove leading/trailing underscores
        filename = filename.strip('_')
        
        # Ensure not empty
        if not filename:
            filename = 'photo'
        
        return filename
    
    async def _create_operation_record(
        self,
        session: AsyncSession,
        photo_id: str,
        operation_type: str,
        metadata: Dict[str, Any]
    ) -> Organization:
        """Create operation record for tracking."""
        operation = Organization(
            id=str(uuid.uuid4()),
            photo_id=photo_id,
            operation_type=operation_type,
            status='pending',
            metadata=metadata,
            created_at=datetime.utcnow()
        )
        
        session.add(operation)
        await session.flush()
        
        return operation
    
    async def rollback_operation(
        self,
        operation_id: str,
        session: Optional[AsyncSession] = None
    ) -> Dict[str, Any]:
        """Rollback a photo operation."""
        if session is None:
            async with get_async_db_session() as session:
                return await self._rollback_operation_impl(operation_id, session)
        else:
            return await self._rollback_operation_impl(operation_id, session)
    
    async def _rollback_operation_impl(
        self,
        operation_id: str,
        session: AsyncSession
    ) -> Dict[str, Any]:
        """Internal rollback implementation."""
        from sqlalchemy import select
        
        # Get operation record
        stmt = select(Organization).where(Organization.id == operation_id)
        result = await session.execute(stmt)
        operation = result.scalar_one_or_none()
        
        if not operation:
            raise ValueError(f"Operation not found: {operation_id}")
        
        if operation.status != 'completed':
            raise ValueError(f"Can only rollback completed operations")
        
        try:
            metadata = operation.metadata
            
            if operation.operation_type == 'rename':
                # Reverse rename
                old_path = Path(metadata['old_path'])
                new_path = Path(metadata['new_path'])
                
                if new_path.exists():
                    new_path.rename(old_path)
                    
                    # Update photo record
                    photo_stmt = select(Photo).where(Photo.id == operation.photo_id)
                    photo_result = await session.execute(photo_stmt)
                    photo = photo_result.scalar_one_or_none()
                    
                    if photo:
                        photo.current_path = str(old_path)
                        photo.filename = old_path.name
                        photo.updated_at = datetime.utcnow()
            
            elif operation.operation_type == 'move':
                # Reverse move
                old_path = Path(metadata['old_path'])
                new_path = Path(metadata['new_path'])
                
                if new_path.exists():
                    shutil.move(str(new_path), str(old_path))
                    
                    # Update photo record
                    photo_stmt = select(Photo).where(Photo.id == operation.photo_id)
                    photo_result = await session.execute(photo_stmt)
                    photo = photo_result.scalar_one_or_none()
                    
                    if photo:
                        photo.current_path = str(old_path)
                        photo.updated_at = datetime.utcnow()
            
            # Mark operation as rolled back
            operation.status = 'rolled_back'
            operation.completed_at = datetime.utcnow()
            
            await session.commit()
            
            logger.info(f"Rolled back operation {operation_id}")
            
            return {
                'operation_id': operation_id,
                'operation_type': operation.operation_type,
                'rollback_success': True
            }
            
        except Exception as e:
            operation.status = 'rollback_failed'
            operation.error_message = f"Rollback failed: {str(e)}"
            operation.completed_at = datetime.utcnow()
            await session.commit()
            
            logger.error(f"Failed to rollback operation {operation_id}: {e}")
            raise
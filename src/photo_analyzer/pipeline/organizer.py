"""Photo organization system with date-based hierarchy and symbolic links."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Set
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import get_config
from ..core.logger import get_logger
from ..database.session import get_async_db_session
from ..models.photo import Photo, Tag
from ..models.organization import OrganizationOperation, OrganizationRule, SymbolicLink
from ..utils.date_utils import DateUtils
from ..utils.file_utils import FileUtils
from .processor import PhotoProcessor

logger = get_logger(__name__)


class PhotoOrganizer:
    """Handles photo organization with date-based structure and symbolic links."""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.processor = PhotoProcessor(config)
        self.logger = logger
    
    async def organize_photo(
        self,
        photo_id: str,
        base_directory: Union[str, Path],
        organization_rules: Optional[Dict[str, Any]] = None,
        session: Optional[AsyncSession] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Organize a single photo into date-based structure."""
        if session is None:
            async with get_async_db_session() as session:
                return await self._organize_photo_impl(
                    photo_id, base_directory, organization_rules, session, dry_run
                )
        else:
            return await self._organize_photo_impl(
                photo_id, base_directory, organization_rules, session, dry_run
            )
    
    async def _organize_photo_impl(
        self,
        photo_id: str,
        base_directory: Union[str, Path],
        organization_rules: Optional[Dict[str, Any]],
        session: AsyncSession,
        dry_run: bool
    ) -> Dict[str, Any]:
        """Internal photo organization implementation."""
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload
        
        # Get photo with tags
        stmt = select(Photo).options(selectinload(Photo.tags)).where(Photo.id == photo_id)
        result = await session.execute(stmt)
        photo = result.scalar_one_or_none()
        
        if not photo:
            raise ValueError(f"Photo not found: {photo_id}")
        
        current_path = Path(photo.current_path)
        if not current_path.exists():
            raise FileNotFoundError(f"Photo file not found: {current_path}")
        
        base_directory = Path(base_directory)
        
        # Use default rules if none provided
        if organization_rules is None:
            organization_rules = self._get_default_organization_rules()
        
        # Determine target date for organization
        target_date = self._determine_target_date(photo)
        if not target_date:
            raise ValueError(f"Cannot determine date for photo {photo_id}")
        
        # Create date-based directory structure
        date_path = self._create_date_path(base_directory, target_date, organization_rules)
        
        # Determine new filename
        new_filename = await self._determine_new_filename(photo, organization_rules)
        
        target_path = date_path / new_filename
        
        # Check if file already exists
        if target_path.exists() and target_path != current_path:
            target_path = self._resolve_filename_conflict(target_path)
        
        operations = []
        symlinks = []
        
        # Plan the move operation
        operations.append({
            'type': 'move',
            'photo_id': photo_id,
            'source_path': str(current_path),
            'target_path': str(target_path),
            'date_path': str(date_path)
        })
        
        # Plan symbolic link creation
        symlink_dirs = await self._plan_symbolic_links(
            photo, target_path, base_directory, organization_rules
        )
        
        for symlink_dir, link_name in symlink_dirs:
            symlink_path = symlink_dir / link_name
            symlinks.append({
                'type': 'symlink',
                'photo_id': photo_id,
                'target_path': str(target_path),
                'link_path': str(symlink_path),
                'category': symlink_dir.name
            })
        
        if dry_run:
            return {
                'photo_id': photo_id,
                'current_path': str(current_path),
                'target_path': str(target_path),
                'date_path': str(date_path),
                'operations': operations,
                'symlinks': symlinks,
                'dry_run': True,
                'success': True
            }
        
        try:
            # Execute move operation
            if current_path != target_path:
                # Ensure target directory exists
                date_path.mkdir(parents=True, exist_ok=True)
                
                # Record operation
                operation = await self._create_operation_record(
                    session, photo_id, 'organize',
                    {
                        'old_path': str(current_path),
                        'new_path': str(target_path),
                        'date_path': str(date_path),
                        'organization_rules': organization_rules
                    }
                )
                
                # Move file
                FileUtils.safe_move_file(current_path, target_path, backup_existing=False)
                
                # Update photo record
                photo.current_path = str(target_path)
                photo.filename = new_filename
                photo.organized_at = datetime.utcnow()
                photo.updated_at = datetime.utcnow()
                
                operation.status = 'completed'
                operation.completed_at = datetime.utcnow()
            
            # Create symbolic links
            created_symlinks = []
            for symlink_info in symlinks:
                symlink_result = await self._create_symbolic_link(
                    session, photo_id, 
                    Path(symlink_info['target_path']),
                    Path(symlink_info['link_path']),
                    symlink_info['category']
                )
                if symlink_result:
                    created_symlinks.append(symlink_result)
            
            await session.commit()
            
            logger.info(f"Organized photo {photo_id}: {current_path} -> {target_path}")
            logger.info(f"Created {len(created_symlinks)} symbolic links")
            
            return {
                'photo_id': photo_id,
                'current_path': str(current_path),
                'target_path': str(target_path),
                'date_path': str(date_path),
                'operations': operations,
                'symlinks': created_symlinks,
                'dry_run': False,
                'success': True
            }
            
        except Exception as e:
            if 'operation' in locals():
                operation.status = 'failed'
                operation.error_message = str(e)
                operation.completed_at = datetime.utcnow()
                await session.commit()
            
            logger.error(f"Failed to organize photo {photo_id}: {e}")
            raise
    
    async def organize_batch(
        self,
        photo_ids: List[str],
        base_directory: Union[str, Path],
        organization_rules: Optional[Dict[str, Any]] = None,
        max_concurrent: int = 5,
        progress_callback: Optional[callable] = None,
        dry_run: bool = False
    ) -> List[Dict[str, Any]]:
        """Organize multiple photos concurrently."""
        results = []
        total_photos = len(photo_ids)
        
        logger.info(f"Starting batch organization of {total_photos} photos")
        
        # Process in batches
        for i in range(0, total_photos, max_concurrent):
            batch = photo_ids[i:i + max_concurrent]
            
            # Create tasks for batch
            batch_tasks = [
                self.organize_photo(
                    photo_id, base_directory, organization_rules, dry_run=dry_run
                )
                for photo_id in batch
            ]
            
            try:
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Failed to organize {batch[j]}: {result}")
                        results.append({
                            'photo_id': batch[j],
                            'error': str(result),
                            'success': False
                        })
                    else:
                        results.append(result)
                
                # Progress callback
                if progress_callback:
                    progress_callback(len(results), total_photos)
                
            except Exception as e:
                logger.error(f"Batch organization failed: {e}")
                raise
        
        success_count = sum(1 for r in results if r.get('success', False))
        logger.info(f"Batch organization complete: {success_count}/{total_photos} successful")
        
        return results
    
    def _get_default_organization_rules(self) -> Dict[str, Any]:
        """Get default organization rules."""
        return {
            'date_format': 'YYYY/MM/DD',  # Year/Month/Day
            'filename_strategy': 'smart',  # smart, preserve, or custom
            'create_symlinks': True,
            'symlink_categories': [
                'tags',       # By photo tags
                'camera',     # By camera model
                'year',       # By year only
                'month',      # By month
            ],
            'conflict_resolution': 'append_number',  # append_number, overwrite, skip
            'preserve_original_structure': False,
        }
    
    def _determine_target_date(self, photo: Photo) -> Optional[datetime]:
        """Determine the best date to use for organization."""
        # Priority: EXIF date_taken > file creation > file modification
        if photo.date_taken:
            return photo.date_taken
        
        # Try to extract date from filename
        filename_date = DateUtils.extract_date_from_filename(photo.filename)
        if filename_date:
            return filename_date
        
        # Use file system dates as fallback
        current_path = Path(photo.current_path)
        if current_path.exists():
            stat = current_path.stat()
            return datetime.fromtimestamp(stat.st_ctime)
        
        return None
    
    def _create_date_path(
        self, 
        base_directory: Path, 
        target_date: datetime, 
        rules: Dict[str, Any]
    ) -> Path:
        """Create date-based directory path."""
        date_format = rules.get('date_format', 'YYYY/MM/DD')
        
        if date_format == 'YYYY/MM/DD':
            return base_directory / str(target_date.year) / f"{target_date.month:02d}" / f"{target_date.day:02d}"
        elif date_format == 'YYYY/MM':
            return base_directory / str(target_date.year) / f"{target_date.month:02d}"
        elif date_format == 'YYYY':
            return base_directory / str(target_date.year)
        elif date_format == 'YYYY/QN':
            quarter = (target_date.month - 1) // 3 + 1
            return base_directory / str(target_date.year) / f"Q{quarter}"
        else:
            # Custom format
            formatted_date = DateUtils.format_date_for_path(target_date, date_format)
            return base_directory / formatted_date
    
    async def _determine_new_filename(
        self, 
        photo: Photo, 
        rules: Dict[str, Any]
    ) -> str:
        """Determine new filename for organized photo."""
        strategy = rules.get('filename_strategy', 'smart')
        
        if strategy == 'preserve':
            return photo.filename
        elif strategy == 'smart':
            # Use intelligent filename generation
            description = photo.description or ''
            tag_names = [tag.name for tag in photo.tags] if photo.tags else []
            
            # Extract camera info from EXIF
            camera_info = None
            if photo.exif_data:
                camera_info = {
                    'make': photo.exif_data.get('camera_make'),
                    'model': photo.exif_data.get('camera_model')
                }
            
            new_name = self.processor.generate_smart_filename(
                description, tag_names, photo.date_taken, camera_info
            )
            
            # Preserve original extension
            original_ext = Path(photo.filename).suffix
            if not new_name.endswith(original_ext):
                new_name += original_ext
            
            return new_name
        else:
            # Custom strategy (would be implemented based on rules)
            return photo.filename
    
    def _resolve_filename_conflict(self, target_path: Path) -> Path:
        """Resolve filename conflicts by appending number."""
        counter = 1
        stem = target_path.stem
        suffix = target_path.suffix
        parent = target_path.parent
        
        while target_path.exists():
            new_name = f"{stem}_{counter}{suffix}"
            target_path = parent / new_name
            counter += 1
        
        return target_path
    
    async def _plan_symbolic_links(
        self,
        photo: Photo,
        target_path: Path,
        base_directory: Path,
        rules: Dict[str, Any]
    ) -> List[tuple[Path, str]]:
        """Plan symbolic link creation for photo."""
        if not rules.get('create_symlinks', True):
            return []
        
        symlink_dirs = []
        categories = rules.get('symlink_categories', [])
        
        for category in categories:
            if category == 'tags' and photo.tags:
                # Create symlinks by tags
                tags_dir = base_directory / 'by_tags'
                for tag in photo.tags:
                    tag_dir = tags_dir / tag.name
                    symlink_dirs.append((tag_dir, target_path.name))
            
            elif category == 'camera' and photo.exif_data:
                # Create symlinks by camera
                camera_make = photo.exif_data.get('camera_make')
                camera_model = photo.exif_data.get('camera_model')
                
                if camera_make or camera_model:
                    cameras_dir = base_directory / 'by_camera'
                    if camera_make and camera_model:
                        camera_dir = cameras_dir / f"{camera_make}_{camera_model}"
                    elif camera_make:
                        camera_dir = cameras_dir / camera_make
                    else:
                        camera_dir = cameras_dir / camera_model
                    
                    symlink_dirs.append((camera_dir, target_path.name))
            
            elif category == 'year' and photo.date_taken:
                # Create symlinks by year
                year_dir = base_directory / 'by_year' / str(photo.date_taken.year)
                symlink_dirs.append((year_dir, target_path.name))
            
            elif category == 'month' and photo.date_taken:
                # Create symlinks by month
                month_name = photo.date_taken.strftime('%B')
                month_dir = base_directory / 'by_month' / month_name
                symlink_dirs.append((month_dir, target_path.name))
        
        return symlink_dirs
    
    async def _create_symbolic_link(
        self,
        session: AsyncSession,
        photo_id: str,
        target_path: Path,
        link_path: Path,
        category: str
    ) -> Optional[Dict[str, Any]]:
        """Create a symbolic link and record it in database."""
        try:
            # Ensure link directory exists
            link_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create symbolic link
            if FileUtils.create_symlink(target_path, link_path, force=True):
                # Record in database
                symlink = SymbolicLink(
                    id=str(uuid.uuid4()),
                    photo_id=photo_id,
                    target_path=str(target_path),
                    link_path=str(link_path),
                    category=category,
                    created_at=datetime.utcnow()
                )
                
                session.add(symlink)
                await session.flush()
                
                return {
                    'symlink_id': symlink.id,
                    'target_path': str(target_path),
                    'link_path': str(link_path),
                    'category': category,
                    'success': True
                }
        
        except Exception as e:
            logger.error(f"Failed to create symbolic link {link_path} -> {target_path}: {e}")
        
        return None
    
    async def _create_operation_record(
        self,
        session: AsyncSession,
        photo_id: str,
        operation_type: str,
        metadata: Dict[str, Any]
    ) -> OrganizationOperation:
        """Create operation record for tracking."""
        operation = OrganizationOperation(
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
    
    async def cleanup_broken_symlinks(
        self,
        base_directory: Union[str, Path],
        session: Optional[AsyncSession] = None
    ) -> Dict[str, Any]:
        """Clean up broken symbolic links."""
        if session is None:
            async with get_async_db_session() as session:
                return await self._cleanup_broken_symlinks_impl(base_directory, session)
        else:
            return await self._cleanup_broken_symlinks_impl(base_directory, session)
    
    async def _cleanup_broken_symlinks_impl(
        self,
        base_directory: Union[str, Path],
        session: AsyncSession
    ) -> Dict[str, Any]:
        """Internal cleanup implementation."""
        from sqlalchemy import select, delete
        
        base_directory = Path(base_directory)
        removed_links = []
        updated_records = 0
        
        # Find all symbolic link records
        stmt = select(SymbolicLink)
        result = await session.execute(stmt)
        symlinks = result.scalars().all()
        
        for symlink in symlinks:
            link_path = Path(symlink.link_path)
            target_path = Path(symlink.target_path)
            
            # Check if link exists and is valid
            if link_path.is_symlink():
                if not target_path.exists():
                    # Broken symlink - remove it
                    try:
                        link_path.unlink()
                        removed_links.append(str(link_path))
                        
                        # Remove from database
                        await session.delete(symlink)
                        updated_records += 1
                        
                        logger.debug(f"Removed broken symlink: {link_path}")
                        
                    except Exception as e:
                        logger.error(f"Failed to remove broken symlink {link_path}: {e}")
            
            elif not link_path.exists():
                # Link doesn't exist - remove from database
                await session.delete(symlink)
                updated_records += 1
                logger.debug(f"Removed non-existent symlink record: {link_path}")
        
        await session.commit()
        
        logger.info(f"Cleanup complete: removed {len(removed_links)} broken symlinks, updated {updated_records} records")
        
        return {
            'removed_symlinks': removed_links,
            'updated_records': updated_records,
            'success': True
        }
    
    async def rebuild_symlinks_for_photo(
        self,
        photo_id: str,
        base_directory: Union[str, Path],
        organization_rules: Optional[Dict[str, Any]] = None,
        session: Optional[AsyncSession] = None
    ) -> Dict[str, Any]:
        """Rebuild all symbolic links for a photo."""
        if session is None:
            async with get_async_db_session() as session:
                return await self._rebuild_symlinks_impl(
                    photo_id, base_directory, organization_rules, session
                )
        else:
            return await self._rebuild_symlinks_impl(
                photo_id, base_directory, organization_rules, session
            )
    
    async def _rebuild_symlinks_impl(
        self,
        photo_id: str,
        base_directory: Union[str, Path],
        organization_rules: Optional[Dict[str, Any]],
        session: AsyncSession
    ) -> Dict[str, Any]:
        """Internal symlink rebuild implementation."""
        from sqlalchemy import select, delete
        from sqlalchemy.orm import selectinload
        
        # Get photo with tags
        stmt = select(Photo).options(selectinload(Photo.tags)).where(Photo.id == photo_id)
        result = await session.execute(stmt)
        photo = result.scalar_one_or_none()
        
        if not photo:
            raise ValueError(f"Photo not found: {photo_id}")
        
        # Remove existing symlinks
        delete_stmt = delete(SymbolicLink).where(SymbolicLink.photo_id == photo_id)
        await session.execute(delete_stmt)
        
        # Use default rules if none provided
        if organization_rules is None:
            organization_rules = self._get_default_organization_rules()
        
        # Plan new symlinks
        target_path = Path(photo.current_path)
        symlink_dirs = await self._plan_symbolic_links(
            photo, target_path, Path(base_directory), organization_rules
        )
        
        # Create new symlinks
        created_symlinks = []
        for symlink_dir, link_name in symlink_dirs:
            symlink_result = await self._create_symbolic_link(
                session, photo_id, target_path,
                symlink_dir / link_name, symlink_dir.name
            )
            if symlink_result:
                created_symlinks.append(symlink_result)
        
        await session.commit()
        
        logger.info(f"Rebuilt {len(created_symlinks)} symlinks for photo {photo_id}")
        
        return {
            'photo_id': photo_id,
            'created_symlinks': created_symlinks,
            'success': True
        }
    
    async def get_organization_stats(
        self,
        base_directory: Union[str, Path],
        session: Optional[AsyncSession] = None
    ) -> Dict[str, Any]:
        """Get organization statistics."""
        if session is None:
            async with get_async_db_session() as session:
                return await self._get_organization_stats_impl(base_directory, session)
        else:
            return await self._get_organization_stats_impl(base_directory, session)
    
    async def _get_organization_stats_impl(
        self,
        base_directory: Union[str, Path],
        session: AsyncSession
    ) -> Dict[str, Any]:
        """Internal stats implementation."""
        from sqlalchemy import select, func
        
        base_directory = Path(base_directory)
        
        # Count photos by organization status
        organized_stmt = select(func.count(Photo.id)).where(Photo.organized_at.isnot(None))
        organized_count = await session.scalar(organized_stmt) or 0
        
        total_stmt = select(func.count(Photo.id))
        total_count = await session.scalar(total_stmt) or 0
        
        # Count symlinks
        symlink_stmt = select(func.count(SymbolicLink.id))
        symlink_count = await session.scalar(symlink_stmt) or 0
        
        # Count operations
        operation_stmt = select(func.count(OrganizationOperation.id))
        operation_count = await session.scalar(operation_stmt) or 0
        
        # Calculate directory statistics
        dir_stats = self._calculate_directory_stats(base_directory)
        
        return {
            'total_photos': total_count,
            'organized_photos': organized_count,
            'unorganized_photos': total_count - organized_count,
            'organization_percentage': (organized_count / total_count * 100) if total_count > 0 else 0,
            'total_symlinks': symlink_count,
            'total_operations': operation_count,
            'directory_stats': dir_stats,
        }
    
    def _calculate_directory_stats(self, base_directory: Path) -> Dict[str, Any]:
        """Calculate directory structure statistics."""
        if not base_directory.exists():
            return {'error': 'Directory not found'}
        
        stats = {
            'total_directories': 0,
            'date_directories': 0,
            'symlink_directories': 0,
            'total_files': 0,
            'total_symlinks': 0,
        }
        
        try:
            for item in base_directory.rglob('*'):
                if item.is_dir():
                    stats['total_directories'] += 1
                    
                    # Check if it's a date directory (YYYY, MM, DD pattern)
                    if item.name.isdigit() and len(item.name) in [2, 4]:
                        stats['date_directories'] += 1
                    
                    # Check if it's a symlink category directory
                    if item.name.startswith('by_'):
                        stats['symlink_directories'] += 1
                
                elif item.is_file():
                    if item.is_symlink():
                        stats['total_symlinks'] += 1
                    else:
                        stats['total_files'] += 1
        
        except Exception as e:
            stats['error'] = str(e)
        
        return stats
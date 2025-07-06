"""Core photo analysis pipeline."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import get_config
from ..core.logger import get_logger
from ..database.session import get_async_db_session
from ..models.photo import Photo, Tag
from ..models.analysis import AnalysisSession, AnalysisResult
from ..analyzer.llm_client import OllamaClient
from ..utils.image import ImageProcessor
from ..utils.exif import ExifExtractor
from ..utils.date_utils import DateUtils
from ..utils.file_utils import FileUtils

logger = get_logger(__name__)


class PhotoAnalyzer:
    """Main photo analysis pipeline."""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.llm_client = OllamaClient(self.config)
        self.image_processor = ImageProcessor()
        self.exif_extractor = ExifExtractor()
        self.logger = logger
    
    async def analyze_photo(
        self, 
        file_path: Union[str, Path],
        session: Optional[AsyncSession] = None,
        analysis_session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze a single photo and store results."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Photo file not found: {file_path}")
        
        try:
            # Use provided session or create new one
            if session is None:
                async with get_async_db_session() as session:
                    return await self._analyze_photo_impl(
                        file_path, session, analysis_session_id
                    )
            else:
                return await self._analyze_photo_impl(
                    file_path, session, analysis_session_id
                )
                
        except Exception as e:
            logger.error(f"Failed to analyze photo {file_path}: {e}")
            raise
    
    async def _analyze_photo_impl(
        self,
        file_path: Path,
        session: AsyncSession,
        analysis_session_id: Optional[str]
    ) -> Dict[str, Any]:
        """Internal photo analysis implementation."""
        logger.info(f"Analyzing photo: {file_path}")
        
        # Validate image
        validation = self.image_processor.validate_image(file_path)
        if not validation['is_valid']:
            raise ValueError(f"Invalid image: {validation['errors']}")
        
        # Extract image metadata
        image_info = self.image_processor.get_image_info(file_path)
        exif_data = self.exif_extractor.extract_exif(file_path)
        
        # Create or get photo record
        photo = await self._get_or_create_photo(session, file_path, image_info, exif_data)
        
        # Create analysis session if needed
        if analysis_session_id is None:
            analysis_session = await self._create_analysis_session(session)
            analysis_session_id = analysis_session.id
        
        # Perform LLM analysis
        llm_results = await self._perform_llm_analysis(file_path)
        
        # Create analysis result record
        analysis_result = await self._create_analysis_result(
            session, photo.id, analysis_session_id, llm_results, image_info, exif_data
        )
        
        # Process tags
        tags = await self._process_tags(session, photo, llm_results.get('tags', []))
        
        # Update photo with analysis results
        await self._update_photo_analysis(session, photo, analysis_result, tags)
        
        # Commit changes
        await session.commit()
        
        result = {
            'photo_id': photo.id,
            'analysis_id': analysis_result.id,
            'session_id': analysis_session_id,
            'description': llm_results.get('description', ''),
            'tags': [tag.name for tag in tags],
            'suggested_filename': llm_results.get('suggested_filename', ''),
            'confidence': llm_results.get('confidence', 0.0),
            'date_taken': exif_data.get('date_taken'),
            'camera_info': self._extract_camera_info(exif_data),
            'location': self._extract_location_info(exif_data),
            'image_properties': {
                'width': image_info['width'],
                'height': image_info['height'],
                'format': image_info['format'],
                'size': image_info['file_size'],
                'orientation': image_info['orientation'],
            }
        }
        
        logger.info(f"Analysis complete for {file_path}: {len(tags)} tags, confidence {result['confidence']:.2f}")
        
        return result
    
    async def _get_or_create_photo(
        self,
        session: AsyncSession,
        file_path: Path,
        image_info: Dict[str, Any],
        exif_data: Dict[str, Any]
    ) -> Photo:
        """Get existing photo or create new record."""
        from sqlalchemy import select
        
        # Check if photo already exists by hash
        file_hash = image_info['file_hash']
        stmt = select(Photo).where(Photo.file_hash == file_hash)
        result = await session.execute(stmt)
        existing_photo = result.scalar_one_or_none()
        
        if existing_photo:
            # Update path if changed
            if existing_photo.current_path != str(file_path):
                existing_photo.current_path = str(file_path)
                existing_photo.updated_at = datetime.utcnow()
            return existing_photo
        
        # Create new photo record
        photo = Photo(
            id=str(uuid.uuid4()),
            original_path=str(file_path),
            current_path=str(file_path),
            filename=file_path.name,
            file_hash=file_hash,
            file_size=image_info['file_size'],
            width=image_info['width'],
            height=image_info['height'],
            format=image_info['format'],
            date_taken=exif_data.get('date_taken'),
            exif_data=exif_data,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        session.add(photo)
        await session.flush()  # Get the ID
        
        return photo
    
    async def _create_analysis_session(self, session: AsyncSession) -> AnalysisSession:
        """Create new analysis session."""
        analysis_session = AnalysisSession(
            id=str(uuid.uuid4()),
            model_name=self.config.llm.default_model,
            model_version="unknown",  # Could be fetched from Ollama
            started_at=datetime.utcnow(),
            status="active"
        )
        
        session.add(analysis_session)
        await session.flush()
        
        return analysis_session
    
    async def _perform_llm_analysis(self, file_path: Path) -> Dict[str, Any]:
        """Perform LLM analysis on the image."""
        try:
            # Check LLM health first
            if not await self.llm_client.health_check():
                raise ConnectionError("LLM service is not available")
            
            # Analyze image
            analysis_result = await self.llm_client.analyze_image(file_path)
            
            # Generate suggested filename
            filename_result = await self.llm_client.generate_filename(
                file_path, 
                description=analysis_result.get('description', ''),
                tags=analysis_result.get('tags', [])
            )
            
            # Combine results
            result = {
                'description': analysis_result.get('description', ''),
                'tags': analysis_result.get('tags', []),
                'confidence': analysis_result.get('confidence', 0.0),
                'suggested_filename': filename_result.get('filename', ''),
                'model_used': analysis_result.get('model_used', ''),
                'processing_time': analysis_result.get('processing_time', 0.0),
            }
            
            return result
            
        except Exception as e:
            logger.error(f"LLM analysis failed for {file_path}: {e}")
            # Return fallback results
            return {
                'description': 'Analysis failed',
                'tags': [],
                'confidence': 0.0,
                'suggested_filename': file_path.name,
                'model_used': 'none',
                'processing_time': 0.0,
                'error': str(e)
            }
    
    async def _create_analysis_result(
        self,
        session: AsyncSession,
        photo_id: str,
        analysis_session_id: str,
        llm_results: Dict[str, Any],
        image_info: Dict[str, Any],
        exif_data: Dict[str, Any]
    ) -> AnalysisResult:
        """Create analysis result record."""
        analysis_result = AnalysisResult(
            id=str(uuid.uuid4()),
            photo_id=photo_id,
            session_id=analysis_session_id,
            model_name=llm_results.get('model_used', ''),
            description=llm_results.get('description', ''),
            confidence_score=llm_results.get('confidence', 0.0),
            processing_time=llm_results.get('processing_time', 0.0),
            raw_output=llm_results,
            metadata={
                'image_info': image_info,
                'exif_summary': self.exif_extractor.extract_summary(
                    # Note: We'd need the file path here, but we already have exif_data
                    # This is a design consideration for optimization
                ),
                'analysis_timestamp': datetime.utcnow().isoformat(),
            },
            created_at=datetime.utcnow()
        )
        
        session.add(analysis_result)
        await session.flush()
        
        return analysis_result
    
    async def _process_tags(
        self, 
        session: AsyncSession, 
        photo: Photo, 
        tag_names: List[str]
    ) -> List[Tag]:
        """Process and create tags for photo."""
        from sqlalchemy import select
        
        tags = []
        
        for tag_name in tag_names:
            if not tag_name or not tag_name.strip():
                continue
                
            tag_name = tag_name.strip().lower()
            
            # Get or create tag
            stmt = select(Tag).where(Tag.name == tag_name)
            result = await session.execute(stmt)
            tag = result.scalar_one_or_none()
            
            if not tag:
                tag = Tag(
                    id=str(uuid.uuid4()),
                    name=tag_name,
                    category=self._categorize_tag(tag_name),
                    created_at=datetime.utcnow()
                )
                session.add(tag)
                await session.flush()
            
            tags.append(tag)
            
            # Create photo-tag association if not exists
            if tag not in photo.tags:
                photo.tags.append(tag)
        
        return tags
    
    def _categorize_tag(self, tag_name: str) -> str:
        """Categorize tag based on name."""
        # Simple categorization logic
        nature_tags = {'landscape', 'mountain', 'tree', 'forest', 'ocean', 'sky', 'sunset', 'sunrise'}
        people_tags = {'person', 'people', 'family', 'child', 'adult', 'portrait'}
        object_tags = {'car', 'building', 'house', 'food', 'animal', 'dog', 'cat'}
        activity_tags = {'sports', 'running', 'swimming', 'hiking', 'travel', 'vacation'}
        
        if tag_name in nature_tags:
            return 'nature'
        elif tag_name in people_tags:
            return 'people'
        elif tag_name in object_tags:
            return 'objects'
        elif tag_name in activity_tags:
            return 'activities'
        else:
            return 'general'
    
    async def _update_photo_analysis(
        self,
        session: AsyncSession,
        photo: Photo,
        analysis_result: AnalysisResult,
        tags: List[Tag]
    ):
        """Update photo with analysis results."""
        photo.description = analysis_result.description
        photo.analysis_confidence = analysis_result.confidence_score
        photo.last_analyzed = datetime.utcnow()
        photo.updated_at = datetime.utcnow()
        
        # Update tag count
        photo.tag_count = len(tags)
    
    def _extract_camera_info(self, exif_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract camera information from EXIF data."""
        if not exif_data:
            return None
        
        camera_info = {}
        
        if 'camera_make' in exif_data:
            camera_info['make'] = exif_data['camera_make']
        
        if 'camera_model' in exif_data:
            camera_info['model'] = exif_data['camera_model']
        
        if 'lens_model' in exif_data:
            camera_info['lens'] = exif_data['lens_model']
        
        # Settings
        settings = {}
        for key in ['iso', 'aperture', 'shutter_speed', 'focal_length']:
            if key in exif_data:
                settings[key] = exif_data[key]
        
        if settings:
            camera_info['settings'] = settings
        
        return camera_info if camera_info else None
    
    def _extract_location_info(self, exif_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract location information from EXIF data."""
        gps_data = exif_data.get('GPS', {})
        
        if 'latitude' in gps_data and 'longitude' in gps_data:
            location = {
                'latitude': gps_data['latitude'],
                'longitude': gps_data['longitude']
            }
            
            if 'altitude' in gps_data:
                location['altitude'] = gps_data['altitude']
            
            return location
        
        return None
    
    async def analyze_batch(
        self,
        file_paths: List[Union[str, Path]],
        batch_size: int = 5,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """Analyze multiple photos in batches."""
        results = []
        total_files = len(file_paths)
        
        # Create analysis session for the batch
        async with get_async_db_session() as session:
            analysis_session = await self._create_analysis_session(session)
            await session.commit()
            
            session_id = analysis_session.id
        
        logger.info(f"Starting batch analysis of {total_files} photos")
        
        # Process in batches
        for i in range(0, total_files, batch_size):
            batch = file_paths[i:i + batch_size]
            
            # Analyze batch concurrently
            batch_tasks = [
                self.analyze_photo(file_path, analysis_session_id=session_id)
                for file_path in batch
            ]
            
            try:
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Failed to analyze {batch[j]}: {result}")
                        results.append({
                            'file_path': str(batch[j]),
                            'error': str(result),
                            'success': False
                        })
                    else:
                        result['file_path'] = str(batch[j])
                        result['success'] = True
                        results.append(result)
                
                # Progress callback
                if progress_callback:
                    progress_callback(len(results), total_files)
                
            except Exception as e:
                logger.error(f"Batch analysis failed: {e}")
                raise
        
        success_count = sum(1 for r in results if r.get('success', False))
        logger.info(f"Batch analysis complete: {success_count}/{total_files} successful")
        
        return results
    
    async def get_analysis_status(self, session_id: str) -> Dict[str, Any]:
        """Get status of analysis session."""
        async with get_async_db_session() as session:
            from sqlalchemy import select, func
            
            # Get session info
            session_stmt = select(AnalysisSession).where(AnalysisSession.id == session_id)
            session_result = await session.execute(session_stmt)
            analysis_session = session_result.scalar_one_or_none()
            
            if not analysis_session:
                raise ValueError(f"Analysis session not found: {session_id}")
            
            # Get result counts
            result_stmt = select(func.count(AnalysisResult.id)).where(
                AnalysisResult.session_id == session_id
            )
            result_count = await session.scalar(result_stmt)
            
            # Calculate average confidence
            confidence_stmt = select(func.avg(AnalysisResult.confidence_score)).where(
                AnalysisResult.session_id == session_id
            )
            avg_confidence = await session.scalar(confidence_stmt) or 0.0
            
            return {
                'session_id': session_id,
                'status': analysis_session.status,
                'started_at': analysis_session.started_at,
                'completed_at': analysis_session.completed_at,
                'model_name': analysis_session.model_name,
                'total_analyzed': result_count,
                'average_confidence': float(avg_confidence),
            }
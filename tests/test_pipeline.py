"""Tests for photo analysis pipeline."""

import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from photo_analyzer.pipeline.analyzer import PhotoAnalyzer
from photo_analyzer.pipeline.processor import PhotoProcessor
from photo_analyzer.pipeline.organizer import PhotoOrganizer


class TestPhotoAnalyzer:
    """Test photo analysis pipeline."""
    
    @pytest.fixture
    def mock_analyzer(self, test_config, mock_ollama_client):
        """Create PhotoAnalyzer with mocked dependencies."""
        with patch('photo_analyzer.pipeline.analyzer.OllamaClient', return_value=mock_ollama_client):
            analyzer = PhotoAnalyzer(test_config)
            return analyzer
    
    @pytest.mark.asyncio
    async def test_analyze_photo_success(self, mock_analyzer, temp_dir, test_db_session):
        """Test successful photo analysis."""
        # Create test image file
        test_image = temp_dir / "test.jpg"
        test_image.write_text("dummy image data")
        
        # Mock image validation and processing
        with patch.object(mock_analyzer.image_processor, 'validate_image') as mock_validate:
            mock_validate.return_value = {'is_valid': True, 'errors': []}
            
            with patch.object(mock_analyzer.image_processor, 'get_image_info') as mock_info:
                mock_info.return_value = {
                    'filename': 'test.jpg',
                    'width': 1920,
                    'height': 1080,
                    'file_size': 1024000,
                    'file_hash': 'abc123',
                    'format': 'JPEG',
                    'orientation': 'landscape'
                }
                
                with patch.object(mock_analyzer.exif_extractor, 'extract_exif') as mock_exif:
                    mock_exif.return_value = {
                        'date_taken': datetime(2023, 6, 15, 14, 30, 0),
                        'camera_make': 'Canon',
                        'camera_model': 'EOS R5'
                    }
                    
                    # Test analysis
                    result = await mock_analyzer.analyze_photo(test_image, test_db_session)
                    
                    assert result['success'] is True
                    assert 'photo_id' in result
                    assert 'description' in result
                    assert 'tags' in result
                    assert 'confidence' in result
    
    @pytest.mark.asyncio
    async def test_analyze_photo_invalid_image(self, mock_analyzer, temp_dir, test_db_session):
        """Test analysis of invalid image."""
        test_file = temp_dir / "invalid.txt"
        test_file.write_text("not an image")
        
        with patch.object(mock_analyzer.image_processor, 'validate_image') as mock_validate:
            mock_validate.return_value = {
                'is_valid': False, 
                'errors': ['Invalid image format']
            }
            
            with pytest.raises(ValueError, match="Invalid image"):
                await mock_analyzer.analyze_photo(test_file, test_db_session)
    
    @pytest.mark.asyncio
    async def test_analyze_batch(self, mock_analyzer, temp_dir):
        """Test batch photo analysis."""
        # Create test images
        test_files = []
        for i in range(3):
            test_file = temp_dir / f"test_{i}.jpg"
            test_file.write_text(f"dummy image data {i}")
            test_files.append(test_file)
        
        # Mock dependencies
        with patch.object(mock_analyzer.image_processor, 'validate_image') as mock_validate:
            mock_validate.return_value = {'is_valid': True, 'errors': []}
            
            with patch.object(mock_analyzer.image_processor, 'get_image_info') as mock_info:
                mock_info.return_value = {
                    'filename': 'test.jpg',
                    'width': 1920,
                    'height': 1080,
                    'file_size': 1024000,
                    'file_hash': 'abc123',
                    'format': 'JPEG',
                    'orientation': 'landscape'
                }
                
                with patch.object(mock_analyzer.exif_extractor, 'extract_exif') as mock_exif:
                    mock_exif.return_value = {}
                    
                    # Test batch analysis
                    results = await mock_analyzer.analyze_batch(test_files, batch_size=2)
                    
                    assert len(results) == 3
                    for result in results:
                        assert result.get('success') is True


class TestPhotoProcessor:
    """Test photo processing operations."""
    
    @pytest.fixture
    def processor(self, test_config):
        """Create PhotoProcessor instance."""
        return PhotoProcessor(test_config)
    
    def test_generate_smart_filename(self, processor):
        """Test intelligent filename generation."""
        # Test with description and tags
        filename = processor.generate_smart_filename(
            description="A beautiful mountain landscape with trees",
            tags=['landscape', 'mountain', 'nature'],
            date_taken=datetime(2023, 6, 15, 14, 30, 0)
        )
        
        assert '20230615' in filename
        assert any(word in filename for word in ['beautiful', 'mountain', 'landscape'])
        assert filename.replace('_', '').replace('.', '').isalnum()
    
    def test_generate_smart_filename_no_description(self, processor):
        """Test filename generation without description."""
        filename = processor.generate_smart_filename(
            description="",
            tags=['portrait', 'family'],
            date_taken=datetime(2023, 6, 15)
        )
        
        assert '20230615' in filename
        assert 'portrait' in filename or 'family' in filename
    
    def test_generate_smart_filename_minimal_data(self, processor):
        """Test filename generation with minimal data."""
        filename = processor.generate_smart_filename(
            description="",
            tags=[],
            date_taken=None
        )
        
        assert filename == 'photo'
    
    def test_sanitize_filename(self, processor):
        """Test filename sanitization."""
        test_cases = [
            ("photo<>test", "photo__test"),
            ("file name.jpg", "file_name.jpg"),
            ("multiple___underscores", "multiple_underscores"),
            ("", "photo"),
            ("normal_filename", "normal_filename")
        ]
        
        for input_name, expected in test_cases:
            result = processor._sanitize_filename(input_name)
            assert result == expected
    
    @pytest.mark.asyncio
    async def test_rename_photo_dry_run(self, processor, test_db_session, temp_dir, sample_photo_data):
        """Test photo rename in dry run mode."""
        # Create test photo file
        test_file = temp_dir / "original.jpg"
        test_file.write_text("dummy")
        
        # Mock photo in database
        with patch('photo_analyzer.pipeline.processor.select') as mock_select:
            mock_photo = MagicMock()
            mock_photo.id = "test-id"
            mock_photo.current_path = str(test_file)
            mock_photo.filename = "original.jpg"
            
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = mock_photo
            mock_select.return_value = mock_result
            
            result = await processor.rename_photo(
                "test-id", "new_name.jpg", test_db_session, dry_run=True
            )
            
            assert result['dry_run'] is True
            assert result['success'] is True
            assert 'new_name.jpg' in result['new_path']


class TestPhotoOrganizer:
    """Test photo organization system."""
    
    @pytest.fixture
    def organizer(self, test_config):
        """Create PhotoOrganizer instance."""
        return PhotoOrganizer(test_config)
    
    def test_get_default_organization_rules(self, organizer):
        """Test default organization rules."""
        rules = organizer._get_default_organization_rules()
        
        assert rules['date_format'] == 'YYYY/MM/DD'
        assert rules['filename_strategy'] == 'smart'
        assert rules['create_symlinks'] is True
        assert 'tags' in rules['symlink_categories']
    
    def test_create_date_path(self, organizer, temp_dir):
        """Test date-based directory path creation."""
        base_dir = temp_dir / "photos"
        test_date = datetime(2023, 6, 15, 14, 30, 0)
        
        # Test YYYY/MM/DD format
        rules = {'date_format': 'YYYY/MM/DD'}
        path = organizer._create_date_path(base_dir, test_date, rules)
        expected = base_dir / "2023" / "06" / "15"
        assert path == expected
        
        # Test YYYY/MM format
        rules = {'date_format': 'YYYY/MM'}
        path = organizer._create_date_path(base_dir, test_date, rules)
        expected = base_dir / "2023" / "06"
        assert path == expected
        
        # Test YYYY format
        rules = {'date_format': 'YYYY'}
        path = organizer._create_date_path(base_dir, test_date, rules)
        expected = base_dir / "2023"
        assert path == expected
    
    def test_resolve_filename_conflict(self, organizer, temp_dir):
        """Test filename conflict resolution."""
        # Create existing file
        existing_file = temp_dir / "photo.jpg"
        existing_file.write_text("existing")
        
        # Test conflict resolution
        new_path = organizer._resolve_filename_conflict(existing_file)
        
        assert new_path != existing_file
        assert new_path.stem == "photo_1"
        assert new_path.suffix == ".jpg"
    
    @pytest.mark.asyncio
    async def test_organize_photo_dry_run(self, organizer, test_db_session, temp_dir):
        """Test photo organization in dry run mode."""
        # Create test photo file
        test_file = temp_dir / "test.jpg"
        test_file.write_text("dummy")
        
        output_dir = temp_dir / "organized"
        
        # Mock photo in database
        with patch('photo_analyzer.pipeline.organizer.select') as mock_select:
            mock_photo = MagicMock()
            mock_photo.id = "test-id"
            mock_photo.current_path = str(test_file)
            mock_photo.filename = "test.jpg"
            mock_photo.date_taken = datetime(2023, 6, 15, 14, 30, 0)
            mock_photo.tags = []
            
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = mock_photo
            mock_select.return_value = mock_result
            
            result = await organizer.organize_photo(
                "test-id", output_dir, dry_run=True, session=test_db_session
            )
            
            assert result['dry_run'] is True
            assert result['success'] is True
            assert 'target_path' in result
            assert '2023/06/15' in result['target_path']
    
    @pytest.mark.asyncio
    async def test_plan_symbolic_links(self, organizer, temp_dir):
        """Test symbolic link planning."""
        # Mock photo with tags
        mock_photo = MagicMock()
        mock_photo.tags = [MagicMock(name='landscape'), MagicMock(name='nature')]
        mock_photo.date_taken = datetime(2023, 6, 15)
        mock_photo.exif_data = {
            'camera_make': 'Canon',
            'camera_model': 'EOS R5'
        }
        
        target_path = temp_dir / "2023/06/15/photo.jpg"
        base_dir = temp_dir
        rules = {
            'create_symlinks': True,
            'symlink_categories': ['tags', 'camera', 'year']
        }
        
        symlink_dirs = await organizer._plan_symbolic_links(
            mock_photo, target_path, base_dir, rules
        )
        
        # Should have symlinks for tags, camera, and year
        assert len(symlink_dirs) >= 3
        
        # Check that expected directories are in the plan
        dir_names = [str(link_dir) for link_dir, _ in symlink_dirs]
        assert any('by_tags/landscape' in name for name in dir_names)
        assert any('by_tags/nature' in name for name in dir_names)
        assert any('by_camera/Canon_EOS R5' in name for name in dir_names)
        assert any('by_year/2023' in name for name in dir_names)


@pytest.mark.integration
class TestPipelineIntegration:
    """Integration tests for the complete pipeline."""
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self, test_config, temp_dir):
        """Test complete analyze -> organize workflow."""
        # Create test image
        test_image = temp_dir / "vacation_photo.jpg"
        test_image.write_text("dummy image data")
        
        output_dir = temp_dir / "organized_photos"
        
        # Mock all external dependencies
        with patch('photo_analyzer.pipeline.analyzer.OllamaClient') as mock_llm:
            mock_llm_instance = AsyncMock()
            mock_llm_instance.health_check.return_value = True
            mock_llm_instance.analyze_image.return_value = {
                'description': 'A vacation photo at the beach',
                'tags': ['vacation', 'beach', 'travel'],
                'confidence': 0.95
            }
            mock_llm_instance.generate_filename.return_value = {
                'filename': 'vacation_beach_travel.jpg'
            }
            mock_llm.return_value = mock_llm_instance
            
            with patch('photo_analyzer.utils.image.ImageProcessor') as mock_img_proc:
                mock_img_proc.return_value.validate_image.return_value = {
                    'is_valid': True, 'errors': []
                }
                mock_img_proc.return_value.get_image_info.return_value = {
                    'filename': 'vacation_photo.jpg',
                    'width': 1920,
                    'height': 1080,
                    'file_size': 2048000,
                    'file_hash': 'test-hash-123',
                    'format': 'JPEG',
                    'orientation': 'landscape'
                }
                
                with patch('photo_analyzer.utils.exif.ExifExtractor') as mock_exif:
                    mock_exif.return_value.extract_exif.return_value = {
                        'date_taken': datetime(2023, 7, 15, 16, 30, 0),
                        'camera_make': 'Sony',
                        'camera_model': 'A7R IV'
                    }
                    
                    # This would be a full integration test with real database
                    # For now, just verify that the components can be instantiated
                    analyzer = PhotoAnalyzer(test_config)
                    processor = PhotoProcessor(test_config)
                    organizer = PhotoOrganizer(test_config)
                    
                    assert analyzer is not None
                    assert processor is not None
                    assert organizer is not None
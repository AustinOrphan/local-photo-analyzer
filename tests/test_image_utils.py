"""Tests for image processing utilities."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image

from photo_analyzer.utils.image import ImageProcessor


class TestImageProcessor:
    """Test image processing utilities."""
    
    def test_supported_formats(self):
        """Test supported image format detection."""
        processor = ImageProcessor()
        
        # Test supported formats
        assert processor.is_supported_image("photo.jpg")
        assert processor.is_supported_image("image.png")
        assert processor.is_supported_image("pic.jpeg")
        assert processor.is_supported_image("shot.tiff")
        assert processor.is_supported_image("raw.nef")
        
        # Test unsupported formats
        assert not processor.is_supported_image("document.pdf")
        assert not processor.is_supported_image("video.mp4")
        assert not processor.is_supported_image("text.txt")
    
    @patch('photo_analyzer.utils.image.Image.open')
    def test_get_image_info(self, mock_open, temp_dir):
        """Test image information extraction."""
        # Create test file
        test_file = temp_dir / "test.jpg"
        test_file.write_text("dummy")
        
        # Mock PIL Image
        mock_img = MagicMock()
        mock_img.format = "JPEG"
        mock_img.mode = "RGB"
        mock_img.size = (1920, 1080)
        mock_img.info = {}
        mock_img.getdata.return_value = [0] * (1920 * 1080)
        
        mock_open.return_value.__enter__.return_value = mock_img
        
        processor = ImageProcessor()
        info = processor.get_image_info(test_file)
        
        assert info['filename'] == "test.jpg"
        assert info['format'] == "JPEG"
        assert info['mode'] == "RGB"
        assert info['width'] == 1920
        assert info['height'] == 1080
        assert info['aspect_ratio'] == pytest.approx(1.777, rel=1e-3)
        assert info['orientation'] == 'landscape'
        assert 'file_hash' in info
    
    def test_get_image_info_file_not_found(self):
        """Test image info for non-existent file."""
        processor = ImageProcessor()
        
        with pytest.raises(FileNotFoundError):
            processor.get_image_info("nonexistent.jpg")
    
    def test_get_image_info_unsupported_format(self, temp_dir):
        """Test image info for unsupported format."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("not an image")
        
        processor = ImageProcessor()
        
        with pytest.raises(ValueError, match="Unsupported image format"):
            processor.get_image_info(test_file)
    
    def test_calculate_image_hash(self, temp_dir):
        """Test image hash calculation."""
        test_file = temp_dir / "test.jpg"
        test_content = b"test image content"
        test_file.write_bytes(test_content)
        
        processor = ImageProcessor()
        hash1 = processor.calculate_image_hash(test_file)
        hash2 = processor.calculate_image_hash(test_file)
        
        # Same file should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex digest length
        
        # Different content should produce different hash
        test_file2 = temp_dir / "test2.jpg"
        test_file2.write_bytes(b"different content")
        hash3 = processor.calculate_image_hash(test_file2)
        
        assert hash1 != hash3
    
    @patch('photo_analyzer.utils.image.Image.open')
    def test_create_thumbnail(self, mock_open, temp_dir):
        """Test thumbnail creation."""
        # Setup
        input_file = temp_dir / "input.jpg"
        input_file.write_text("dummy")
        output_file = temp_dir / "thumb.jpg"
        
        mock_img = MagicMock()
        mock_img.mode = "RGB"
        mock_img.size = (200, 200)
        
        mock_open.return_value.__enter__.return_value = mock_img
        
        processor = ImageProcessor()
        result = processor.create_thumbnail(input_file, output_file)
        
        assert result is True
        mock_img.thumbnail.assert_called_once()
        mock_img.save.assert_called_once()
    
    @patch('photo_analyzer.utils.image.Image.open')
    def test_analyze_brightness(self, mock_open, temp_dir):
        """Test brightness analysis."""
        test_file = temp_dir / "test.jpg"
        test_file.write_text("dummy")
        
        # Mock a bright image
        mock_img = MagicMock()
        mock_gray = MagicMock()
        mock_gray.histogram.return_value = [0] * 200 + [1000] * 55  # Bright image
        mock_img.convert.return_value = mock_gray
        
        mock_open.return_value.__enter__.return_value = mock_img
        
        processor = ImageProcessor()
        analysis = processor.analyze_brightness(test_file)
        
        assert 'brightness' in analysis
        assert 'contrast' in analysis
        assert 'is_bright' in analysis
        assert 'is_dark' in analysis
        assert 0 <= analysis['brightness'] <= 1
    
    @patch('photo_analyzer.utils.image.Image.open')
    def test_extract_dominant_colors(self, mock_open, temp_dir):
        """Test dominant color extraction."""
        test_file = temp_dir / "test.jpg"
        test_file.write_text("dummy")
        
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        mock_img.thumbnail.return_value = None
        mock_img.getcolors.return_value = [
            (1000, (255, 0, 0)),    # Red
            (800, (0, 255, 0)),     # Green
            (600, (0, 0, 255)),     # Blue
        ]
        
        mock_open.return_value.__enter__.return_value = mock_img
        
        processor = ImageProcessor()
        colors = processor.extract_dominant_colors(test_file, num_colors=3)
        
        assert len(colors) == 3
        assert colors[0] == (255, 0, 0)  # Most frequent color first
        assert colors[1] == (0, 255, 0)
        assert colors[2] == (0, 0, 255)
    
    @patch('photo_analyzer.utils.image.Image.open')
    def test_validate_image_valid(self, mock_open, temp_dir):
        """Test image validation for valid image."""
        test_file = temp_dir / "test.jpg"
        test_file.write_text("dummy")
        
        mock_img = MagicMock()
        mock_img.format = "JPEG"
        mock_img.size = (1920, 1080)
        mock_img.load.return_value = None  # No exception
        
        mock_open.return_value.__enter__.return_value = mock_img
        
        processor = ImageProcessor()
        validation = processor.validate_image(test_file)
        
        assert validation['is_valid'] is True
        assert validation['exists'] is True
        assert validation['is_file'] is True
        assert validation['is_supported_format'] is True
        assert validation['is_readable'] is True
        assert validation['corruption_detected'] is False
        assert len(validation['errors']) == 0
    
    def test_validate_image_not_found(self):
        """Test image validation for non-existent file."""
        processor = ImageProcessor()
        validation = processor.validate_image("nonexistent.jpg")
        
        assert validation['is_valid'] is False
        assert validation['exists'] is False
        assert 'File does not exist' in validation['errors']
    
    def test_validate_image_unsupported_format(self, temp_dir):
        """Test image validation for unsupported format."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("not an image")
        
        processor = ImageProcessor()
        validation = processor.validate_image(test_file)
        
        assert validation['is_valid'] is False
        assert validation['is_supported_format'] is False
        assert any('Unsupported format' in error for error in validation['errors'])
    
    @patch('photo_analyzer.utils.image.Image.open')
    def test_validate_image_corrupted(self, mock_open, temp_dir):
        """Test image validation for corrupted image."""
        test_file = temp_dir / "test.jpg"
        test_file.write_text("dummy")
        
        # Mock corruption - Image.open raises exception
        mock_open.side_effect = Exception("Corrupted image")
        
        processor = ImageProcessor()
        validation = processor.validate_image(test_file)
        
        assert validation['is_valid'] is False
        assert validation['corruption_detected'] is True
        assert any('Cannot read image' in error for error in validation['errors'])
    
    @patch('photo_analyzer.utils.image.Image.open')
    def test_convert_format(self, mock_open, temp_dir):
        """Test image format conversion."""
        input_file = temp_dir / "input.png"
        input_file.write_text("dummy")
        output_file = temp_dir / "output.jpg"
        
        mock_img = MagicMock()
        mock_img.mode = "RGBA"
        mock_img.size = (100, 100)
        
        mock_open.return_value.__enter__.return_value = mock_img
        
        processor = ImageProcessor()
        result = processor.convert_format(input_file, output_file, 'JPEG', quality=90)
        
        assert result is True
        mock_img.save.assert_called_once()
        
        # Check that RGBA was converted for JPEG
        save_args = mock_img.save.call_args
        assert 'JPEG' in save_args[0]
    
    @patch('photo_analyzer.utils.image.Image.open')
    def test_resize_image(self, mock_open, temp_dir):
        """Test image resizing."""
        input_file = temp_dir / "input.jpg"
        input_file.write_text("dummy")
        output_file = temp_dir / "resized.jpg"
        
        mock_img = MagicMock()
        mock_img.format = "JPEG"
        mock_img.size = (3000, 2000)
        
        mock_open.return_value.__enter__.return_value = mock_img
        
        processor = ImageProcessor()
        result = processor.resize_image(
            input_file, output_file, 
            max_size=(1920, 1080), 
            preserve_aspect_ratio=True
        )
        
        assert result is True
        mock_img.thumbnail.assert_called_once_with((1920, 1080), mock_open.return_value.__enter__.return_value.Resampling.LANCZOS)
        mock_img.save.assert_called_once()


@pytest.mark.integration
class TestImageProcessorIntegration:
    """Integration tests using real images."""
    
    def test_create_sample_image(self, temp_dir):
        """Test with actual PIL image creation."""
        # Create a real test image
        test_image = Image.new('RGB', (100, 100), color='red')
        test_file = temp_dir / "test.jpg"
        test_image.save(test_file, 'JPEG')
        
        processor = ImageProcessor()
        
        # Test basic info extraction
        info = processor.get_image_info(test_file)
        assert info['width'] == 100
        assert info['height'] == 100
        assert info['format'] == 'JPEG'
        
        # Test validation
        validation = processor.validate_image(test_file)
        assert validation['is_valid'] is True
        
        # Test thumbnail creation
        thumb_file = temp_dir / "thumb.jpg"
        result = processor.create_thumbnail(test_file, thumb_file)
        assert result is True
        assert thumb_file.exists()
        
        # Verify thumbnail is smaller
        thumb_info = processor.get_image_info(thumb_file)
        assert thumb_info['width'] <= 200
        assert thumb_info['height'] <= 200
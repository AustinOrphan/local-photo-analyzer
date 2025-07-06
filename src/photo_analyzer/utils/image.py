"""Image processing utilities."""

import hashlib
import io
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union

from PIL import Image, ImageOps, ExifTags
from PIL.ExifTags import TAGS

from ..core.logger import get_logger

logger = get_logger(__name__)


class ImageProcessor:
    """Image processing and analysis utilities."""
    
    SUPPORTED_FORMATS = {
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif',
        '.webp', '.heic', '.heif', '.raw', '.cr2', '.nef', '.arw'
    }
    
    def __init__(self):
        self.logger = logger
    
    def is_supported_image(self, file_path: Union[str, Path]) -> bool:
        """Check if file is a supported image format."""
        path = Path(file_path)
        return path.suffix.lower() in self.SUPPORTED_FORMATS
    
    def get_image_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Get basic image information."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        
        if not self.is_supported_image(path):
            raise ValueError(f"Unsupported image format: {path.suffix}")
        
        try:
            with Image.open(path) as img:
                # Basic image info
                info = {
                    'filename': path.name,
                    'format': img.format,
                    'mode': img.mode,
                    'size': img.size,
                    'width': img.size[0],
                    'height': img.size[1],
                    'has_transparency': img.mode in ('RGBA', 'LA') or 'transparency' in img.info,
                    'file_size': path.stat().st_size,
                    'file_path': str(path),
                }
                
                # Calculate aspect ratio
                if img.size[1] > 0:
                    info['aspect_ratio'] = img.size[0] / img.size[1]
                else:
                    info['aspect_ratio'] = 0
                
                # Orientation info
                info['orientation'] = 'landscape' if info['aspect_ratio'] > 1 else 'portrait'
                if abs(info['aspect_ratio'] - 1) < 0.1:
                    info['orientation'] = 'square'
                
                # Color analysis
                info['is_grayscale'] = img.mode in ('L', 'LA', '1')
                
                # File hash for deduplication
                info['file_hash'] = self.calculate_image_hash(path)
                
                return info
                
        except Exception as e:
            logger.error(f"Failed to get image info for {path}: {e}")
            raise
    
    def calculate_image_hash(self, file_path: Union[str, Path]) -> str:
        """Calculate SHA256 hash of image file."""
        path = Path(file_path)
        hash_sha256 = hashlib.sha256()
        
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    def create_thumbnail(
        self, 
        file_path: Union[str, Path], 
        output_path: Union[str, Path],
        size: Tuple[int, int] = (200, 200),
        quality: int = 85
    ) -> bool:
        """Create a thumbnail of the image."""
        try:
            input_path = Path(file_path)
            output_path = Path(output_path)
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with Image.open(input_path) as img:
                # Apply EXIF orientation
                img = ImageOps.exif_transpose(img)
                
                # Create thumbnail
                img.thumbnail(size, Image.Resampling.LANCZOS)
                
                # Convert to RGB if necessary (for JPEG)
                if img.mode in ('RGBA', 'LA', 'P'):
                    # Create white background
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                    img = background
                
                # Save thumbnail
                img.save(output_path, 'JPEG', quality=quality, optimize=True)
                
                logger.debug(f"Created thumbnail: {output_path}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to create thumbnail for {file_path}: {e}")
            return False
    
    def resize_image(
        self,
        file_path: Union[str, Path],
        output_path: Union[str, Path],
        max_size: Tuple[int, int] = (1920, 1080),
        quality: int = 90,
        preserve_aspect_ratio: bool = True
    ) -> bool:
        """Resize image while preserving quality."""
        try:
            input_path = Path(file_path)
            output_path = Path(output_path)
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with Image.open(input_path) as img:
                # Apply EXIF orientation
                img = ImageOps.exif_transpose(img)
                
                # Calculate new size
                if preserve_aspect_ratio:
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)
                else:
                    img = img.resize(max_size, Image.Resampling.LANCZOS)
                
                # Preserve original format if possible
                format_to_save = img.format or 'JPEG'
                
                # Save resized image
                save_kwargs = {'optimize': True}
                if format_to_save in ('JPEG', 'WEBP'):
                    save_kwargs['quality'] = quality
                
                img.save(output_path, format_to_save, **save_kwargs)
                
                logger.debug(f"Resized image: {output_path}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to resize image {file_path}: {e}")
            return False
    
    def extract_dominant_colors(
        self, 
        file_path: Union[str, Path], 
        num_colors: int = 5
    ) -> List[Tuple[int, int, int]]:
        """Extract dominant colors from image."""
        try:
            with Image.open(file_path) as img:
                # Convert to RGB
                img = img.convert('RGB')
                
                # Resize for faster processing
                img.thumbnail((150, 150), Image.Resampling.LANCZOS)
                
                # Get colors
                colors = img.getcolors(maxcolors=256 * 256 * 256)
                if not colors:
                    return []
                
                # Sort by frequency and get top colors
                colors.sort(key=lambda x: x[0], reverse=True)
                dominant_colors = [color[1] for color in colors[:num_colors]]
                
                return dominant_colors
                
        except Exception as e:
            logger.error(f"Failed to extract colors from {file_path}: {e}")
            return []
    
    def detect_faces(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Detect faces in image (placeholder for future implementation)."""
        # This would require additional dependencies like opencv-python or face_recognition
        # For now, return empty list
        logger.info(f"Face detection not implemented yet for {file_path}")
        return []
    
    def analyze_brightness(self, file_path: Union[str, Path]) -> Dict[str, float]:
        """Analyze image brightness and contrast."""
        try:
            with Image.open(file_path) as img:
                # Convert to grayscale for analysis
                gray = img.convert('L')
                
                # Calculate histogram
                histogram = gray.histogram()
                
                # Calculate average brightness (0-255)
                total_pixels = sum(histogram)
                brightness = sum(i * count for i, count in enumerate(histogram)) / total_pixels
                
                # Calculate contrast (standard deviation)
                variance = sum((i - brightness) ** 2 * count for i, count in enumerate(histogram)) / total_pixels
                contrast = variance ** 0.5
                
                # Normalize values
                brightness_normalized = brightness / 255.0
                contrast_normalized = contrast / 127.5  # Max possible std dev
                
                return {
                    'brightness': brightness_normalized,
                    'contrast': contrast_normalized,
                    'brightness_raw': brightness,
                    'contrast_raw': contrast,
                    'is_dark': brightness_normalized < 0.3,
                    'is_bright': brightness_normalized > 0.7,
                    'is_low_contrast': contrast_normalized < 0.2,
                    'is_high_contrast': contrast_normalized > 0.8,
                }
                
        except Exception as e:
            logger.error(f"Failed to analyze brightness for {file_path}: {e}")
            return {}
    
    def convert_format(
        self,
        file_path: Union[str, Path],
        output_path: Union[str, Path],
        target_format: str = 'JPEG',
        quality: int = 90
    ) -> bool:
        """Convert image to different format."""
        try:
            input_path = Path(file_path)
            output_path = Path(output_path)
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with Image.open(input_path) as img:
                # Apply EXIF orientation
                img = ImageOps.exif_transpose(img)
                
                # Convert mode if necessary
                if target_format == 'JPEG' and img.mode in ('RGBA', 'LA', 'P'):
                    # Create white background for JPEG
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                    img = background
                
                # Save in target format
                save_kwargs = {'optimize': True}
                if target_format in ('JPEG', 'WEBP'):
                    save_kwargs['quality'] = quality
                
                img.save(output_path, target_format, **save_kwargs)
                
                logger.debug(f"Converted {input_path} to {target_format}: {output_path}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to convert {file_path} to {target_format}: {e}")
            return False
    
    def validate_image(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Validate image file and return detailed status."""
        path = Path(file_path)
        validation = {
            'is_valid': False,
            'exists': path.exists(),
            'is_file': path.is_file() if path.exists() else False,
            'is_supported_format': False,
            'is_readable': False,
            'corruption_detected': False,
            'errors': [],
            'warnings': [],
        }
        
        if not validation['exists']:
            validation['errors'].append('File does not exist')
            return validation
        
        if not validation['is_file']:
            validation['errors'].append('Path is not a file')
            return validation
        
        validation['is_supported_format'] = self.is_supported_image(path)
        if not validation['is_supported_format']:
            validation['errors'].append(f'Unsupported format: {path.suffix}')
            return validation
        
        # Try to open and verify the image
        try:
            with Image.open(path) as img:
                # Try to load the image data
                img.load()
                validation['is_readable'] = True
                
                # Basic validation checks
                if img.size[0] <= 0 or img.size[1] <= 0:
                    validation['errors'].append('Invalid image dimensions')
                
                # Check for common corruption indicators
                if img.format is None:
                    validation['warnings'].append('Unable to detect image format')
                
        except Exception as e:
            validation['errors'].append(f'Cannot read image: {str(e)}')
            validation['corruption_detected'] = True
        
        validation['is_valid'] = (
            validation['exists'] and 
            validation['is_file'] and 
            validation['is_supported_format'] and 
            validation['is_readable'] and 
            not validation['corruption_detected'] and
            not validation['errors']
        )
        
        return validation
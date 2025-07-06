"""EXIF data extraction utilities."""

import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Union, Tuple

from PIL import Image, ExifTags
from PIL.ExifTags import TAGS, GPSTAGS

from ..core.logger import get_logger

logger = get_logger(__name__)


class ExifExtractor:
    """Extract and process EXIF data from images."""
    
    def __init__(self):
        self.logger = logger
    
    def extract_exif(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Extract all EXIF data from image."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        
        try:
            with Image.open(path) as img:
                # Get raw EXIF data
                exifdata = img.getexif()
                
                if not exifdata:
                    logger.debug(f"No EXIF data found in {path}")
                    return {}
                
                # Convert to human-readable format
                exif_dict = {}
                for tag_id, value in exifdata.items():
                    tag = TAGS.get(tag_id, tag_id)
                    exif_dict[tag] = value
                
                # Process special fields
                processed_exif = self._process_exif_data(exif_dict)
                
                # Extract GPS data
                gps_data = self._extract_gps_data(exifdata)
                if gps_data:
                    processed_exif['GPS'] = gps_data
                
                return processed_exif
                
        except Exception as e:
            logger.error(f"Failed to extract EXIF data from {path}: {e}")
            return {}
    
    def _process_exif_data(self, exif_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Process and clean EXIF data."""
        processed = {}
        
        # Copy basic fields
        for key, value in exif_dict.items():
            if isinstance(value, (str, int, float)):
                processed[key] = value
            elif isinstance(value, bytes):
                try:
                    processed[key] = value.decode('utf-8', errors='ignore')
                except:
                    processed[key] = str(value)
            else:
                processed[key] = str(value)
        
        # Extract camera information
        camera_info = self._extract_camera_info(exif_dict)
        processed.update(camera_info)
        
        # Extract photo settings
        photo_settings = self._extract_photo_settings(exif_dict)
        processed.update(photo_settings)
        
        # Extract timestamps
        timestamps = self._extract_timestamps(exif_dict)
        processed.update(timestamps)
        
        return processed
    
    def _extract_camera_info(self, exif_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Extract camera and lens information."""
        camera_info = {}
        
        # Camera make and model
        if 'Make' in exif_dict:
            camera_info['camera_make'] = str(exif_dict['Make']).strip()
        
        if 'Model' in exif_dict:
            camera_info['camera_model'] = str(exif_dict['Model']).strip()
        
        # Lens information
        if 'LensModel' in exif_dict:
            camera_info['lens_model'] = str(exif_dict['LensModel']).strip()
        
        if 'LensMake' in exif_dict:
            camera_info['lens_make'] = str(exif_dict['LensMake']).strip()
        
        # Software
        if 'Software' in exif_dict:
            camera_info['software'] = str(exif_dict['Software']).strip()
        
        return camera_info
    
    def _extract_photo_settings(self, exif_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Extract photo capture settings."""
        settings = {}
        
        # ISO
        if 'ISOSpeedRatings' in exif_dict:
            settings['iso'] = exif_dict['ISOSpeedRatings']
        elif 'PhotographicSensitivity' in exif_dict:
            settings['iso'] = exif_dict['PhotographicSensitivity']
        
        # Aperture
        if 'FNumber' in exif_dict:
            f_number = exif_dict['FNumber']
            if hasattr(f_number, 'numerator') and hasattr(f_number, 'denominator'):
                settings['aperture'] = f_number.numerator / f_number.denominator
            else:
                settings['aperture'] = float(f_number)
        
        # Shutter speed
        if 'ExposureTime' in exif_dict:
            exposure = exif_dict['ExposureTime']
            if hasattr(exposure, 'numerator') and hasattr(exposure, 'denominator'):
                settings['shutter_speed'] = exposure.numerator / exposure.denominator
                settings['shutter_speed_text'] = f"{exposure.numerator}/{exposure.denominator}s"
            else:
                settings['shutter_speed'] = float(exposure)
                settings['shutter_speed_text'] = f"{exposure}s"
        
        # Focal length
        if 'FocalLength' in exif_dict:
            focal = exif_dict['FocalLength']
            if hasattr(focal, 'numerator') and hasattr(focal, 'denominator'):
                settings['focal_length'] = focal.numerator / focal.denominator
            else:
                settings['focal_length'] = float(focal)
        
        # Flash
        if 'Flash' in exif_dict:
            flash_value = exif_dict['Flash']
            settings['flash_fired'] = bool(flash_value & 0x01)
            settings['flash_mode'] = flash_value
        
        # Exposure mode
        if 'ExposureMode' in exif_dict:
            settings['exposure_mode'] = exif_dict['ExposureMode']
        
        # White balance
        if 'WhiteBalance' in exif_dict:
            settings['white_balance'] = exif_dict['WhiteBalance']
        
        # Metering mode
        if 'MeteringMode' in exif_dict:
            settings['metering_mode'] = exif_dict['MeteringMode']
        
        return settings
    
    def _extract_timestamps(self, exif_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Extract timestamp information."""
        timestamps = {}
        
        # Date taken
        for date_field in ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized']:
            if date_field in exif_dict:
                try:
                    date_str = str(exif_dict[date_field])
                    # Parse EXIF date format: "YYYY:MM:DD HH:MM:SS"
                    dt = datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
                    timestamps[f'{date_field.lower()}'] = dt
                    
                    # Use DateTimeOriginal as primary if available
                    if date_field == 'DateTimeOriginal':
                        timestamps['date_taken'] = dt
                except ValueError as e:
                    logger.warning(f"Failed to parse {date_field} '{date_str}': {e}")
        
        # Use any available date as fallback
        if 'date_taken' not in timestamps:
            for key, value in timestamps.items():
                if isinstance(value, datetime):
                    timestamps['date_taken'] = value
                    break
        
        return timestamps
    
    def _extract_gps_data(self, exifdata) -> Optional[Dict[str, Any]]:
        """Extract GPS coordinates and related data."""
        try:
            gps_info = exifdata.get_ifd(0x8825)  # GPS IFD
            if not gps_info:
                return None
            
            gps_data = {}
            
            # Convert GPS tags to human-readable format
            for tag_id, value in gps_info.items():
                tag = GPSTAGS.get(tag_id, tag_id)
                gps_data[tag] = value
            
            # Extract coordinates
            coords = self._parse_gps_coordinates(gps_data)
            if coords:
                gps_data.update(coords)
            
            return gps_data
            
        except Exception as e:
            logger.debug(f"No GPS data or failed to extract GPS: {e}")
            return None
    
    def _parse_gps_coordinates(self, gps_data: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Parse GPS coordinates from GPS data."""
        try:
            # Get latitude
            if 'GPSLatitude' in gps_data and 'GPSLatitudeRef' in gps_data:
                lat_dms = gps_data['GPSLatitude']
                lat_ref = gps_data['GPSLatitudeRef']
                latitude = self._dms_to_decimal(lat_dms, lat_ref)
            else:
                return None
            
            # Get longitude
            if 'GPSLongitude' in gps_data and 'GPSLongitudeRef' in gps_data:
                lon_dms = gps_data['GPSLongitude']
                lon_ref = gps_data['GPSLongitudeRef']
                longitude = self._dms_to_decimal(lon_dms, lon_ref)
            else:
                return None
            
            coords = {
                'latitude': latitude,
                'longitude': longitude,
            }
            
            # Get altitude if available
            if 'GPSAltitude' in gps_data:
                altitude = gps_data['GPSAltitude']
                if hasattr(altitude, 'numerator') and hasattr(altitude, 'denominator'):
                    altitude_meters = altitude.numerator / altitude.denominator
                else:
                    altitude_meters = float(altitude)
                
                # Check altitude reference (0 = above sea level, 1 = below)
                if gps_data.get('GPSAltitudeRef', 0) == 1:
                    altitude_meters = -altitude_meters
                
                coords['altitude'] = altitude_meters
            
            return coords
            
        except Exception as e:
            logger.warning(f"Failed to parse GPS coordinates: {e}")
            return None
    
    def _dms_to_decimal(self, dms_tuple, ref: str) -> float:
        """Convert degrees, minutes, seconds to decimal degrees."""
        degrees, minutes, seconds = dms_tuple
        
        # Convert fractions to float
        if hasattr(degrees, 'numerator'):
            degrees = degrees.numerator / degrees.denominator
        if hasattr(minutes, 'numerator'):
            minutes = minutes.numerator / minutes.denominator
        if hasattr(seconds, 'numerator'):
            seconds = seconds.numerator / seconds.denominator
        
        decimal = float(degrees) + float(minutes) / 60 + float(seconds) / 3600
        
        # Apply hemisphere reference
        if ref in ['S', 'W']:
            decimal = -decimal
        
        return decimal
    
    def get_date_taken(self, file_path: Union[str, Path]) -> Optional[datetime]:
        """Get the date photo was taken from EXIF data."""
        exif_data = self.extract_exif(file_path)
        return exif_data.get('date_taken')
    
    def get_camera_info(self, file_path: Union[str, Path]) -> Dict[str, str]:
        """Get camera make and model from EXIF data."""
        exif_data = self.extract_exif(file_path)
        
        camera_info = {}
        if 'camera_make' in exif_data:
            camera_info['make'] = exif_data['camera_make']
        if 'camera_model' in exif_data:
            camera_info['model'] = exif_data['camera_model']
        
        return camera_info
    
    def get_gps_coordinates(self, file_path: Union[str, Path]) -> Optional[Tuple[float, float]]:
        """Get GPS coordinates as (latitude, longitude) tuple."""
        exif_data = self.extract_exif(file_path)
        
        gps_data = exif_data.get('GPS', {})
        if 'latitude' in gps_data and 'longitude' in gps_data:
            return (gps_data['latitude'], gps_data['longitude'])
        
        return None
    
    def has_exif(self, file_path: Union[str, Path]) -> bool:
        """Check if image has EXIF data."""
        try:
            with Image.open(file_path) as img:
                exifdata = img.getexif()
                return bool(exifdata)
        except Exception:
            return False
    
    def remove_exif(self, file_path: Union[str, Path], output_path: Union[str, Path]) -> bool:
        """Remove EXIF data from image and save to output path."""
        try:
            input_path = Path(file_path)
            output_path = Path(output_path)
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with Image.open(input_path) as img:
                # Create image without EXIF
                clean_img = Image.new(img.mode, img.size)
                clean_img.putdata(list(img.getdata()))
                
                # Save without EXIF
                clean_img.save(output_path, img.format, quality=95, optimize=True)
                
                logger.info(f"Removed EXIF data: {output_path}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to remove EXIF from {file_path}: {e}")
            return False
    
    def extract_summary(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Extract key EXIF information as a summary."""
        exif_data = self.extract_exif(file_path)
        
        summary = {
            'has_exif': bool(exif_data),
            'date_taken': exif_data.get('date_taken'),
            'camera': None,
            'lens': None,
            'settings': {},
            'location': None,
        }
        
        # Camera info
        camera_parts = []
        if 'camera_make' in exif_data:
            camera_parts.append(exif_data['camera_make'])
        if 'camera_model' in exif_data:
            camera_parts.append(exif_data['camera_model'])
        if camera_parts:
            summary['camera'] = ' '.join(camera_parts)
        
        # Lens info
        if 'lens_model' in exif_data:
            summary['lens'] = exif_data['lens_model']
        
        # Key settings
        for setting in ['iso', 'aperture', 'shutter_speed', 'focal_length']:
            if setting in exif_data:
                summary['settings'][setting] = exif_data[setting]
        
        # Location
        gps_data = exif_data.get('GPS', {})
        if 'latitude' in gps_data and 'longitude' in gps_data:
            summary['location'] = {
                'latitude': gps_data['latitude'],
                'longitude': gps_data['longitude'],
            }
            if 'altitude' in gps_data:
                summary['location']['altitude'] = gps_data['altitude']
        
        return summary
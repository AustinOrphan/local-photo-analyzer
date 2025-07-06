"""Date parsing and manipulation utilities."""

import re
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Union, List, Tuple
import calendar

from ..core.logger import get_logger

logger = get_logger(__name__)


class DateUtils:
    """Date parsing and manipulation utilities."""
    
    # Common date patterns in filenames
    DATE_PATTERNS = [
        # YYYY-MM-DD formats
        (r'(\d{4})-(\d{2})-(\d{2})', '%Y-%m-%d'),
        (r'(\d{4})_(\d{2})_(\d{2})', '%Y_%m_%d'),
        (r'(\d{4})(\d{2})(\d{2})', '%Y%m%d'),
        
        # DD-MM-YYYY formats
        (r'(\d{2})-(\d{2})-(\d{4})', '%d-%m-%Y'),
        (r'(\d{2})_(\d{2})_(\d{4})', '%d_%m_%Y'),
        (r'(\d{2})(\d{2})(\d{4})', '%d%m%Y'),
        
        # MM-DD-YYYY formats
        (r'(\d{2})-(\d{2})-(\d{4})', '%m-%d-%Y'),
        (r'(\d{2})_(\d{2})_(\d{4})', '%m_%d_%Y'),
        
        # YYYY formats
        (r'(\d{4})', '%Y'),
    ]
    
    # Timestamp patterns
    TIMESTAMP_PATTERNS = [
        # With time: YYYY-MM-DD HH:MM:SS
        (r'(\d{4})-(\d{2})-(\d{2})[T\s](\d{2}):(\d{2}):(\d{2})', '%Y-%m-%d %H:%M:%S'),
        (r'(\d{4})(\d{2})(\d{2})[T\s](\d{2})(\d{2})(\d{2})', '%Y%m%d %H%M%S'),
        
        # With time: YYYY-MM-DD HH:MM
        (r'(\d{4})-(\d{2})-(\d{2})[T\s](\d{2}):(\d{2})', '%Y-%m-%d %H:%M'),
        (r'(\d{4})(\d{2})(\d{2})[T\s](\d{2})(\d{2})', '%Y%m%d %H%M'),
    ]
    
    # Month name patterns
    MONTH_PATTERNS = [
        # Full month names
        (r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', 'month_name'),
        
        # Short month names
        (r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b', 'month_abbr'),
    ]
    
    MONTH_NAMES = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12,
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
        'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
        'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
    }
    
    @staticmethod
    def extract_date_from_filename(filename: Union[str, Path]) -> Optional[datetime]:
        """Extract date from filename using various patterns."""
        if isinstance(filename, Path):
            filename = filename.name
        
        filename_lower = filename.lower()
        
        # Try timestamp patterns first (more specific)
        for pattern, date_format in DateUtils.TIMESTAMP_PATTERNS:
            match = re.search(pattern, filename)
            if match:
                try:
                    date_str = ' '.join(match.groups())
                    return datetime.strptime(date_str, date_format)
                except ValueError:
                    continue
        
        # Try date patterns
        for pattern, date_format in DateUtils.DATE_PATTERNS:
            match = re.search(pattern, filename)
            if match:
                try:
                    if len(match.groups()) == 1:  # Year only
                        year = int(match.group(1))
                        return datetime(year, 1, 1)
                    else:
                        date_str = '-'.join(match.groups())
                        dt = datetime.strptime(date_str, date_format)
                        return dt
                except ValueError:
                    continue
        
        # Try month name patterns
        month_match = None
        year_match = None
        
        for pattern, pattern_type in DateUtils.MONTH_PATTERNS:
            match = re.search(pattern, filename_lower)
            if match:
                month_name = match.group(1)
                month_match = DateUtils.MONTH_NAMES.get(month_name)
                break
        
        # Look for year near month
        if month_match:
            year_pattern = r'\b(\d{4})\b'
            year_matches = re.findall(year_pattern, filename)
            for year_str in year_matches:
                year = int(year_str)
                if 1900 <= year <= 2100:  # Reasonable year range
                    year_match = year
                    break
        
        if month_match and year_match:
            return datetime(year_match, month_match, 1)
        
        return None
    
    @staticmethod
    def parse_date_string(date_str: str) -> Optional[datetime]:
        """Parse date from string using various formats."""
        if not date_str:
            return None
        
        date_str = date_str.strip()
        
        # Common formats to try
        formats = [
            '%Y-%m-%d',
            '%Y/%m/%d',
            '%d-%m-%Y',
            '%d/%m/%Y',
            '%m-%d-%Y',
            '%m/%d/%Y',
            '%Y-%m-%d %H:%M:%S',
            '%Y/%m/%d %H:%M:%S',
            '%d-%m-%Y %H:%M:%S',
            '%d/%m/%Y %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%S.%fZ',
            '%Y%m%d',
            '%Y%m%d%H%M%S',
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    @staticmethod
    def format_date_for_path(dt: datetime, format_type: str = 'YYYY/MM/DD') -> str:
        """Format date for use in file paths."""
        if format_type == 'YYYY/MM/DD':
            return dt.strftime('%Y/%m/%d')
        elif format_type == 'YYYY-MM-DD':
            return dt.strftime('%Y-%m-%d')
        elif format_type == 'YYYY/MM':
            return dt.strftime('%Y/%m')
        elif format_type == 'YYYY':
            return dt.strftime('%Y')
        else:
            return dt.strftime(format_type)
    
    @staticmethod
    def get_date_parts(dt: datetime) -> dict:
        """Get date components as dictionary."""
        return {
            'year': dt.year,
            'month': dt.month,
            'day': dt.day,
            'month_name': calendar.month_name[dt.month],
            'month_abbr': calendar.month_abbr[dt.month],
            'weekday': dt.weekday(),
            'weekday_name': calendar.day_name[dt.weekday()],
            'quarter': (dt.month - 1) // 3 + 1,
            'day_of_year': dt.timetuple().tm_yday,
            'week_of_year': dt.isocalendar()[1],
        }
    
    @staticmethod
    def generate_date_paths(dt: datetime) -> List[str]:
        """Generate various date-based path formats."""
        parts = DateUtils.get_date_parts(dt)
        
        paths = [
            # Year/Month/Day
            f"{parts['year']}/{parts['month']:02d}/{parts['day']:02d}",
            
            # Year/Month
            f"{parts['year']}/{parts['month']:02d}",
            
            # Year/Quarter
            f"{parts['year']}/Q{parts['quarter']}",
            
            # Year/Month Name
            f"{parts['year']}/{parts['month_name']}",
            
            # Year only
            f"{parts['year']}",
            
            # Year/Week
            f"{parts['year']}/Week{parts['week_of_year']:02d}",
        ]
        
        return paths
    
    @staticmethod
    def create_date_hierarchy(base_path: Path, dt: datetime) -> Path:
        """Create date-based directory hierarchy."""
        year_dir = base_path / str(dt.year)
        month_dir = year_dir / f"{dt.month:02d}"
        day_dir = month_dir / f"{dt.day:02d}"
        
        # Create directories
        day_dir.mkdir(parents=True, exist_ok=True)
        
        return day_dir
    
    @staticmethod
    def is_valid_date_range(start_date: datetime, end_date: datetime) -> bool:
        """Check if date range is valid."""
        return start_date <= end_date
    
    @staticmethod
    def get_date_from_path(path: Union[str, Path]) -> Optional[datetime]:
        """Extract date from file path components."""
        path_obj = Path(path)
        
        # Try to parse date from path parts
        parts = path_obj.parts
        
        # Look for YYYY/MM/DD pattern in path
        for i in range(len(parts) - 2):
            try:
                year = int(parts[i])
                month = int(parts[i + 1])
                day = int(parts[i + 2])
                
                if 1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
                    return datetime(year, month, day)
            except (ValueError, IndexError):
                continue
        
        # Look for YYYY/MM pattern
        for i in range(len(parts) - 1):
            try:
                year = int(parts[i])
                month = int(parts[i + 1])
                
                if 1900 <= year <= 2100 and 1 <= month <= 12:
                    return datetime(year, month, 1)
            except (ValueError, IndexError):
                continue
        
        # Look for just year
        for part in parts:
            try:
                year = int(part)
                if 1900 <= year <= 2100:
                    return datetime(year, 1, 1)
            except ValueError:
                continue
        
        return None
    
    @staticmethod
    def suggest_date_from_context(
        filename: str, 
        file_created: Optional[datetime] = None,
        file_modified: Optional[datetime] = None,
        exif_date: Optional[datetime] = None
    ) -> Optional[datetime]:
        """Suggest best date from available context."""
        candidates = []
        
        # EXIF date is most reliable
        if exif_date:
            candidates.append((exif_date, 10))  # Highest priority
        
        # Date from filename
        filename_date = DateUtils.extract_date_from_filename(filename)
        if filename_date:
            candidates.append((filename_date, 8))
        
        # File creation/modification dates (lower priority)
        if file_created:
            candidates.append((file_created, 5))
        
        if file_modified:
            candidates.append((file_modified, 3))
        
        if not candidates:
            return None
        
        # Return highest priority date
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]


def extract_date_from_filename(filename: Union[str, Path]) -> Optional[datetime]:
    """Extract date from filename using various patterns."""
    return DateUtils.extract_date_from_filename(filename)


def parse_date_string(date_str: str) -> Optional[datetime]:
    """Parse date from string using various formats."""
    return DateUtils.parse_date_string(date_str)
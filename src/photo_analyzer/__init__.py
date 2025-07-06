"""
Local Photo Analyzer

A secure, privacy-first photo organization system that uses local AI to intelligently
analyze, rename, tag, and organize photos while maintaining complete data privacy.
"""

__version__ = "0.1.0"
__author__ = "Austin Orphan"
__email__ = "austin@example.com"
__license__ = "MIT"

from photo_analyzer.core.config import Config
from photo_analyzer.core.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Version info
__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "Config",
    "get_logger",
]
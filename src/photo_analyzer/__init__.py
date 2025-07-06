"""Photo Analyzer - Secure local LLM-based photo analysis and organization."""

__version__ = "0.1.0"
__author__ = "Photo Analyzer Team"
__description__ = "Secure local LLM-based photo analyzer with intelligent tagging and organization"

from .core.config import Config, get_config
from .core.logger import get_logger, setup_logging
from .pipeline.analyzer import PhotoAnalyzer
from .pipeline.processor import PhotoProcessor
from .pipeline.organizer import PhotoOrganizer

__all__ = [
    'Config',
    'get_config',
    'get_logger', 
    'setup_logging',
    'PhotoAnalyzer',
    'PhotoProcessor',
    'PhotoOrganizer',
]
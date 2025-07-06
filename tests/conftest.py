"""Pytest configuration and fixtures."""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from photo_analyzer.core.config import Config, DatabaseConfig, LLMConfig
from photo_analyzer.database.engine import DatabaseEngine
from photo_analyzer.database.session import get_async_db_session


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_config(temp_dir):
    """Create test configuration."""
    db_path = temp_dir / "test.db"
    
    config = Config(
        database=DatabaseConfig(
            type="sqlite",
            path=str(db_path),
            echo=False
        ),
        llm=LLMConfig(
            base_url="http://localhost:11434",
            default_model="llava:latest",
            timeout=30
        )
    )
    
    return config


@pytest.fixture
async def test_db_engine(test_config):
    """Create test database engine."""
    engine = DatabaseEngine(test_config)
    
    # Create tables
    await engine.create_all_tables()
    
    yield engine
    
    # Cleanup
    await engine.close()


@pytest.fixture
async def test_db_session(test_db_engine):
    """Create test database session."""
    async with test_db_engine.get_session() as session:
        yield session


@pytest.fixture
def sample_image_data():
    """Sample image metadata for testing."""
    return {
        'filename': 'IMG_001.jpg',
        'format': 'JPEG',
        'size': (1920, 1080),
        'file_size': 2048000,
        'file_hash': 'abc123def456',
        'aspect_ratio': 1.777,
        'orientation': 'landscape',
    }


@pytest.fixture
def sample_exif_data():
    """Sample EXIF data for testing."""
    return {
        'DateTime': datetime(2023, 6, 15, 14, 30, 0),
        'DateTimeOriginal': datetime(2023, 6, 15, 14, 30, 0),
        'camera_make': 'Canon',
        'camera_model': 'EOS R5',
        'lens_model': 'RF24-70mm F2.8 L IS USM',
        'iso': 200,
        'aperture': 2.8,
        'shutter_speed': 0.008,  # 1/125
        'focal_length': 50.0,
        'GPS': {
            'latitude': 37.7749,
            'longitude': -122.4194,
            'altitude': 10.0
        }
    }


@pytest.fixture
def mock_ollama_client():
    """Mock Ollama client for testing."""
    client = AsyncMock()
    
    # Mock analyze_image response
    client.analyze_image.return_value = {
        'description': 'A beautiful landscape with mountains and trees',
        'tags': ['landscape', 'mountains', 'trees', 'nature'],
        'confidence': 0.95,
        'model_used': 'llava:latest'
    }
    
    # Mock generate_filename response
    client.generate_filename.return_value = {
        'filename': 'mountain_landscape_trees.jpg',
        'confidence': 0.90
    }
    
    # Mock health check
    client.health_check.return_value = True
    
    return client


@pytest.fixture
def sample_photo_paths(temp_dir):
    """Create sample photo file paths for testing."""
    photos_dir = temp_dir / "photos"
    photos_dir.mkdir()
    
    # Create some dummy files
    sample_files = [
        "IMG_20230615_143000.jpg",
        "DSC_001.NEF",
        "photo_2023-06-15.png",
        "vacation_pic.jpeg"
    ]
    
    paths = []
    for filename in sample_files:
        file_path = photos_dir / filename
        file_path.write_text("dummy image data")
        paths.append(file_path)
    
    return paths


@pytest.fixture
def mock_image_processor():
    """Mock image processor for testing."""
    processor = MagicMock()
    
    processor.get_image_info.return_value = {
        'filename': 'test.jpg',
        'format': 'JPEG',
        'size': (1920, 1080),
        'width': 1920,
        'height': 1080,
        'file_size': 1024000,
        'aspect_ratio': 1.777,
        'orientation': 'landscape',
        'file_hash': 'abcdef123456'
    }
    
    processor.create_thumbnail.return_value = True
    processor.validate_image.return_value = {
        'is_valid': True,
        'exists': True,
        'is_file': True,
        'is_supported_format': True,
        'is_readable': True,
        'errors': [],
        'warnings': []
    }
    
    return processor


@pytest.fixture
def mock_exif_extractor():
    """Mock EXIF extractor for testing."""
    extractor = MagicMock()
    
    extractor.extract_exif.return_value = {
        'date_taken': datetime(2023, 6, 15, 14, 30, 0),
        'camera_make': 'Canon',
        'camera_model': 'EOS R5',
        'iso': 200,
        'aperture': 2.8,
        'focal_length': 50.0
    }
    
    extractor.get_date_taken.return_value = datetime(2023, 6, 15, 14, 30, 0)
    extractor.has_exif.return_value = True
    
    return extractor


# Event loop fixture for async tests
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Markers for different test categories
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "database: mark test as requiring database"
    )
    config.addinivalue_line(
        "markers", "llm: mark test as requiring LLM connection"
    )
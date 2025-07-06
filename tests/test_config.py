"""Tests for configuration system."""

import pytest
from pathlib import Path
import tempfile
import os

from photo_analyzer.core.config import Config, DatabaseConfig, LLMConfig, SecurityConfig


class TestConfig:
    """Test configuration system."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = Config()
        
        assert config.app_name == "Photo Analyzer"
        assert config.debug is False
        assert isinstance(config.database, DatabaseConfig)
        assert isinstance(config.llm, LLMConfig)
        assert isinstance(config.security, SecurityConfig)
    
    def test_database_config_defaults(self):
        """Test database configuration defaults."""
        db_config = DatabaseConfig()
        
        assert db_config.type == "sqlite"
        assert db_config.echo is False
        assert "photo_analyzer.db" in db_config.path
    
    def test_llm_config_defaults(self):
        """Test LLM configuration defaults."""
        llm_config = LLMConfig()
        
        assert llm_config.base_url == "http://localhost:11434"
        assert llm_config.default_model == "llava:latest"
        assert llm_config.timeout == 60
        assert llm_config.max_retries == 3
    
    def test_security_config_defaults(self):
        """Test security configuration defaults."""
        security_config = SecurityConfig()
        
        assert security_config.enable_audit_log is True
        assert security_config.max_file_size == 100 * 1024 * 1024  # 100MB
        assert len(security_config.allowed_extensions) > 0
    
    def test_config_from_env(self):
        """Test configuration from environment variables."""
        # Set environment variables
        env_vars = {
            'PHOTO_ANALYZER_DEBUG': 'true',
            'PHOTO_ANALYZER_DATABASE_TYPE': 'postgresql',
            'PHOTO_ANALYZER_DATABASE_HOST': 'localhost',
            'PHOTO_ANALYZER_DATABASE_PORT': '5432',
            'PHOTO_ANALYZER_LLM_BASE_URL': 'http://custom:11434',
            'PHOTO_ANALYZER_LLM_DEFAULT_MODEL': 'custom-model',
        }
        
        # Temporarily set environment variables
        old_env = {}
        for key, value in env_vars.items():
            old_env[key] = os.environ.get(key)
            os.environ[key] = value
        
        try:
            config = Config()
            
            assert config.debug is True
            assert config.database.type == "postgresql"
            assert config.database.host == "localhost"
            assert config.database.port == 5432
            assert config.llm.base_url == "http://custom:11434"
            assert config.llm.default_model == "custom-model"
            
        finally:
            # Restore environment
            for key, old_value in old_env.items():
                if old_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_value
    
    def test_sqlite_path_creation(self):
        """Test SQLite database path creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "subdir" / "test.db"
            
            config = Config(
                database=DatabaseConfig(
                    type="sqlite",
                    path=str(db_path)
                )
            )
            
            # The directory should be created when accessing the path
            assert config.database.path == str(db_path)
    
    def test_invalid_database_type(self):
        """Test invalid database type handling."""
        with pytest.raises(ValueError):
            DatabaseConfig(type="invalid_db_type")
    
    def test_postgresql_config(self):
        """Test PostgreSQL configuration."""
        config = DatabaseConfig(
            type="postgresql",
            host="localhost",
            port=5432,
            database="test_db",
            username="test_user",
            password="test_pass"
        )
        
        assert config.type == "postgresql"
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "test_db"
        assert config.username == "test_user"
        assert config.password == "test_pass"
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test negative timeout
        with pytest.raises(ValueError):
            LLMConfig(timeout=-1)
        
        # Test negative max_retries
        with pytest.raises(ValueError):
            LLMConfig(max_retries=-1)
        
        # Test negative max_file_size
        with pytest.raises(ValueError):
            SecurityConfig(max_file_size=-1)
    
    def test_config_serialization(self):
        """Test configuration serialization."""
        config = Config(
            app_name="Test App",
            debug=True,
            database=DatabaseConfig(type="sqlite", path="/tmp/test.db"),
            llm=LLMConfig(base_url="http://test:11434")
        )
        
        # Test dict conversion
        config_dict = config.model_dump()
        
        assert config_dict['app_name'] == "Test App"
        assert config_dict['debug'] is True
        assert config_dict['database']['type'] == "sqlite"
        assert config_dict['llm']['base_url'] == "http://test:11434"
        
        # Test reconstruction
        new_config = Config.model_validate(config_dict)
        
        assert new_config.app_name == config.app_name
        assert new_config.debug == config.debug
        assert new_config.database.type == config.database.type
        assert new_config.llm.base_url == config.llm.base_url


class TestConfigIntegration:
    """Integration tests for configuration system."""
    
    def test_config_file_loading(self, temp_dir):
        """Test loading configuration from file."""
        config_file = temp_dir / "config.json"
        
        config_data = {
            "app_name": "Test Photo Analyzer",
            "debug": True,
            "database": {
                "type": "sqlite",
                "path": str(temp_dir / "test.db")
            },
            "llm": {
                "base_url": "http://localhost:11434",
                "default_model": "test-model"
            }
        }
        
        import json
        with config_file.open('w') as f:
            json.dump(config_data, f)
        
        # Load config from file
        with config_file.open() as f:
            data = json.load(f)
            config = Config.model_validate(data)
        
        assert config.app_name == "Test Photo Analyzer"
        assert config.debug is True
        assert config.database.type == "sqlite"
        assert config.llm.default_model == "test-model"
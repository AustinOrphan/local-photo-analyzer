"""Configuration management for the photo analyzer."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import toml
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseModel):
    """Database configuration settings."""
    
    url: str = Field(default="sqlite:///photo_analyzer.db", description="Database URL")
    echo: bool = Field(default=False, description="Enable SQL query logging")
    pool_size: int = Field(default=10, description="Connection pool size")
    max_overflow: int = Field(default=20, description="Max overflow connections")


class LLMConfig(BaseModel):
    """Local LLM configuration settings."""
    
    primary_model: str = Field(default="llava", description="Primary vision model")
    fallback_model: str = Field(default="llama3.2-vision", description="Fallback model")
    ollama_url: str = Field(default="http://localhost:11434", description="Ollama API URL")
    timeout: int = Field(default=120, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    temperature: float = Field(default=0.3, description="Model temperature")
    max_tokens: int = Field(default=500, description="Maximum response tokens")


class OrganizationConfig(BaseModel):
    """File organization configuration settings."""
    
    date_format: str = Field(default="YYYY/MM/DD", description="Date folder structure")
    duplicate_handling: str = Field(default="smart_merge", description="Duplicate handling strategy")
    symlink_strategy: str = Field(default="relative", description="Symbolic link strategy")
    backup_before_move: bool = Field(default=True, description="Create backups before moving files")
    max_filename_length: int = Field(default=100, description="Maximum filename length")
    allowed_extensions: List[str] = Field(
        default=[".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".gif", ".webp", ".heic", ".raw"],
        description="Allowed image file extensions"
    )


class AnalysisConfig(BaseModel):
    """Photo analysis configuration settings."""
    
    confidence_threshold: float = Field(default=0.7, description="Minimum confidence for tags")
    max_tags_per_image: int = Field(default=10, description="Maximum tags per image")
    enable_face_detection: bool = Field(default=False, description="Enable face detection")
    enable_object_detection: bool = Field(default=True, description="Enable object detection")
    enable_scene_analysis: bool = Field(default=True, description="Enable scene analysis")
    batch_size: int = Field(default=10, description="Batch processing size")
    enable_gpu: bool = Field(default=True, description="Use GPU if available")


class SecurityConfig(BaseModel):
    """Security and privacy configuration settings."""
    
    enable_audit_logging: bool = Field(default=True, description="Enable audit logging")
    log_retention_days: int = Field(default=90, description="Log retention period")
    encrypt_database: bool = Field(default=False, description="Encrypt database")
    secure_temp_files: bool = Field(default=True, description="Use secure temporary files")
    require_confirmation: bool = Field(default=True, description="Require user confirmation for operations")


class UIConfig(BaseModel):
    """User interface configuration settings."""
    
    theme: str = Field(default="auto", description="UI theme (light/dark/auto)")
    page_size: int = Field(default=50, description="Items per page")
    thumbnail_size: int = Field(default=200, description="Thumbnail size in pixels")
    enable_previews: bool = Field(default=True, description="Enable image previews")
    auto_refresh: bool = Field(default=True, description="Auto-refresh UI")


class Config(BaseSettings):
    """Main configuration class."""
    
    # Application settings
    app_name: str = Field(default="Photo Analyzer", description="Application name")
    version: str = Field(default="0.1.0", description="Application version")
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Data directories
    data_dir: Path = Field(
        default_factory=lambda: Path.home() / ".local" / "share" / "photo-analyzer",
        description="Application data directory"
    )
    config_dir: Path = Field(
        default_factory=lambda: Path.home() / ".config" / "photo-analyzer",
        description="Configuration directory"
    )
    cache_dir: Path = Field(
        default_factory=lambda: Path.home() / ".cache" / "photo-analyzer",
        description="Cache directory"
    )
    log_dir: Path = Field(
        default_factory=lambda: Path.home() / ".local" / "share" / "photo-analyzer" / "logs",
        description="Log directory"
    )
    
    # Configuration sections
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    organization: OrganizationConfig = Field(default_factory=OrganizationConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    
    class Config:
        env_prefix = "PHOTO_ANALYZER_"
        env_nested_delimiter = "__"
        case_sensitive = False

    def __init__(self, config_file: Optional[Path] = None, **kwargs):
        """Initialize configuration with optional config file."""
        # Load from config file if provided
        if config_file and config_file.exists():
            file_config = self._load_config_file(config_file)
            kwargs.update(file_config)
        
        super().__init__(**kwargs)
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _load_config_file(self, config_file: Path) -> Dict[str, Any]:
        """Load configuration from file."""
        if config_file.suffix.lower() in ['.yaml', '.yml']:
            with open(config_file, 'r') as f:
                return yaml.safe_load(f) or {}
        elif config_file.suffix.lower() == '.toml':
            return toml.load(config_file)
        else:
            raise ValueError(f"Unsupported config file format: {config_file.suffix}")
    
    def _ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        for directory in [self.data_dir, self.config_dir, self.cache_dir, self.log_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def save_config(self, config_file: Optional[Path] = None) -> None:
        """Save current configuration to file."""
        if config_file is None:
            config_file = self.config_dir / "config.yaml"
        
        config_data = self.dict(exclude={'data_dir', 'config_dir', 'cache_dir', 'log_dir'})
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
    
    @classmethod
    def load_from_file(cls, config_file: Path) -> "Config":
        """Load configuration from file."""
        return cls(config_file=config_file)
    
    @property
    def database_url(self) -> str:
        """Get the database URL with data directory substitution."""
        if self.database.url.startswith("sqlite:///"):
            # Make SQLite path absolute within data directory
            db_file = self.database.url.replace("sqlite:///", "")
            if not os.path.isabs(db_file):
                db_file = self.data_dir / db_file
            return f"sqlite:///{db_file}"
        return self.database.url


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        # Try to load from default config file
        default_config_file = Path.home() / ".config" / "photo-analyzer" / "config.yaml"
        if default_config_file.exists():
            _config = Config.load_from_file(default_config_file)
        else:
            _config = Config()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


def reset_config() -> None:
    """Reset the global configuration instance."""
    global _config
    _config = None
"""Logging configuration for the photo analyzer."""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

import colorlog


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    enable_color: bool = True,
    enable_file_logging: bool = True,
) -> logging.Logger:
    """Set up logging configuration."""
    # Create root logger
    logger = logging.getLogger("photo_analyzer")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Console handler with color support
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    if enable_color:
        # Colored formatter for console
        console_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
    else:
        # Plain formatter for console
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if enabled and log_dir provided)
    if enable_file_logging and log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Main log file
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "photo_analyzer.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Error log file (errors and critical only)
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / "errors.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8"
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        logger.addHandler(error_handler)
    
    # Audit log handler (for security-sensitive operations)
    if log_dir:
        audit_handler = logging.handlers.RotatingFileHandler(
            log_dir / "audit.log",
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10,
            encoding="utf-8"
        )
        audit_handler.setLevel(logging.INFO)
        
        # More detailed formatter for audit logs
        audit_formatter = logging.Formatter(
            "%(asctime)s - AUDIT - %(levelname)s - %(name)s - %(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        audit_handler.setFormatter(audit_formatter)
        
        # Create audit logger
        audit_logger = logging.getLogger("photo_analyzer.audit")
        audit_logger.addHandler(audit_handler)
        audit_logger.setLevel(logging.INFO)
        # Prevent audit logs from propagating to root logger
        audit_logger.propagate = False
    
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """Get a logger instance."""
    if name is None:
        name = "photo_analyzer"
    elif not name.startswith("photo_analyzer"):
        name = f"photo_analyzer.{name}"
    
    logger = logging.getLogger(name)
    
    # If this is the first time getting the root logger, set it up
    if name == "photo_analyzer" and not logger.handlers:
        from photo_analyzer.core.config import get_config
        config = get_config()
        setup_logging(
            log_level=config.log_level,
            log_dir=config.log_dir,
            enable_color=not config.debug,  # Disable color in debug mode for better IDE integration
            enable_file_logging=True
        )
    
    return logger


def get_audit_logger() -> logging.Logger:
    """Get the audit logger for security-sensitive operations."""
    return logging.getLogger("photo_analyzer.audit")


class LogContext:
    """Context manager for adding context to log messages."""
    
    def __init__(self, logger: logging.Logger, **context):
        self.logger = logger
        self.context = context
        self.old_filter = None
    
    def __enter__(self):
        # Create a filter that adds context to log records
        def add_context(record):
            for key, value in self.context.items():
                setattr(record, key, value)
            return True
        
        self.old_filter = getattr(self.logger, '_context_filter', None)
        self.logger._context_filter = add_context
        self.logger.addFilter(add_context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.old_filter:
            self.logger.removeFilter(self.logger._context_filter)
            self.logger._context_filter = self.old_filter
        else:
            if hasattr(self.logger, '_context_filter'):
                self.logger.removeFilter(self.logger._context_filter)
                delattr(self.logger, '_context_filter')


def log_function_call(logger: logging.Logger = None):
    """Decorator to log function calls and returns."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)
            
            # Log function entry
            logger.debug(f"Entering {func.__name__} with args={args}, kwargs={kwargs}")
            
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Exiting {func.__name__} with result type: {type(result).__name__}")
                return result
            except Exception as e:
                logger.error(f"Exception in {func.__name__}: {e}", exc_info=True)
                raise
        
        return wrapper
    return decorator


def audit_log(operation: str, **details):
    """Log an audit event with optional details."""
    audit_logger = get_audit_logger()
    
    # Format details as key=value pairs
    detail_str = " ".join([f"{k}={v}" for k, v in details.items()])
    
    if detail_str:
        audit_logger.info(f"{operation} - {detail_str}")
    else:
        audit_logger.info(operation)
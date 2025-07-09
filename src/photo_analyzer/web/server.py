#!/usr/bin/env python3
"""
Web server startup script for Photo Analyzer.

This script provides an easy way to start the web server with various configuration options.
"""

import argparse
import asyncio
import sys
from pathlib import Path

import uvicorn

from photo_analyzer.core.config import get_config
from photo_analyzer.core.logger import get_logger

logger = get_logger(__name__)


async def check_dependencies():
    """Check that all required dependencies are available."""
    try:
        # Check database connectivity
        from photo_analyzer.database.engine import get_database_engine
        engine = get_database_engine()
        logger.info("Database engine initialized successfully")
        
        # Check LLM service
        from photo_analyzer.analyzer.llm_client import OllamaClient
        config = get_config()
        llm_client = OllamaClient(config)
        
        if await llm_client.check_connection():
            logger.info("LLM service (Ollama) is available")
        else:
            logger.warning("LLM service (Ollama) is not available - some features may be limited")
        
        return True
        
    except Exception as e:
        logger.error(f"Dependency check failed: {e}")
        return False


def main():
    """Main entry point for the web server."""
    parser = argparse.ArgumentParser(description="Photo Analyzer Web Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes (default: 1)")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"],
                        help="Log level (default: info)")
    parser.add_argument("--check-deps", action="store_true", help="Check dependencies before starting")
    parser.add_argument("--no-access-log", action="store_true", help="Disable access logging")
    
    args = parser.parse_args()
    
    # Check dependencies if requested
    if args.check_deps:
        print("Checking dependencies...")
        if not asyncio.run(check_dependencies()):
            print("Dependency check failed. Some features may not work correctly.")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                sys.exit(1)
    
    # Configuration info
    print(f"Starting Photo Analyzer Web Server")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Workers: {args.workers}")
    print(f"Reload: {args.reload}")
    print(f"Log Level: {args.log_level}")
    print(f"Web Interface: http://{args.host}:{args.port}")
    print(f"API Documentation: http://{args.host}:{args.port}/docs")
    print(f"Health Check: http://{args.host}:{args.port}/health")
    print("")
    print("Press Ctrl+C to stop the server")
    
    try:
        uvicorn.run(
            "photo_analyzer.web.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1,  # reload mode requires single worker
            log_level=args.log_level,
            access_log=not args.no_access_log,
            loop="asyncio"
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
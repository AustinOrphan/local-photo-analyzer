#!/usr/bin/env python3
"""
Test script to verify the web server can start and respond to basic requests.
"""

import asyncio
import sys
import time
import threading
import requests
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import uvicorn
from photo_analyzer.web.app import app


def start_server_in_thread(host="127.0.0.1", port=8002):
    """Start the server in a separate thread."""
    def run_server():
        uvicorn.run(app, host=host, port=port, log_level="warning")
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    return thread


def test_server_endpoints():
    """Test basic server endpoints."""
    base_url = "http://127.0.0.1:8002"
    
    # Wait for server to start
    print("Starting server...")
    server_thread = start_server_in_thread()
    time.sleep(3)  # Give server time to start
    
    try:
        # Test health endpoint
        print("Testing health endpoint...")
        response = requests.get(f"{base_url}/health", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        print(f"✓ Health check: {data}")
        
        # Test API info endpoint
        print("Testing API info endpoint...")
        response = requests.get(f"{base_url}/", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        print(f"✓ API info: {data}")
        
        # Test Ollama health (may fail if Ollama not running)
        print("Testing Ollama health endpoint...")
        try:
            response = requests.get(f"{base_url}/health/ollama", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Ollama health: {data}")
            else:
                print(f"⚠ Ollama health check failed with status {response.status_code}")
        except Exception as e:
            print(f"⚠ Ollama health check failed: {e}")
        
        # Test OpenAPI docs endpoint
        print("Testing OpenAPI docs...")
        response = requests.get(f"{base_url}/docs", timeout=5)
        assert response.status_code == 200
        print("✓ OpenAPI docs accessible")
        
        print("\n🎉 All tests passed! Web server is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_server_endpoints()
    sys.exit(0 if success else 1)
"""Ollama LLM client for photo analysis."""

import asyncio
import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx
from PIL import Image

from photo_analyzer.core.config import LLMConfig, get_config
from photo_analyzer.core.logger import get_logger

logger = get_logger(__name__)


class OllamaClient:
    """Client for interacting with Ollama API for photo analysis."""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize Ollama client."""
        self.config = config or get_config().llm
        self.base_url = self.config.ollama_url.rstrip('/')
        self.timeout = httpx.Timeout(self.config.timeout)
        
        logger.info(f"Initialized Ollama client with URL: {self.base_url}")
    
    async def check_connection(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/api/version")
                response.raise_for_status()
                
                version_info = response.json()
                logger.info(f"Connected to Ollama version: {version_info.get('version', 'unknown')}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            return False
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                
                data = response.json()
                models = data.get('models', [])
                
                logger.debug(f"Found {len(models)} available models")
                return models
                
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    async def pull_model(self, model_name: str) -> bool:
        """Pull a model if not already available."""
        try:
            # Check if model is already available
            models = await self.list_models()
            model_names = [m['name'] for m in models]
            
            if model_name in model_names:
                logger.info(f"Model {model_name} already available")
                return True
            
            logger.info(f"Pulling model {model_name}...")
            
            async with httpx.AsyncClient(timeout=httpx.Timeout(300)) as client:  # 5 minute timeout for pulling
                response = await client.post(
                    f"{self.base_url}/api/pull",
                    json={"name": model_name},
                    timeout=300
                )
                response.raise_for_status()
                
                logger.info(f"Successfully pulled model {model_name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False
    
    def _encode_image(self, image_path: Union[str, Path]) -> str:
        """Encode image to base64 string."""
        try:
            image_path = Path(image_path)
            
            # Open and potentially resize image if too large
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                
                # Resize if image is very large (to save on token usage)
                max_size = 1024
                if max(img.size) > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                    logger.debug(f"Resized image from {image_path} to {img.size}")
                
                # Save to bytes and encode
                import io
                byte_arr = io.BytesIO()
                img.save(byte_arr, format='JPEG', quality=85)
                byte_arr = byte_arr.getvalue()
                
                return base64.b64encode(byte_arr).decode('utf-8')
                
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            raise
    
    async def analyze_image(
        self,
        image_path: Union[str, Path],
        prompt: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Analyze an image using the specified model."""
        model = model or self.config.primary_model
        
        if prompt is None:
            prompt = self._get_default_analysis_prompt()
        
        try:
            # Encode the image
            base64_image = self._encode_image(image_path)
            
            # Prepare the request
            request_data = {
                "model": model,
                "prompt": prompt,
                "images": [base64_image],
                "stream": False,
                "options": {
                    "temperature": kwargs.get('temperature', self.config.temperature),
                    "num_predict": kwargs.get('max_tokens', self.config.max_tokens),
                }
            }
            
            logger.debug(f"Analyzing image {image_path} with model {model}")
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=request_data
                )
                response.raise_for_status()
                
                result = response.json()
                
                # Extract the response text
                analysis_text = result.get('response', '').strip()
                
                if not analysis_text:
                    raise ValueError("Empty response from model")
                
                logger.info(f"Successfully analyzed image {image_path}")
                
                return {
                    'model': model,
                    'prompt': prompt,
                    'response': analysis_text,
                    'raw_response': result,
                    'image_path': str(image_path),
                    'tokens_used': result.get('eval_count', 0) + result.get('prompt_eval_count', 0),
                }
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.error(f"Model {model} not found. Try pulling it first.")
                # Try fallback model if available
                if model != self.config.fallback_model:
                    logger.info(f"Trying fallback model: {self.config.fallback_model}")
                    return await self.analyze_image(
                        image_path, prompt, self.config.fallback_model, **kwargs
                    )
            logger.error(f"HTTP error analyzing image: {e}")
            raise
            
        except Exception as e:
            logger.error(f"Failed to analyze image {image_path}: {e}")
            raise
    
    def _get_default_analysis_prompt(self) -> str:
        """Get the default prompt for image analysis."""
        return """Analyze this image and provide the following information in JSON format:

{
  "description": "A detailed description of what you see in the image",
  "objects": ["list", "of", "objects", "you", "can", "identify"],
  "scene": "type of scene (indoor, outdoor, nature, urban, etc.)",
  "colors": ["dominant", "colors", "in", "the", "image"],
  "mood": "overall mood or atmosphere",
  "tags": ["relevant", "tags", "for", "organization"],
  "suggested_filename": "suggested descriptive filename",
  "confidence": 0.85
}

Be specific and accurate in your analysis. Focus on visible elements that would be useful for organizing and searching photos."""
    
    async def generate_description(
        self,
        image_path: Union[str, Path],
        model: Optional[str] = None
    ) -> str:
        """Generate a description of the image."""
        prompt = "Describe this image in detail, focusing on the main subjects, setting, and notable features."
        
        result = await self.analyze_image(image_path, prompt, model)
        return result['response']
    
    async def extract_tags(
        self,
        image_path: Union[str, Path],
        model: Optional[str] = None,
        max_tags: int = 10
    ) -> List[str]:
        """Extract relevant tags from the image."""
        prompt = f"""Analyze this image and provide up to {max_tags} relevant tags that would be useful for organizing and searching photos.
        
Return only a JSON list of tags, like: ["tag1", "tag2", "tag3"]

Focus on:
- Objects and subjects in the image
- Scene type and setting
- Activities or events
- Notable features or characteristics
- Colors if distinctive
- Mood or atmosphere if relevant

Be specific but use common terms that would be useful for searching."""
        
        result = await self.analyze_image(image_path, prompt, model)
        
        # Try to parse JSON response
        try:
            response_text = result['response'].strip()
            # Look for JSON array in the response
            import re
            json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
            if json_match:
                tags = json.loads(json_match.group())
                return tags[:max_tags]  # Limit to max_tags
            else:
                # Fallback: split by commas and clean up
                tags = [tag.strip().strip('"\'') for tag in response_text.split(',')]
                return [tag for tag in tags if tag][:max_tags]
                
        except Exception as e:
            logger.warning(f"Failed to parse tags JSON, using fallback: {e}")
            # Fallback: extract words that look like tags
            words = result['response'].split()
            tags = [word.strip('.,!?()[]{}') for word in words if len(word) > 2]
            return tags[:max_tags]
    
    async def suggest_filename(
        self,
        image_path: Union[str, Path],
        model: Optional[str] = None,
        max_length: int = 50
    ) -> str:
        """Suggest a descriptive filename for the image."""
        prompt = f"""Look at this image and suggest a descriptive filename that would be useful for organizing photos.

Requirements:
- Maximum {max_length} characters
- Use descriptive words about the main subject/content
- Use underscores or hyphens instead of spaces
- Don't include file extension
- Make it meaningful for searching later

Return only the suggested filename, nothing else."""
        
        result = await self.analyze_image(image_path, prompt, model)
        
        # Clean up the suggested filename
        filename = result['response'].strip().strip('"\'')
        
        # Remove file extension if present
        filename = Path(filename).stem
        
        # Replace spaces with underscores and clean up
        filename = filename.replace(' ', '_').replace('-', '_')
        filename = ''.join(c for c in filename if c.isalnum() or c in '_-.')
        
        # Truncate if too long
        if len(filename) > max_length:
            filename = filename[:max_length].rstrip('_-.')
        
        return filename or "analyzed_photo"
    
    async def analyze_batch(
        self,
        image_paths: List[Union[str, Path]],
        model: Optional[str] = None,
        max_concurrent: int = 3
    ) -> List[Dict[str, Any]]:
        """Analyze multiple images concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def analyze_with_semaphore(image_path):
            async with semaphore:
                try:
                    return await self.analyze_image(image_path, model=model)
                except Exception as e:
                    logger.error(f"Failed to analyze {image_path}: {e}")
                    return {
                        'image_path': str(image_path),
                        'error': str(e),
                        'success': False
                    }
        
        logger.info(f"Starting batch analysis of {len(image_paths)} images")
        
        tasks = [analyze_with_semaphore(path) for path in image_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log them
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Exception analyzing {image_paths[i]}: {result}")
                processed_results.append({
                    'image_path': str(image_paths[i]),
                    'error': str(result),
                    'success': False
                })
            else:
                processed_results.append(result)
        
        success_count = sum(1 for r in processed_results if r.get('success', True))
        logger.info(f"Batch analysis completed: {success_count}/{len(image_paths)} successful")
        
        return processed_results
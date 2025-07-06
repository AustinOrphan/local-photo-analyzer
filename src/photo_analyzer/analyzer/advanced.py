"""Advanced image analysis with multiple models and ensemble methods."""

import asyncio
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime

from PIL import Image, ImageStat
import imagehash

from photo_analyzer.core.config import get_config
from photo_analyzer.core.logger import get_logger, audit_log
from photo_analyzer.analyzer.llm_client import OllamaClient

logger = get_logger(__name__)


@dataclass
class AnalysisResult:
    """Enhanced analysis result with confidence scores and model agreement."""
    description: str
    tags: List[str]
    suggested_filename: str
    confidence_score: float
    model_consensus: float
    duplicate_hash: str
    image_quality: Dict[str, float]
    scene_analysis: Dict[str, Any]
    object_detection: List[Dict[str, Any]]
    color_analysis: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class ModelResult:
    """Result from a single model analysis."""
    model_name: str
    description: str
    tags: List[str]
    suggested_filename: str
    confidence: float
    processing_time: float
    raw_response: Dict[str, Any]


class AdvancedImageAnalyzer:
    """Advanced image analyzer with multiple models and ensemble methods."""
    
    def __init__(self, config=None):
        """Initialize the advanced analyzer."""
        self.config = config or get_config()
        self.llm_client = OllamaClient(self.config.llm)
        
        # Model configurations for ensemble analysis
        self.models = {
            'llava': {
                'name': 'llava',
                'strength': 'general_vision',
                'weight': 0.4,
                'timeout': 60
            },
            'llama3.2-vision': {
                'name': 'llama3.2-vision',
                'strength': 'detailed_analysis',
                'weight': 0.3,
                'timeout': 90
            },
            'bakllava': {
                'name': 'bakllava',
                'strength': 'artistic_content',
                'weight': 0.2,
                'timeout': 45
            },
            'moondream': {
                'name': 'moondream',
                'strength': 'technical_details',
                'weight': 0.1,
                'timeout': 30
            }
        }
        
        logger.info(f"Initialized advanced analyzer with {len(self.models)} models")
    
    async def analyze_image_advanced(
        self,
        image_path: Union[str, Path],
        use_ensemble: bool = True,
        quality_analysis: bool = True,
        duplicate_detection: bool = True,
        scene_analysis: bool = True
    ) -> AnalysisResult:
        """Perform comprehensive advanced analysis of an image."""
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        logger.info(f"Starting advanced analysis of {image_path}")
        start_time = datetime.now()
        
        try:
            # Parallel analysis tasks
            tasks = []
            
            # LLM ensemble analysis
            if use_ensemble:
                tasks.append(self._ensemble_analysis(image_path))
            else:
                tasks.append(self._single_model_analysis(image_path))
            
            # Image quality analysis
            if quality_analysis:
                tasks.append(self._analyze_image_quality(image_path))
            
            # Duplicate detection hash
            if duplicate_detection:
                tasks.append(self._generate_duplicate_hash(image_path))
            
            # Scene and color analysis
            if scene_analysis:
                tasks.append(self._analyze_scene_and_colors(image_path))
            
            # Execute all tasks in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            model_results = results[0] if not isinstance(results[0], Exception) else []
            quality_metrics = results[1] if len(results) > 1 and not isinstance(results[1], Exception) else {}
            duplicate_hash = results[2] if len(results) > 2 and not isinstance(results[2], Exception) else ""
            scene_data = results[3] if len(results) > 3 and not isinstance(results[3], Exception) else {}
            
            # Combine results using ensemble methods
            final_result = self._combine_model_results(
                model_results, quality_metrics, duplicate_hash, scene_data
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Advanced analysis completed in {processing_time:.2f}s")
            
            audit_log(
                "ADVANCED_ANALYSIS_COMPLETE",
                image_path=str(image_path),
                models_used=len(model_results),
                processing_time=processing_time,
                confidence=final_result.confidence_score
            )
            
            return final_result
            
        except Exception as e:
            logger.error(f"Advanced analysis failed for {image_path}: {e}")
            audit_log("ADVANCED_ANALYSIS_ERROR", image_path=str(image_path), error=str(e))
            raise
    
    async def _ensemble_analysis(self, image_path: Path) -> List[ModelResult]:
        """Perform ensemble analysis using multiple models."""
        available_models = await self.llm_client.list_models()
        available_names = [m['name'] for m in available_models]
        
        # Filter to only use available models
        models_to_use = {
            name: config for name, config in self.models.items()
            if config['name'] in available_names
        }
        
        if not models_to_use:
            logger.warning("No ensemble models available, falling back to primary model")
            return await self._single_model_analysis(image_path)
        
        logger.info(f"Running ensemble analysis with {len(models_to_use)} models")
        
        # Create analysis tasks for each model
        tasks = []
        for model_config in models_to_use.values():
            task = self._analyze_with_model(image_path, model_config)
            tasks.append(task)
        
        # Execute all model analyses
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out failed analyses
        successful_results = [
            result for result in results 
            if not isinstance(result, Exception)
        ]
        
        if not successful_results:
            raise RuntimeError("All ensemble models failed")
        
        logger.info(f"Ensemble analysis completed: {len(successful_results)}/{len(models_to_use)} models succeeded")
        return successful_results
    
    async def _single_model_analysis(self, image_path: Path) -> List[ModelResult]:
        """Perform analysis with a single model."""
        primary_model = self.config.llm.primary_model
        model_config = {
            'name': primary_model,
            'strength': 'general',
            'weight': 1.0,
            'timeout': 60
        }
        
        result = await self._analyze_with_model(image_path, model_config)
        return [result]
    
    async def _analyze_with_model(self, image_path: Path, model_config: Dict) -> ModelResult:
        """Analyze image with a specific model."""
        start_time = datetime.now()
        
        try:
            # Specialized prompt based on model strength
            prompt = self._get_specialized_prompt(model_config['strength'])
            
            # Perform analysis
            result = await self.llm_client.analyze_image(
                image_path,
                prompt=prompt,
                model=model_config['name']
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Parse structured response
            parsed = self._parse_model_response(result['response'])
            
            return ModelResult(
                model_name=model_config['name'],
                description=parsed.get('description', ''),
                tags=parsed.get('tags', []),
                suggested_filename=parsed.get('suggested_filename', ''),
                confidence=parsed.get('confidence', 0.5),
                processing_time=processing_time,
                raw_response=result
            )
            
        except Exception as e:
            logger.error(f"Model {model_config['name']} analysis failed: {e}")
            # Return empty result rather than failing completely
            return ModelResult(
                model_name=model_config['name'],
                description="",
                tags=[],
                suggested_filename="",
                confidence=0.0,
                processing_time=(datetime.now() - start_time).total_seconds(),
                raw_response={}
            )
    
    def _get_specialized_prompt(self, strength: str) -> str:
        """Get specialized prompt based on model strength."""
        prompts = {
            'general_vision': """Analyze this image and provide detailed information in JSON format:
{
  "description": "Comprehensive description of the image content",
  "tags": ["relevant", "descriptive", "tags"],
  "suggested_filename": "descriptive_filename",
  "confidence": 0.85,
  "objects": ["identified", "objects"],
  "scene_type": "indoor/outdoor/nature/urban/etc",
  "activities": ["any", "activities", "visible"]
}""",
            
            'detailed_analysis': """Provide an in-depth analysis of this image with technical details in JSON format:
{
  "description": "Detailed technical and contextual description",
  "tags": ["technical", "precise", "tags"],
  "suggested_filename": "technical_filename",
  "confidence": 0.85,
  "composition": "description of visual composition",
  "lighting": "analysis of lighting conditions",
  "quality_assessment": "image quality evaluation"
}""",
            
            'artistic_content': """Analyze the artistic and aesthetic elements of this image in JSON format:
{
  "description": "Artistic and aesthetic description",
  "tags": ["artistic", "style", "mood", "tags"],
  "suggested_filename": "artistic_filename",
  "confidence": 0.85,
  "artistic_style": "style or artistic movement",
  "mood": "emotional tone and atmosphere",
  "colors": ["dominant", "colors"]
}""",
            
            'technical_details': """Focus on technical aspects and fine details in this image in JSON format:
{
  "description": "Technical detail-focused description",
  "tags": ["technical", "detail", "tags"],
  "suggested_filename": "technical_filename",
  "confidence": 0.85,
  "technical_details": ["specific", "technical", "observations"],
  "text_content": "any visible text",
  "measurements": "any measurable elements"
}"""
        }
        
        return prompts.get(strength, prompts['general_vision'])
    
    def _parse_model_response(self, response: str) -> Dict[str, Any]:
        """Parse and extract structured data from model response."""
        import json
        import re
        
        try:
            # Try to parse as JSON first
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Fallback: extract information using patterns
        result = {}
        
        # Extract description
        desc_match = re.search(r'"description":\s*"([^"]*)"', response, re.IGNORECASE)
        if desc_match:
            result['description'] = desc_match.group(1)
        
        # Extract tags
        tags_match = re.search(r'"tags":\s*\[(.*?)\]', response, re.DOTALL)
        if tags_match:
            tags_str = tags_match.group(1)
            tags = [tag.strip().strip('"\'') for tag in tags_str.split(',')]
            result['tags'] = [tag for tag in tags if tag]
        
        # Extract filename
        filename_match = re.search(r'"suggested_filename":\s*"([^"]*)"', response, re.IGNORECASE)
        if filename_match:
            result['suggested_filename'] = filename_match.group(1)
        
        # Extract confidence
        conf_match = re.search(r'"confidence":\s*([0-9.]+)', response, re.IGNORECASE)
        if conf_match:
            result['confidence'] = float(conf_match.group(1))
        
        return result
    
    async def _analyze_image_quality(self, image_path: Path) -> Dict[str, float]:
        """Analyze technical image quality metrics."""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Get image statistics
                stat = ImageStat.Stat(img)
                
                # Calculate quality metrics
                metrics = {
                    'resolution': img.width * img.height,
                    'aspect_ratio': img.width / img.height,
                    'brightness': sum(stat.mean) / len(stat.mean) / 255.0,
                    'contrast': sum(stat.stddev) / len(stat.stddev) / 255.0,
                    'sharpness': self._calculate_sharpness(img),
                    'color_diversity': self._calculate_color_diversity(img),
                    'file_size': image_path.stat().st_size
                }
                
                return metrics
                
        except Exception as e:
            logger.warning(f"Quality analysis failed for {image_path}: {e}")
            return {}
    
    def _calculate_sharpness(self, img: Image.Image) -> float:
        """Calculate image sharpness using variance of Laplacian."""
        try:
            import cv2
            import numpy as np
            
            # Convert to grayscale array
            gray = np.array(img.convert('L'))
            
            # Calculate Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = laplacian.var()
            
            # Normalize to 0-1 range
            return min(variance / 1000.0, 1.0)
            
        except ImportError:
            logger.warning("OpenCV not available for sharpness calculation")
            return 0.5
        except Exception as e:
            logger.warning(f"Sharpness calculation failed: {e}")
            return 0.5
    
    def _calculate_color_diversity(self, img: Image.Image) -> float:
        """Calculate color diversity in the image."""
        try:
            # Get color histogram
            colors = img.getcolors(maxcolors=256*256*256)
            if not colors:
                return 0.0
            
            # Calculate entropy-based diversity
            total_pixels = sum(count for count, _ in colors)
            entropy = 0.0
            
            for count, _ in colors:
                p = count / total_pixels
                if p > 0:
                    entropy -= p * np.log2(p)
            
            # Normalize to 0-1 range
            max_entropy = np.log2(len(colors))
            return entropy / max_entropy if max_entropy > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Color diversity calculation failed: {e}")
            return 0.5
    
    async def _generate_duplicate_hash(self, image_path: Path) -> str:
        """Generate perceptual hash for duplicate detection."""
        try:
            with Image.open(image_path) as img:
                # Generate multiple hash types for robust duplicate detection
                phash = str(imagehash.phash(img))
                dhash = str(imagehash.dhash(img))
                whash = str(imagehash.whash(img))
                
                # Combine hashes
                combined = f"{phash}:{dhash}:{whash}"
                
                # Create final hash
                final_hash = hashlib.sha256(combined.encode()).hexdigest()[:16]
                
                return final_hash
                
        except Exception as e:
            logger.warning(f"Duplicate hash generation failed for {image_path}: {e}")
            return ""
    
    async def _analyze_scene_and_colors(self, image_path: Path) -> Dict[str, Any]:
        """Analyze scene composition and color palette."""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Extract dominant colors
                colors = self._extract_dominant_colors(img)
                
                # Analyze composition
                composition = self._analyze_composition(img)
                
                return {
                    'dominant_colors': colors,
                    'composition': composition,
                    'scene_complexity': self._calculate_scene_complexity(img)
                }
                
        except Exception as e:
            logger.warning(f"Scene analysis failed for {image_path}: {e}")
            return {}
    
    def _extract_dominant_colors(self, img: Image.Image, num_colors: int = 5) -> List[Dict[str, Any]]:
        """Extract dominant colors from image."""
        try:
            from collections import Counter
            
            # Resize image for faster processing
            img_small = img.resize((150, 150))
            
            # Get all colors
            colors = img_small.getdata()
            
            # Count color frequencies
            color_counts = Counter(colors)
            
            # Get most common colors
            dominant = color_counts.most_common(num_colors)
            
            result = []
            total_pixels = len(colors)
            
            for color, count in dominant:
                percentage = (count / total_pixels) * 100
                hex_color = '#{:02x}{:02x}{:02x}'.format(*color)
                
                result.append({
                    'rgb': color,
                    'hex': hex_color,
                    'percentage': round(percentage, 2)
                })
            
            return result
            
        except Exception as e:
            logger.warning(f"Color extraction failed: {e}")
            return []
    
    def _analyze_composition(self, img: Image.Image) -> Dict[str, Any]:
        """Analyze image composition using rule of thirds and other principles."""
        try:
            width, height = img.size
            
            # Rule of thirds grid
            third_x = width // 3
            third_y = height // 3
            
            # Convert to grayscale for analysis
            gray = img.convert('L')
            gray_array = np.array(gray)
            
            # Calculate interest points
            interest_points = []
            
            # Rule of thirds intersections
            for x in [third_x, 2 * third_x]:
                for y in [third_y, 2 * third_y]:
                    if x < width and y < height:
                        interest_points.append({
                            'x': x,
                            'y': y,
                            'type': 'rule_of_thirds'
                        })
            
            return {
                'aspect_ratio': width / height,
                'interest_points': interest_points,
                'rule_of_thirds_score': self._calculate_rule_of_thirds_score(gray_array),
                'symmetry_score': self._calculate_symmetry_score(gray_array)
            }
            
        except Exception as e:
            logger.warning(f"Composition analysis failed: {e}")
            return {}
    
    def _calculate_rule_of_thirds_score(self, gray_array: np.ndarray) -> float:
        """Calculate how well the image follows rule of thirds."""
        try:
            height, width = gray_array.shape
            
            # Define rule of thirds lines
            lines = [
                width // 3, 2 * width // 3,  # Vertical lines
                height // 3, 2 * height // 3  # Horizontal lines
            ]
            
            # Calculate edge strength along these lines
            # This is a simplified implementation
            score = 0.0
            
            # Vertical lines
            for x in lines[:2]:
                if x < width - 1:
                    edge_strength = np.mean(np.abs(gray_array[:, x] - gray_array[:, x + 1]))
                    score += edge_strength
            
            # Horizontal lines
            for y in lines[2:]:
                if y < height - 1:
                    edge_strength = np.mean(np.abs(gray_array[y, :] - gray_array[y + 1, :]))
                    score += edge_strength
            
            # Normalize to 0-1 range
            return min(score / (4 * 255), 1.0)
            
        except Exception as e:
            logger.warning(f"Rule of thirds calculation failed: {e}")
            return 0.5
    
    def _calculate_symmetry_score(self, gray_array: np.ndarray) -> float:
        """Calculate image symmetry score."""
        try:
            height, width = gray_array.shape
            
            # Horizontal symmetry
            left_half = gray_array[:, :width//2]
            right_half = np.fliplr(gray_array[:, width//2:])
            
            # Resize to match if needed
            min_width = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_width]
            right_half = right_half[:, :min_width]
            
            # Calculate difference
            diff = np.mean(np.abs(left_half - right_half))
            symmetry = 1.0 - (diff / 255.0)
            
            return max(0.0, symmetry)
            
        except Exception as e:
            logger.warning(f"Symmetry calculation failed: {e}")
            return 0.5
    
    def _calculate_scene_complexity(self, img: Image.Image) -> float:
        """Calculate scene complexity based on edge density."""
        try:
            import cv2
            
            # Convert to grayscale
            gray = np.array(img.convert('L'))
            
            # Calculate edges using Canny
            edges = cv2.Canny(gray, 50, 150)
            
            # Calculate edge density
            edge_pixels = np.sum(edges > 0)
            total_pixels = edges.size
            
            complexity = edge_pixels / total_pixels
            
            return complexity
            
        except ImportError:
            logger.warning("OpenCV not available for complexity calculation")
            return 0.5
        except Exception as e:
            logger.warning(f"Complexity calculation failed: {e}")
            return 0.5
    
    def _combine_model_results(
        self,
        model_results: List[ModelResult],
        quality_metrics: Dict[str, float],
        duplicate_hash: str,
        scene_data: Dict[str, Any]
    ) -> AnalysisResult:
        """Combine results from multiple models using ensemble methods."""
        if not model_results:
            raise ValueError("No model results to combine")
        
        # Calculate weighted averages and consensus
        total_weight = sum(
            self.models.get(result.model_name, {}).get('weight', 1.0)
            for result in model_results
        )
        
        # Combine descriptions (use highest confidence)
        best_result = max(model_results, key=lambda r: r.confidence)
        final_description = best_result.description
        
        # Combine tags with frequency weighting
        tag_scores = {}
        for result in model_results:
            weight = self.models.get(result.model_name, {}).get('weight', 1.0)
            for tag in result.tags:
                tag_scores[tag] = tag_scores.get(tag, 0) + weight * result.confidence
        
        # Select top tags
        sorted_tags = sorted(tag_scores.items(), key=lambda x: x[1], reverse=True)
        final_tags = [tag for tag, _ in sorted_tags[:self.config.analysis.max_tags_per_image]]
        
        # Combine filenames (use highest confidence)
        final_filename = best_result.suggested_filename or "analyzed_photo"
        
        # Calculate overall confidence
        weighted_confidence = sum(
            result.confidence * self.models.get(result.model_name, {}).get('weight', 1.0)
            for result in model_results
        ) / total_weight
        
        # Calculate model consensus
        consensus = self._calculate_model_consensus(model_results)
        
        return AnalysisResult(
            description=final_description,
            tags=final_tags,
            suggested_filename=final_filename,
            confidence_score=weighted_confidence,
            model_consensus=consensus,
            duplicate_hash=duplicate_hash,
            image_quality=quality_metrics,
            scene_analysis=scene_data,
            object_detection=[],  # Placeholder for future object detection
            color_analysis=scene_data.get('dominant_colors', []),
            metadata={
                'models_used': [r.model_name for r in model_results],
                'processing_times': {r.model_name: r.processing_time for r in model_results},
                'individual_confidences': {r.model_name: r.confidence for r in model_results}
            }
        )
    
    def _calculate_model_consensus(self, model_results: List[ModelResult]) -> float:
        """Calculate agreement between models."""
        if len(model_results) <= 1:
            return 1.0
        
        # Calculate tag overlap
        all_tags = []
        for result in model_results:
            all_tags.extend(result.tags)
        
        if not all_tags:
            return 0.5
        
        # Count tag frequencies
        from collections import Counter
        tag_counts = Counter(all_tags)
        
        # Calculate consensus as percentage of overlapping tags
        overlapping_tags = sum(1 for count in tag_counts.values() if count > 1)
        total_unique_tags = len(tag_counts)
        
        if total_unique_tags == 0:
            return 0.5
        
        consensus = overlapping_tags / total_unique_tags
        return consensus
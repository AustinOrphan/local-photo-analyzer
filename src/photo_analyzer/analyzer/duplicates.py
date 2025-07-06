"""Intelligent duplicate detection system."""

import asyncio
import hashlib
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

import imagehash
from PIL import Image
import numpy as np

from photo_analyzer.core.config import get_config
from photo_analyzer.core.logger import get_logger, audit_log
from photo_analyzer.models.photo import Photo

logger = get_logger(__name__)


@dataclass
class DuplicateGroup:
    """A group of duplicate or similar images."""
    representative_id: str
    photo_ids: List[str]
    similarity_score: float
    duplicate_type: str  # 'exact', 'near', 'similar'
    detection_method: str
    metadata: Dict[str, Any]


@dataclass
class SimilarityResult:
    """Result of comparing two images."""
    photo1_id: str
    photo2_id: str
    similarity_score: float
    hash_distance: int
    size_difference: float
    date_difference: float
    is_duplicate: bool
    similarity_type: str


class DuplicateDetector:
    """Intelligent duplicate detection using multiple algorithms."""
    
    def __init__(self, config=None):
        """Initialize duplicate detector."""
        self.config = config or get_config()
        
        # Similarity thresholds
        self.thresholds = {
            'exact_hash': 0,
            'near_duplicate': 5,
            'similar_image': 15,
            'size_similarity': 0.1,  # 10% size difference
            'date_similarity': 86400  # 1 day in seconds
        }
        
        logger.info("Initialized duplicate detector")
    
    async def detect_duplicates(
        self,
        photos: List[Photo],
        detection_types: List[str] = None
    ) -> List[DuplicateGroup]:
        """Detect duplicates among a list of photos."""
        if not photos:
            return []
        
        detection_types = detection_types or ['exact', 'near', 'similar']
        
        logger.info(f"Starting duplicate detection for {len(photos)} photos")
        start_time = datetime.now()
        
        try:
            # Generate hashes for all photos
            photo_hashes = await self._generate_all_hashes(photos)
            
            # Find duplicate groups using different methods
            duplicate_groups = []
            
            if 'exact' in detection_types:
                exact_groups = self._find_exact_duplicates(photos, photo_hashes)
                duplicate_groups.extend(exact_groups)
            
            if 'near' in detection_types:
                near_groups = self._find_near_duplicates(photos, photo_hashes)
                duplicate_groups.extend(near_groups)
            
            if 'similar' in detection_types:
                similar_groups = self._find_similar_images(photos, photo_hashes)
                duplicate_groups.extend(similar_groups)
            
            # Remove overlapping groups and merge when appropriate
            merged_groups = self._merge_overlapping_groups(duplicate_groups)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(
                f"Duplicate detection completed in {processing_time:.2f}s: "
                f"{len(merged_groups)} groups found"
            )
            
            audit_log(
                "DUPLICATE_DETECTION_COMPLETE",
                photos_analyzed=len(photos),
                groups_found=len(merged_groups),
                processing_time=processing_time
            )
            
            return merged_groups
            
        except Exception as e:
            logger.error(f"Duplicate detection failed: {e}")
            audit_log("DUPLICATE_DETECTION_ERROR", error=str(e))
            raise
    
    async def _generate_all_hashes(self, photos: List[Photo]) -> Dict[str, Dict[str, Any]]:
        """Generate all types of hashes for photos."""
        logger.info(f"Generating hashes for {len(photos)} photos")
        
        # Create tasks for parallel hash generation
        tasks = []
        for photo in photos:
            task = self._generate_photo_hashes(photo)
            tasks.append(task)
        
        # Execute in batches to avoid overwhelming the system
        batch_size = 10
        all_hashes = {}
        
        for i in range(0, len(tasks), batch_size):
            batch_tasks = tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for j, result in enumerate(batch_results):
                if not isinstance(result, Exception):
                    photo_id = photos[i + j].id
                    all_hashes[photo_id] = result
                else:
                    logger.warning(f"Failed to generate hash for photo {photos[i + j].id}: {result}")
        
        logger.info(f"Generated hashes for {len(all_hashes)} photos")
        return all_hashes
    
    async def _generate_photo_hashes(self, photo: Photo) -> Dict[str, Any]:
        """Generate multiple types of hashes for a single photo."""
        try:
            image_path = Path(photo.current_path)
            
            if not image_path.exists():
                raise FileNotFoundError(f"Photo not found: {image_path}")
            
            with Image.open(image_path) as img:
                # Generate perceptual hashes
                phash = imagehash.phash(img)
                dhash = imagehash.dhash(img)
                whash = imagehash.whash(img)
                ahash = imagehash.average_hash(img)
                
                # Generate content hash (MD5 of file)
                content_hash = self._generate_content_hash(image_path)
                
                # Get image metadata
                metadata = {
                    'size': img.size,
                    'mode': img.mode,
                    'file_size': image_path.stat().st_size,
                    'format': img.format
                }
                
                return {
                    'phash': phash,
                    'dhash': dhash,
                    'whash': whash,
                    'ahash': ahash,
                    'content_hash': content_hash,
                    'metadata': metadata
                }
                
        except Exception as e:
            logger.error(f"Failed to generate hashes for {photo.current_path}: {e}")
            return {}
    
    def _generate_content_hash(self, image_path: Path) -> str:
        """Generate MD5 hash of file content."""
        try:
            hasher = hashlib.md5()
            with open(image_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.warning(f"Failed to generate content hash for {image_path}: {e}")
            return ""
    
    def _find_exact_duplicates(
        self,
        photos: List[Photo],
        photo_hashes: Dict[str, Dict[str, Any]]
    ) -> List[DuplicateGroup]:
        """Find exact duplicates using content hashes."""
        content_hash_groups = defaultdict(list)
        
        # Group photos by content hash
        for photo in photos:
            photo_id = photo.id
            if photo_id in photo_hashes:
                content_hash = photo_hashes[photo_id].get('content_hash', '')
                if content_hash:
                    content_hash_groups[content_hash].append(photo_id)
        
        # Create duplicate groups for photos with same content hash
        duplicate_groups = []
        for content_hash, photo_ids in content_hash_groups.items():
            if len(photo_ids) > 1:
                # Choose representative (e.g., oldest photo)
                representative_id = photo_ids[0]  # Simplified selection
                
                duplicate_groups.append(DuplicateGroup(
                    representative_id=representative_id,
                    photo_ids=photo_ids,
                    similarity_score=1.0,
                    duplicate_type='exact',
                    detection_method='content_hash',
                    metadata={'content_hash': content_hash}
                ))
        
        logger.info(f"Found {len(duplicate_groups)} exact duplicate groups")
        return duplicate_groups
    
    def _find_near_duplicates(
        self,
        photos: List[Photo],
        photo_hashes: Dict[str, Dict[str, Any]]
    ) -> List[DuplicateGroup]:
        """Find near duplicates using perceptual hashes."""
        near_duplicate_groups = []
        processed_pairs = set()
        
        # Compare all pairs of photos
        for i, photo1 in enumerate(photos):
            for j, photo2 in enumerate(photos[i + 1:], i + 1):
                pair_key = tuple(sorted([photo1.id, photo2.id]))
                if pair_key in processed_pairs:
                    continue
                
                processed_pairs.add(pair_key)
                
                # Calculate similarity
                similarity = self._calculate_perceptual_similarity(
                    photo1.id, photo2.id, photo_hashes
                )
                
                if similarity and similarity.similarity_score >= 0.8:
                    # Check if either photo is already in a group
                    existing_group = None
                    for group in near_duplicate_groups:
                        if photo1.id in group.photo_ids or photo2.id in group.photo_ids:
                            existing_group = group
                            break
                    
                    if existing_group:
                        # Add to existing group
                        if photo1.id not in existing_group.photo_ids:
                            existing_group.photo_ids.append(photo1.id)
                        if photo2.id not in existing_group.photo_ids:
                            existing_group.photo_ids.append(photo2.id)
                    else:
                        # Create new group
                        near_duplicate_groups.append(DuplicateGroup(
                            representative_id=photo1.id,
                            photo_ids=[photo1.id, photo2.id],
                            similarity_score=similarity.similarity_score,
                            duplicate_type='near',
                            detection_method='perceptual_hash',
                            metadata={'hash_distance': similarity.hash_distance}
                        ))
        
        logger.info(f"Found {len(near_duplicate_groups)} near duplicate groups")
        return near_duplicate_groups
    
    def _find_similar_images(
        self,
        photos: List[Photo],
        photo_hashes: Dict[str, Dict[str, Any]]
    ) -> List[DuplicateGroup]:
        """Find similar images using multiple similarity metrics."""
        similar_groups = []
        processed_pairs = set()
        
        # Compare all pairs of photos
        for i, photo1 in enumerate(photos):
            for j, photo2 in enumerate(photos[i + 1:], i + 1):
                pair_key = tuple(sorted([photo1.id, photo2.id]))
                if pair_key in processed_pairs:
                    continue
                
                processed_pairs.add(pair_key)
                
                # Calculate comprehensive similarity
                similarity = self._calculate_comprehensive_similarity(
                    photo1, photo2, photo_hashes
                )
                
                if similarity and similarity.similarity_score >= 0.6:
                    # Check if either photo is already in a group
                    existing_group = None
                    for group in similar_groups:
                        if photo1.id in group.photo_ids or photo2.id in group.photo_ids:
                            existing_group = group
                            break
                    
                    if existing_group:
                        # Add to existing group
                        if photo1.id not in existing_group.photo_ids:
                            existing_group.photo_ids.append(photo1.id)
                        if photo2.id not in existing_group.photo_ids:
                            existing_group.photo_ids.append(photo2.id)
                    else:
                        # Create new group
                        similar_groups.append(DuplicateGroup(
                            representative_id=photo1.id,
                            photo_ids=[photo1.id, photo2.id],
                            similarity_score=similarity.similarity_score,
                            duplicate_type='similar',
                            detection_method='comprehensive',
                            metadata={
                                'hash_distance': similarity.hash_distance,
                                'size_difference': similarity.size_difference,
                                'date_difference': similarity.date_difference
                            }
                        ))
        
        logger.info(f"Found {len(similar_groups)} similar image groups")
        return similar_groups
    
    def _calculate_perceptual_similarity(
        self,
        photo1_id: str,
        photo2_id: str,
        photo_hashes: Dict[str, Dict[str, Any]]
    ) -> Optional[SimilarityResult]:
        """Calculate perceptual similarity between two photos."""
        try:
            hash1 = photo_hashes.get(photo1_id, {})
            hash2 = photo_hashes.get(photo2_id, {})
            
            if not hash1 or not hash2:
                return None
            
            # Calculate hash distances
            phash_dist = hash1.get('phash', imagehash.hex_to_hash('0'*16)) - hash2.get('phash', imagehash.hex_to_hash('0'*16))
            dhash_dist = hash1.get('dhash', imagehash.hex_to_hash('0'*16)) - hash2.get('dhash', imagehash.hex_to_hash('0'*16))
            whash_dist = hash1.get('whash', imagehash.hex_to_hash('0'*16)) - hash2.get('whash', imagehash.hex_to_hash('0'*16))
            
            # Use minimum distance (best match)
            min_distance = min(phash_dist, dhash_dist, whash_dist)
            
            # Convert distance to similarity score
            max_distance = 64  # Maximum possible hash distance
            similarity_score = 1.0 - (min_distance / max_distance)
            
            # Determine if it's a duplicate
            is_duplicate = min_distance <= self.thresholds['near_duplicate']
            
            similarity_type = 'exact' if min_distance == 0 else 'near' if is_duplicate else 'different'
            
            return SimilarityResult(
                photo1_id=photo1_id,
                photo2_id=photo2_id,
                similarity_score=similarity_score,
                hash_distance=min_distance,
                size_difference=0.0,
                date_difference=0.0,
                is_duplicate=is_duplicate,
                similarity_type=similarity_type
            )
            
        except Exception as e:
            logger.warning(f"Failed to calculate perceptual similarity: {e}")
            return None
    
    def _calculate_comprehensive_similarity(
        self,
        photo1: Photo,
        photo2: Photo,
        photo_hashes: Dict[str, Dict[str, Any]]
    ) -> Optional[SimilarityResult]:
        """Calculate comprehensive similarity using multiple factors."""
        try:
            # Get perceptual similarity
            perceptual = self._calculate_perceptual_similarity(
                photo1.id, photo2.id, photo_hashes
            )
            
            if not perceptual:
                return None
            
            # Calculate size similarity
            hash1 = photo_hashes.get(photo1.id, {})
            hash2 = photo_hashes.get(photo2.id, {})
            
            meta1 = hash1.get('metadata', {})
            meta2 = hash2.get('metadata', {})
            
            size1 = meta1.get('file_size', 0)
            size2 = meta2.get('file_size', 0)
            
            if size1 > 0 and size2 > 0:
                size_ratio = min(size1, size2) / max(size1, size2)
                size_difference = 1.0 - size_ratio
            else:
                size_difference = 1.0
            
            # Calculate date similarity
            date1 = photo1.date_taken or photo1.created_at
            date2 = photo2.date_taken or photo2.created_at
            
            if isinstance(date1, str):
                date1 = datetime.fromisoformat(date1.replace('Z', '+00:00'))
            if isinstance(date2, str):
                date2 = datetime.fromisoformat(date2.replace('Z', '+00:00'))
            
            date_difference = abs((date1 - date2).total_seconds())
            
            # Calculate weighted similarity score
            weights = {
                'perceptual': 0.6,
                'size': 0.2,
                'date': 0.2
            }
            
            # Normalize factors to 0-1 range
            size_score = 1.0 - min(size_difference / self.thresholds['size_similarity'], 1.0)
            date_score = 1.0 - min(date_difference / self.thresholds['date_similarity'], 1.0)
            
            # Calculate weighted average
            comprehensive_score = (
                weights['perceptual'] * perceptual.similarity_score +
                weights['size'] * size_score +
                weights['date'] * date_score
            )
            
            return SimilarityResult(
                photo1_id=photo1.id,
                photo2_id=photo2.id,
                similarity_score=comprehensive_score,
                hash_distance=perceptual.hash_distance,
                size_difference=size_difference,
                date_difference=date_difference,
                is_duplicate=comprehensive_score >= 0.8,
                similarity_type='similar'
            )
            
        except Exception as e:
            logger.warning(f"Failed to calculate comprehensive similarity: {e}")
            return None
    
    def _merge_overlapping_groups(self, groups: List[DuplicateGroup]) -> List[DuplicateGroup]:
        """Merge overlapping duplicate groups."""
        if not groups:
            return []
        
        merged_groups = []
        used_photo_ids = set()
        
        for group in sorted(groups, key=lambda g: g.similarity_score, reverse=True):
            # Check if any photos in this group are already used
            overlap = any(photo_id in used_photo_ids for photo_id in group.photo_ids)
            
            if not overlap:
                # No overlap, add the group
                merged_groups.append(group)
                used_photo_ids.update(group.photo_ids)
            else:
                # Find overlapping group and merge
                for existing_group in merged_groups:
                    if any(photo_id in existing_group.photo_ids for photo_id in group.photo_ids):
                        # Merge the groups
                        new_photo_ids = list(set(existing_group.photo_ids + group.photo_ids))
                        existing_group.photo_ids = new_photo_ids
                        
                        # Update similarity score (use higher score)
                        if group.similarity_score > existing_group.similarity_score:
                            existing_group.similarity_score = group.similarity_score
                        
                        used_photo_ids.update(group.photo_ids)
                        break
        
        logger.info(f"Merged {len(groups)} groups into {len(merged_groups)} groups")
        return merged_groups
    
    async def suggest_duplicate_resolution(
        self,
        duplicate_group: DuplicateGroup,
        photos: List[Photo]
    ) -> Dict[str, Any]:
        """Suggest how to resolve a duplicate group."""
        group_photos = [p for p in photos if p.id in duplicate_group.photo_ids]
        
        if not group_photos:
            return {'action': 'no_action', 'reason': 'No photos found'}
        
        # Analyze photos to determine best representative
        best_photo = self._select_best_representative(group_photos)
        photos_to_remove = [p for p in group_photos if p.id != best_photo.id]
        
        suggestions = {
            'action': 'keep_best_remove_others',
            'keep_photo_id': best_photo.id,
            'remove_photo_ids': [p.id for p in photos_to_remove],
            'reasoning': self._generate_resolution_reasoning(best_photo, photos_to_remove),
            'savings': {
                'files_removed': len(photos_to_remove),
                'space_saved': sum(p.file_size for p in photos_to_remove),
            }
        }
        
        return suggestions
    
    def _select_best_representative(self, photos: List[Photo]) -> Photo:
        """Select the best photo to keep from a duplicate group."""
        # Scoring criteria (higher is better)
        def score_photo(photo):
            score = 0
            
            # Prefer larger file size (higher quality)
            score += photo.file_size / 1000000  # MB
            
            # Prefer photos that have been analyzed
            if photo.analyzed:
                score += 10
            
            # Prefer photos that are organized
            if photo.organized:
                score += 5
            
            # Prefer photos with more tags
            score += len(photo.tags or [])
            
            # Prefer photos with descriptions
            if photo.description:
                score += 5
            
            # Prefer older photos (original timestamp)
            if photo.date_taken:
                # Older photos get higher score
                age_days = (datetime.now() - photo.date_taken).days
                score += min(age_days / 365, 10)  # Max 10 points for age
            
            return score
        
        # Return photo with highest score
        return max(photos, key=score_photo)
    
    def _generate_resolution_reasoning(
        self,
        keep_photo: Photo,
        remove_photos: List[Photo]
    ) -> List[str]:
        """Generate human-readable reasoning for duplicate resolution."""
        reasons = []
        
        # File size comparison
        keep_size = keep_photo.file_size
        avg_remove_size = sum(p.file_size for p in remove_photos) / len(remove_photos)
        
        if keep_size > avg_remove_size * 1.1:
            reasons.append(f"Keeping largest file ({keep_size / 1000000:.1f}MB)")
        
        # Analysis status
        if keep_photo.analyzed and not all(p.analyzed for p in remove_photos):
            reasons.append("Keeping analyzed photo")
        
        # Organization status
        if keep_photo.organized and not all(p.organized for p in remove_photos):
            reasons.append("Keeping organized photo")
        
        # Metadata richness
        keep_tags = len(keep_photo.tags or [])
        avg_remove_tags = sum(len(p.tags or []) for p in remove_photos) / len(remove_photos)
        
        if keep_tags > avg_remove_tags:
            reasons.append(f"Keeping photo with more metadata ({keep_tags} tags)")
        
        # Date preference
        if keep_photo.date_taken:
            reasons.append("Keeping photo with original timestamp")
        
        if not reasons:
            reasons.append("Keeping first photo in group")
        
        return reasons
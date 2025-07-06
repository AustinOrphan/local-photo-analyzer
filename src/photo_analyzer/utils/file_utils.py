"""File system utilities."""

import hashlib
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import os
import tempfile

from ..core.logger import get_logger

logger = get_logger(__name__)


class FileUtils:
    """File system operations and utilities."""
    
    @staticmethod
    def calculate_file_hash(file_path: Union[str, Path], algorithm: str = 'sha256') -> str:
        """Calculate hash of file contents."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        hash_func = getattr(hashlib, algorithm.lower())()
        
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    @staticmethod
    def safe_move_file(source: Union[str, Path], destination: Union[str, Path], 
                      backup_existing: bool = True) -> bool:
        """Safely move file with backup option."""
        src_path = Path(source)
        dst_path = Path(destination)
        
        if not src_path.exists():
            logger.error(f"Source file does not exist: {src_path}")
            return False
        
        try:
            # Ensure destination directory exists
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Handle existing destination file
            if dst_path.exists():
                if backup_existing:
                    backup_path = dst_path.with_suffix(dst_path.suffix + '.backup')
                    counter = 1
                    while backup_path.exists():
                        backup_path = dst_path.with_suffix(f'{dst_path.suffix}.backup.{counter}')
                        counter += 1
                    
                    shutil.move(str(dst_path), str(backup_path))
                    logger.info(f"Backed up existing file to: {backup_path}")
                else:
                    dst_path.unlink()
                    logger.info(f"Removed existing file: {dst_path}")
            
            # Move the file
            shutil.move(str(src_path), str(dst_path))
            logger.info(f"Moved file: {src_path} -> {dst_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to move file {src_path} to {dst_path}: {e}")
            return False
    
    @staticmethod
    def safe_copy_file(source: Union[str, Path], destination: Union[str, Path],
                      preserve_metadata: bool = True) -> bool:
        """Safely copy file with metadata preservation."""
        src_path = Path(source)
        dst_path = Path(destination)
        
        if not src_path.exists():
            logger.error(f"Source file does not exist: {src_path}")
            return False
        
        try:
            # Ensure destination directory exists
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy the file
            if preserve_metadata:
                shutil.copy2(str(src_path), str(dst_path))
            else:
                shutil.copy(str(src_path), str(dst_path))
            
            logger.debug(f"Copied file: {src_path} -> {dst_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to copy file {src_path} to {dst_path}: {e}")
            return False
    
    @staticmethod
    def create_symlink(target: Union[str, Path], link_path: Union[str, Path],
                      force: bool = False) -> bool:
        """Create symbolic link."""
        target_path = Path(target)
        link_path = Path(link_path)
        
        try:
            # Ensure link directory exists
            link_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Handle existing link
            if link_path.exists() or link_path.is_symlink():
                if force:
                    link_path.unlink()
                else:
                    logger.warning(f"Link already exists: {link_path}")
                    return False
            
            # Create symlink
            link_path.symlink_to(target_path)
            logger.debug(f"Created symlink: {link_path} -> {target_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create symlink {link_path} -> {target_path}: {e}")
            return False
    
    @staticmethod
    def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
        """Get detailed file information."""
        path = Path(file_path)
        
        if not path.exists():
            return {'exists': False, 'path': str(path)}
        
        stat = path.stat()
        
        return {
            'exists': True,
            'path': str(path),
            'name': path.name,
            'stem': path.stem,
            'suffix': path.suffix,
            'size': stat.st_size,
            'size_human': FileUtils.format_size(stat.st_size),
            'created': stat.st_ctime,
            'modified': stat.st_mtime,
            'accessed': stat.st_atime,
            'is_file': path.is_file(),
            'is_dir': path.is_dir(),
            'is_symlink': path.is_symlink(),
            'permissions': oct(stat.st_mode)[-3:],
            'parent': str(path.parent),
        }
    
    @staticmethod
    def format_size(size_bytes: int) -> str:
        """Format file size in human-readable format."""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        size = float(size_bytes)
        
        while size >= 1024.0 and i < len(size_names) - 1:
            size /= 1024.0
            i += 1
        
        return f"{size:.1f} {size_names[i]}"
    
    @staticmethod
    def find_files(directory: Union[str, Path], pattern: str = "*", 
                  recursive: bool = True, include_hidden: bool = False) -> List[Path]:
        """Find files matching pattern in directory."""
        dir_path = Path(directory)
        
        if not dir_path.exists():
            logger.warning(f"Directory does not exist: {dir_path}")
            return []
        
        try:
            if recursive:
                files = dir_path.rglob(pattern)
            else:
                files = dir_path.glob(pattern)
            
            result = []
            for file_path in files:
                if file_path.is_file():
                    if include_hidden or not file_path.name.startswith('.'):
                        result.append(file_path)
            
            return sorted(result)
            
        except Exception as e:
            logger.error(f"Failed to find files in {dir_path}: {e}")
            return []
    
    @staticmethod
    def ensure_directory(directory: Union[str, Path], mode: int = 0o755) -> bool:
        """Ensure directory exists with proper permissions."""
        dir_path = Path(directory)
        
        try:
            dir_path.mkdir(parents=True, exist_ok=True, mode=mode)
            return True
        except Exception as e:
            logger.error(f"Failed to create directory {dir_path}: {e}")
            return False
    
    @staticmethod
    def cleanup_empty_directories(root_dir: Union[str, Path], 
                                exclude_dirs: Optional[List[str]] = None) -> int:
        """Remove empty directories recursively."""
        root_path = Path(root_dir)
        exclude_dirs = exclude_dirs or []
        removed_count = 0
        
        try:
            # Walk directories bottom-up to handle nested empty dirs
            for dir_path in sorted(root_path.rglob('*'), reverse=True):
                if dir_path.is_dir() and dir_path.name not in exclude_dirs:
                    try:
                        # Check if directory is empty
                        if not any(dir_path.iterdir()):
                            dir_path.rmdir()
                            logger.debug(f"Removed empty directory: {dir_path}")
                            removed_count += 1
                    except OSError:
                        # Directory not empty or permission error
                        pass
            
            return removed_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup empty directories in {root_path}: {e}")
            return 0
    
    @staticmethod
    def atomic_write(file_path: Union[str, Path], content: Union[str, bytes],
                    mode: str = 'w', encoding: str = 'utf-8') -> bool:
        """Atomically write content to file."""
        path = Path(file_path)
        
        try:
            # Ensure directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to temporary file first
            with tempfile.NamedTemporaryFile(
                mode=mode,
                encoding=encoding if 'b' not in mode else None,
                dir=path.parent,
                prefix=f'.{path.name}.',
                suffix='.tmp',
                delete=False
            ) as tmp_file:
                tmp_file.write(content)
                tmp_path = Path(tmp_file.name)
            
            # Atomic move
            tmp_path.replace(path)
            logger.debug(f"Atomically wrote file: {path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to atomically write {path}: {e}")
            # Cleanup temporary file if it exists
            try:
                if 'tmp_path' in locals() and tmp_path.exists():
                    tmp_path.unlink()
            except:
                pass
            return False
    
    @staticmethod
    def get_available_space(directory: Union[str, Path]) -> int:
        """Get available disk space in bytes."""
        dir_path = Path(directory)
        
        try:
            statvfs = os.statvfs(dir_path)
            return statvfs.f_frsize * statvfs.f_bavail
        except Exception as e:
            logger.error(f"Failed to get available space for {dir_path}: {e}")
            return 0
    
    @staticmethod
    def is_same_file(path1: Union[str, Path], path2: Union[str, Path]) -> bool:
        """Check if two paths refer to the same file."""
        try:
            return Path(path1).samefile(Path(path2))
        except (OSError, FileNotFoundError):
            return False


def calculate_file_hash(file_path: Union[str, Path], algorithm: str = 'sha256') -> str:
    """Calculate hash of file contents."""
    return FileUtils.calculate_file_hash(file_path, algorithm)


def safe_move_file(source: Union[str, Path], destination: Union[str, Path], 
                  backup_existing: bool = True) -> bool:
    """Safely move file with backup option."""
    return FileUtils.safe_move_file(source, destination, backup_existing)
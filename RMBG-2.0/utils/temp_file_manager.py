"""
Temporary file management utilities
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List, Union
import atexit


class TempFileManager:
    """
    Unified temporary file management
    Eliminates duplicate temporary file handling code across multiple files
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize temporary file manager
        
        Args:
            base_dir: Base directory for temporary files (None for system default)
        """
        self.base_dir = base_dir
        self.temp_dirs: List[str] = []
        self.temp_files: List[str] = []
        
        # Register cleanup on exit
        atexit.register(self.cleanup_all)
    
    def create_temp_dir(self, prefix: str = "rmbg_") -> str:
        """
        Create a temporary directory
        
        Args:
            prefix: Prefix for directory name
            
        Returns:
            Path to temporary directory
        """
        if self.base_dir:
            os.makedirs(self.base_dir, exist_ok=True)
            temp_dir = tempfile.mkdtemp(prefix=prefix, dir=self.base_dir)
        else:
            temp_dir = tempfile.mkdtemp(prefix=prefix)
        
        self.temp_dirs.append(temp_dir)
        return temp_dir
    
    def create_temp_file(self, prefix: str = "rmbg_", suffix: str = ".tmp") -> str:
        """
        Create a temporary file
        
        Args:
            prefix: Prefix for file name
            suffix: Suffix for file name
            
        Returns:
            Path to temporary file
        """
        if self.base_dir:
            os.makedirs(self.base_dir, exist_ok=True)
            fd, temp_file = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=self.base_dir)
            os.close(fd)
        else:
            fd, temp_file = tempfile.mkstemp(prefix=prefix, suffix=suffix)
            os.close(fd)
        
        self.temp_files.append(temp_file)
        return temp_file
    
    def cleanup_dir(self, dir_path: str):
        """
        Clean up a specific temporary directory
        
        Args:
            dir_path: Path to directory to clean up
        """
        if dir_path in self.temp_dirs:
            try:
                shutil.rmtree(dir_path)
            except FileNotFoundError:
                pass
            except Exception:
                # 静默忽略其他清理异常，避免打扰用户
                pass
            finally:
                try:
                    self.temp_dirs.remove(dir_path)
                except ValueError:
                    pass
    
    def cleanup_file(self, file_path: str):
        """
        Clean up a specific temporary file
        
        Args:
            file_path: Path to file to clean up
        """
        if file_path in self.temp_files:
            try:
                os.remove(file_path)
            except FileNotFoundError:
                pass
            except Exception:
                # 静默忽略其他清理异常
                pass
            finally:
                try:
                    self.temp_files.remove(file_path)
                except ValueError:
                    pass
    
    def cleanup_all(self):
        """
        Clean up all temporary files and directories
        """
        # Clean up directories
        for dir_path in self.temp_dirs[:]:
            self.cleanup_dir(dir_path)
        
        # Clean up files
        for file_path in self.temp_files[:]:
            self.cleanup_file(file_path)
    
    def get_temp_dir_size(self) -> int:
        """
        Get total size of all temporary directories
        
        Returns:
            Total size in bytes
        """
        total_size = 0
        for dir_path in self.temp_dirs:
            if os.path.exists(dir_path):
                for root, dirs, files in os.walk(dir_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            total_size += os.path.getsize(file_path)
                        except OSError:
                            pass
        return total_size
    
    def list_temp_files(self) -> List[str]:
        """
        List all temporary files
        
        Returns:
            List of temporary file paths
        """
        all_files = []
        
        for dir_path in self.temp_dirs:
            if os.path.exists(dir_path):
                for root, dirs, files in os.walk(dir_path):
                    for file in files:
                        all_files.append(os.path.join(root, file))
        
        all_files.extend(self.temp_files)
        return all_files
    
    def __del__(self):
        """Cleanup on destruction"""
        self.cleanup_all()


# Global temporary file manager instance
_global_temp_manager = None


def get_temp_manager() -> TempFileManager:
    """Get the global temporary file manager instance"""
    global _global_temp_manager
    if _global_temp_manager is None:
        _global_temp_manager = TempFileManager()
    return _global_temp_manager


def create_temp_workspace(prefix: str = "rmbg_workspace_") -> str:
    """
    Create a temporary workspace for video processing
    
    Args:
        prefix: Prefix for workspace directory
        
    Returns:
        Path to workspace directory
    """
    manager = get_temp_manager()
    workspace = manager.create_temp_dir(prefix)
    
    # Create subdirectories
    os.makedirs(os.path.join(workspace, 'input_frames'), exist_ok=True)
    os.makedirs(os.path.join(workspace, 'processed_frames'), exist_ok=True)
    os.makedirs(os.path.join(workspace, 'background_frames'), exist_ok=True)
    os.makedirs(os.path.join(workspace, 'output'), exist_ok=True)
    
    return workspace


def cleanup_workspace(workspace: str):
    """
    Clean up a temporary workspace
    
    Args:
        workspace: Path to workspace directory
    """
    manager = get_temp_manager()
    manager.cleanup_dir(workspace)
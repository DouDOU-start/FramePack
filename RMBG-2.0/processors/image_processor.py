"""
Unified image processor for background removal and replacement
"""

import os
import sys
from pathlib import Path
from typing import Optional, Union, Tuple
from PIL import Image
import torch

# Import core modules
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent))
from core.model_manager import ModelManager
from core.color_utils import ColorUtils
from utils.temp_file_manager import get_temp_manager


class ImageProcessor:
    """
    Unified image processor for background operations
    Consolidates functionality from multiple files
    """
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize image processor
        
        Args:
            model_path: Path to the model directory
            device: Device to use for processing
        """
        self.model_manager = ModelManager(model_path, device)
        self.temp_manager = get_temp_manager()
    
    def remove_background(self, image_path: str, output_path: Optional[str] = None) -> str:
        """
        Remove background from an image
        
        Args:
            image_path: Path to input image
            output_path: Path to output image (optional)
            
        Returns:
            Path to output image
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Generate mask
        mask = self.model_manager.generate_mask(image)

        # Resize mask to image size if needed
        if mask.shape[-2:] != (image.height, image.width):
            mask_np = torch.nn.functional.interpolate(
                mask.unsqueeze(0).unsqueeze(0), size=(image.height, image.width), mode='bilinear', align_corners=False
            ).squeeze().cpu().numpy()
        else:
            mask_np = mask.cpu().numpy()

        # Convert mask to PIL Image
        mask_image = Image.fromarray((mask_np * 255).astype('uint8'))

        # Create transparent image using alpha from mask
        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        result = Image.new('RGBA', image.size)
        result.paste(image, (0, 0), mask_image)
        
        # Determine output path
        if output_path is None:
            output_path = self.temp_manager.create_temp_file(suffix='.png')
        
        # Save result
        result.save(output_path, 'PNG')
        
        return output_path
    
    def replace_background(self, image_path: str, background: Union[str, Tuple[int, int, int], Image.Image],
                          output_path: Optional[str] = None) -> str:
        """
        Replace background of an image
        
        Args:
            image_path: Path to input image
            background: Background color, RGB tuple, or image
            output_path: Path to output image (optional)
            
        Returns:
            Path to output image
        """
        # Remove background first
        transparent_path = self.remove_background(image_path)
        
        # Load transparent image
        transparent_image = Image.open(transparent_path)
        
        # Apply new background
        result = ColorUtils.create_background_image(transparent_image, background)
        
        # Determine output path
        if output_path is None:
            output_path = self.temp_manager.create_temp_file(suffix='.jpg')
        
        # Save result
        result.save(output_path, 'JPEG')
        
        # Clean up temporary file
        self.temp_manager.cleanup_file(transparent_path)
        
        return output_path
    
    def change_background_color(self, image_path: str, color: str, 
                               output_path: Optional[str] = None) -> str:
        """
        Change background to a specific color
        
        Args:
            image_path: Path to input image
            color: Background color (name or hex)
            output_path: Path to output image (optional)
            
        Returns:
            Path to output image
        """
        return self.replace_background(image_path, color, output_path)
    
    def process_batch(self, image_paths: list, background: Union[str, Tuple[int, int, int], Image.Image] = None,
                     output_dir: Optional[str] = None) -> list:
        """
        Process multiple images in batch
        
        Args:
            image_paths: List of input image paths
            background: Background to apply (optional)
            output_dir: Directory for output files (optional)
            
        Returns:
            List of output paths
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        results = []
        for i, image_path in enumerate(image_paths):
            try:
                if background:
                    output_path = self.replace_background(image_path, background)
                else:
                    output_path = self.remove_background(image_path)
                
                if output_dir:
                    # Move to output directory
                    filename = f"processed_{i+1}.png" if background else f"transparent_{i+1}.png"
                    final_path = os.path.join(output_dir, filename)
                    os.rename(output_path, final_path)
                    results.append(final_path)
                else:
                    results.append(output_path)
                    
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        return results
    
    def cleanup(self):
        """Clean up resources"""
        self.model_manager.cleanup()
        self.temp_manager.cleanup_all()
    
    def __del__(self):
        """Cleanup on destruction"""
        self.cleanup()


# Convenience functions for backward compatibility
def remove_background(image_path: str, model_path: str, output_path: Optional[str] = None) -> str:
    """
    Remove background from an image (convenience function)
    
    Args:
        image_path: Path to input image
        model_path: Path to model directory
        output_path: Path to output image (optional)
        
    Returns:
        Path to output image
    """
    processor = ImageProcessor(model_path)
    result = processor.remove_background(image_path, output_path)
    processor.cleanup()
    return result


def change_background_color(image_path: str, color: str, model_path: str,
                           output_path: Optional[str] = None) -> str:
    """
    Change background color (convenience function)
    
    Args:
        image_path: Path to input image
        color: Background color
        model_path: Path to model directory
        output_path: Path to output image (optional)
        
    Returns:
        Path to output image
    """
    processor = ImageProcessor(model_path)
    result = processor.change_background_color(image_path, color, output_path)
    processor.cleanup()
    return result
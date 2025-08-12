"""
Color utilities for background processing
"""

from typing import Union, Tuple
from PIL import ImageColor, Image
import re


class ColorUtils:
    """
    Unified color utilities for background processing
    Eliminates duplicate color parsing code across multiple files
    """
    
    # Predefined colors
    PREDEFINED_COLORS = {
        'white': (255, 255, 255),
        'black': (0, 0, 0),
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255),
        'yellow': (255, 255, 0),
        'cyan': (0, 255, 255),
        'magenta': (255, 0, 255),
        'gray': (128, 128, 128),
        'grey': (128, 128, 128),
        'silver': (192, 192, 192),
        'maroon': (128, 0, 0),
        'olive': (128, 128, 0),
        'purple': (128, 0, 128),
        'teal': (0, 128, 128),
        'navy': (0, 0, 128),
        'orange': (255, 165, 0),
        'pink': (255, 192, 203),
        'brown': (165, 42, 42),
        'transparent': (0, 0, 0, 0)
    }
    
    @staticmethod
    def parse_color_string(color_str: str) -> Tuple[int, int, int]:
        """
        Parse color string to RGB tuple
        
        Args:
            color_str: Color string (name, hex, or rgb)
            
        Returns:
            RGB tuple (r, g, b)
        """
        if not color_str:
            return (255, 255, 255)  # Default white
        
        color_str = color_str.strip().lower()
        
        # Check predefined colors
        if color_str in ColorUtils.PREDEFINED_COLORS:
            return ColorUtils.PREDEFINED_COLORS[color_str]
        
        # Parse hex color
        if color_str.startswith('#'):
            hex_color = color_str[1:]
            
            # Handle #RGB format
            if len(hex_color) == 3:
                hex_color = ''.join([c * 2 for c in hex_color])
            
            # Handle #RRGGBB format
            if len(hex_color) == 6:
                try:
                    r = int(hex_color[0:2], 16)
                    g = int(hex_color[2:4], 16)
                    b = int(hex_color[4:6], 16)
                    return (r, g, b)
                except ValueError:
                    pass
        
        # Parse rgb(r,g,b) format
        rgb_match = re.match(r'rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', color_str)
        if rgb_match:
            try:
                r = int(rgb_match.group(1))
                g = int(rgb_match.group(2))
                b = int(rgb_match.group(3))
                return (min(255, max(0, r)), min(255, max(0, g)), min(255, max(0, b)))
            except ValueError:
                pass
        
        # Use PIL's color parsing
        try:
            rgb = ImageColor.getrgb(color_str)
            return rgb
        except ValueError:
            pass
        
        # Fallback to white
        return (255, 255, 255)
    
    @staticmethod
    def create_background_image(image: Image.Image, background: Union[str, Tuple[int, int, int], Image.Image]) -> Image.Image:
        """
        Create background image and apply to foreground
        
        Args:
            image: Foreground image (with alpha channel)
            background: Background color string, RGB tuple, or image
            
        Returns:
            Composite image
        """
        if isinstance(background, str):
            bg_color = ColorUtils.parse_color_string(background)
            background = Image.new('RGB', image.size, bg_color)
        elif isinstance(background, tuple):
            background = Image.new('RGB', image.size, background)
        elif isinstance(background, Image.Image):
            # Resize background image to "cover" the foreground canvas, then center-crop
            if background.size != image.size:
                target_w, target_h = image.size
                bg_w, bg_h = background.size
                bg_ratio = bg_w / bg_h if bg_h != 0 else 1.0
                target_ratio = target_w / target_h if target_h != 0 else 1.0

                if bg_ratio > target_ratio:
                    # Background is wider → scale height to target_h, width will be larger than target_w
                    new_h = target_h
                    new_w = int(new_h * bg_ratio)
                else:
                    # Background is taller → scale width to target_w, height will be larger than target_h
                    new_w = target_w
                    new_h = int(new_w / bg_ratio) if bg_ratio != 0 else target_h

                resized = background.resize((max(1, new_w), max(1, new_h)), Image.Resampling.LANCZOS)

                # Center-crop to exactly the target size
                left = max(0, (resized.width - target_w) // 2)
                top = max(0, (resized.height - target_h) // 2)
                background = resized.crop((left, top, left + target_w, top + target_h))
        
        # Ensure image has alpha channel
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Composite images (ensure both RGBA and same size)
        bg_rgba = background.convert('RGBA')
        if bg_rgba.size != image.size:
            bg_rgba = bg_rgba.resize(image.size, Image.Resampling.LANCZOS)
        result = Image.alpha_composite(bg_rgba, image)
        
        # Convert back to RGB if no transparency needed
        if result.mode == 'RGBA':
            result = result.convert('RGB')
        
        return result
    
    @staticmethod
    def resize_background_image(background: Image.Image, target_size: Tuple[int, int], 
                               maintain_aspect: bool = True, 
                               background_color: Union[str, Tuple[int, int, int]] = 'black') -> Image.Image:
        """
        Resize background image to target size
        
        Args:
            background: Background image to resize
            target_size: Target size (width, height)
            maintain_aspect: Whether to maintain aspect ratio
            background_color: Color for letterboxing/pillarboxing
            
        Returns:
            Resized background image
        """
        if maintain_aspect:
            # Calculate resize dimensions maintaining aspect ratio
            bg_ratio = background.width / background.height
            target_ratio = target_size[0] / target_size[1]
            
            if bg_ratio > target_ratio:
                # Background is wider - fit to width
                new_width = target_size[0]
                new_height = int(target_size[0] / bg_ratio)
            else:
                # Background is taller - fit to height
                new_height = target_size[1]
                new_width = int(target_size[1] * bg_ratio)
            
            # Resize background
            background = background.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create canvas and center background
            bg_color = ColorUtils.parse_color_string(background_color) if isinstance(background_color, str) else background_color
            canvas = Image.new('RGB', target_size, bg_color)
            
            # Calculate position to center the background
            x = (target_size[0] - new_width) // 2
            y = (target_size[1] - new_height) // 2
            
            canvas.paste(background, (x, y))
            return canvas
        else:
            # Stretch to fit
            return background.resize(target_size, Image.Resampling.LANCZOS)
    
    @staticmethod
    def get_color_info(color_str: str) -> dict:
        """
        Get detailed information about a color
        
        Args:
            color_str: Color string
            
        Returns:
            Dictionary with color information
        """
        rgb = ColorUtils.parse_color_string(color_str)
        
        return {
            'rgb': rgb,
            'hex': f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}",
            'brightness': sum(rgb) / 3,
            'is_dark': sum(rgb) < 384,  # Average < 128
            'is_light': sum(rgb) > 512,  # Average > 170
            'complementary': (255 - rgb[0], 255 - rgb[1], 255 - rgb[2])
        }
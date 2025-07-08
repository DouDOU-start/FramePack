import os
import sys
import argparse
from PIL import Image
from bg_processor import BGProcessor

class BackgroundColorChanger:
    """
    A class to handle background color change using a pre-loaded AI model.
    This approach avoids reloading the model for each image, significantly
    improving performance when processing multiple files.
    """
    def __init__(self, model_path: str):
        """
        Initializes the changer by loading the model and setting up the device.
        
        Args:
            model_path (str): The path to the local Hugging Face model directory.
        """
        self.processor = BGProcessor(model_path)

    def process(self, input_path: str, output_path: str, bg_color: str = "black"):
        """
        Changes the background of a single image to a specified color.

        Args:
            input_path (str): Path to the input image.
            output_path (str): Path to save the output image with the new background.
            bg_color (str): The background color to apply (default: "black").
        """
        # 1. Open image
        try:
            image = Image.open(input_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Input file not found at '{input_path}'", file=sys.stderr)
            return

        # 2. Get mask
        print(f"Processing '{os.path.basename(input_path)}'...")
        mask = self.processor.get_mask(image)
        
        # 3. Create a new image with the specified background color
        background = Image.new('RGB', image.size, bg_color)
        
        # 4. Composite the original image onto the new background using the mask
        composite_image = Image.composite(image, background, mask)

        # 5. Save the result
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        composite_image.save(output_path)
        print(f"Successfully saved result to '{output_path}'")

def main():
    """
    Main function to parse command-line arguments and run the background color change process.
    """
    
    parser = argparse.ArgumentParser(
        description="Change the background of an image to a solid color using the RMBG-2.0 model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", "--input", required=True, help="Path to the input image file.")
    parser.add_argument("-o", "--output", required=True, help="Path to save the output image file.")
    parser.add_argument("-c", "--color", type=str, default="black", help="Background color to apply (e.g., 'black', 'white', '#RRGGBB').")
    parser.add_argument("--model-path", type=str, default="../hf_download/RMBG-2.0", help="Path to the local RMBG-2.0 model directory.")
    args = parser.parse_args()

    # --- Execution ---
    # 1. Instantiate the changer (loads the model once)
    changer = BackgroundColorChanger(model_path=args.model_path)
    
    # 2. Process the specified image
    changer.process(input_path=args.input, output_path=args.output, bg_color=args.color)


'''
使用示例：
python change_bg_color.py --input "../inputs/华佗.png" --output "../outputs/华佗_black_bg.png" --color "black"
python change_bg_color.py --input "../inputs/华佗.png" --output "../outputs/华佗_white_bg.png" --color "white"
'''
if __name__ == '__main__':
    main() 
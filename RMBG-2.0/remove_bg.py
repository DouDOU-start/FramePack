import os
import sys
import argparse
from PIL import Image
from bg_processor import BGProcessor

class BackgroundRemover:
    """
    A class to handle background removal using a pre-loaded AI model.
    This approach avoids reloading the model for each image, significantly
    improving performance when processing multiple files.
    """
    def __init__(self, model_path: str):
        """
        Initializes the remover by loading the model and setting up the device.
        
        Args:
            model_path (str): The path to the local Hugging Face model directory.
        """
        self.processor = BGProcessor(model_path)

    def process(self, input_path: str, output_path: str):
        """
        Removes the background from a single image.

        Args:
            input_path (str): Path to the input image.
            output_path (str): Path to save the output image with a transparent background.
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

        # 3. Apply the mask
        image.putalpha(mask)

        # 4. Save the result
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)
        print(f"Successfully saved result to '{output_path}'")

def main():
    """
    Main function to parse command-line arguments and run the background removal process.
    """
    
    parser = argparse.ArgumentParser(
        description="Remove the background from an image using the RMBG-2.0 model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", "--input", required=True, help="Path to the input image file.")
    parser.add_argument("-o", "--output", required=True, help="Path to save the output PNG file.")
    parser.add_argument("--model-path", type=str, default="../hf_download/RMBG-2.0", help="Path to the local RMBG-2.0 model directory.")
    args = parser.parse_args()

    # --- Execution ---
    # 1. Instantiate the remover (loads the model once)
    remover = BackgroundRemover(model_path=args.model_path)
    
    # 2. Process the specified image
    remover.process(input_path=args.input, output_path=args.output)


'''
使用示例：
python remove_bg.py --input "../inputs/华佗.png" --output "../outputs/华佗.png"
'''
if __name__ == '__main__':
    main()
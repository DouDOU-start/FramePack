import os
import sys
import argparse
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

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
        # 1. Initialize device and model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        try:
            print(f"Loading model from: {model_path}")
            self.model = AutoModelForImageSegmentation.from_pretrained(
                model_path, trust_remote_code=True
            ).to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error: Failed to load model from '{model_path}'.", file=sys.stderr)
            print(f"Reason: {e}", file=sys.stderr)
            sys.exit(1) # Exit if model loading fails
        
        # 2. Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        print("Model loaded successfully.")

    def process(self, input_path: str, output_path: str):
        """
        Removes the background from a single image.

        Args:
            input_path (str): Path to the input image.
            output_path (str): Path to save the output image with a transparent background.
        """
        # 1. Open and transform the image
        try:
            image = Image.open(input_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Input file not found at '{input_path}'", file=sys.stderr)
            return

        original_size = image.size
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 2. Perform prediction
        print(f"Processing '{os.path.basename(input_path)}'...")
        with torch.no_grad():
            # Use autocast for better performance on modern GPUs
            if self.device.type == 'cuda':
                 with torch.autocast(device_type="cuda"):
                    preds = self.model(input_tensor)[-1].sigmoid()
            else:
                 preds = self.model(input_tensor)[-1].sigmoid()

        # 3. Process the mask and apply it
        mask = transforms.ToPILImage()(preds.cpu().squeeze()).resize(original_size, Image.LANCZOS)
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
python remove_bg.py --input "../inputs/华佗.jpg" --output "../outputs/华佗.png"
'''
if __name__ == '__main__':
    main()
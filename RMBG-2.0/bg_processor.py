import os
import sys
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

class BGProcessor:
    """
    A class to handle background processing using a pre-loaded AI model.
    It encapsulates model loading and mask generation.
    """
    def __init__(self, model_path: str):
        """
        Initializes the processor by loading the model and setting up the device.
        
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
    
    def get_mask(self, image: Image.Image) -> Image.Image:
        """
        Generates a background mask for a given image.

        Args:
            image (Image.Image): The input PIL image (in RGB format).
        
        Returns:
            Image.Image: The generated mask as a PIL image.
        """
        original_size = image.size
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Perform prediction
        with torch.no_grad():
            if self.device.type == 'cuda':
                 with torch.autocast(device_type="cuda"):
                    preds = self.model(input_tensor)[-1].sigmoid()
            else:
                 preds = self.model(input_tensor)[-1].sigmoid()

        # Process the mask
        mask = transforms.ToPILImage()(preds.cpu().squeeze()).resize(original_size, Image.LANCZOS)
        return mask 
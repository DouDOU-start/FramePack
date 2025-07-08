import os
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

# Set Hugging Face home directory to save models to './hf_download'
os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), '../hf_download')))

def remove_background(input_image_path, output_image_path):
    """
    Removes the background from an image using the briaai/RMBG-2.0 model.
    """
    # Check for CUDA availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Construct the absolute path to the local model directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    local_model_path = os.path.join(project_root, 'hf_download', 'RMBG-2.0')

    print(f"Loading model from local directory: {local_model_path}")
    
    # Load model from the specified local directory
    try:
        model = AutoModelForImageSegmentation.from_pretrained(local_model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load model from {local_model_path}. Error: {e}")
        print("Please ensure the model files are correctly placed in 'hf_download/RMBG-2.0/'.")
        return
    
    # Set precision and move model to device
    torch.set_float32_matmul_precision('high')
    model.to(device)
    model.eval()
    print("Model loaded.")

    # Image transformations
    image_size = (1024, 1024)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Open and transform the image
    print(f"Opening image: {input_image_path}")
    if not os.path.exists(input_image_path):
        print(f"Error: Input image not found at {input_image_path}")
        # Create a dummy image if cats.jpg doesn't exist
        print("Creating a dummy black image for demonstration.")
        image = Image.new('RGB', (1280, 720), 'black')
        image.save(input_image_path)
        print(f"Dummy image saved as {input_image_path}")
    
    image = Image.open(input_image_path).convert("RGB")
    input_tensor = transform_image(image).unsqueeze(0).to(device)

    # Perform prediction
    print("Removing background...")
    with torch.no_grad():
        preds = model(input_tensor)[-1].sigmoid().cpu()
    
    # Process the mask
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)

    # Apply the mask as an alpha channel
    image.putalpha(mask)

    # Save the result
    # Ensure output directory exists
    output_dir = os.path.dirname(output_image_path)
    os.makedirs(output_dir, exist_ok=True)
    
    image.save(output_image_path)
    print(f"Background removed. Image saved to: {output_image_path}")


if __name__ == '__main__':
    # NOTE: Please make sure 'cats.jpg' exists in the project root directory.
    # If not, this script will create a dummy black image.
    input_path = "华佗-全身-√.png"
    output_path = "outputs/华佗-全身-√.png"
    remove_background(input_path, output_path) 
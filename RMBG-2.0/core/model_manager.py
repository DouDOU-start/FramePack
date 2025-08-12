"""
Core model management for background processing
"""

import torch
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from typing import Optional
import sys


class ModelManager:
    """
    Unified model manager for background processing
    Eliminates duplicate model loading code across multiple files
    """
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize model manager
        
        Args:
            model_path: Path to the model directory
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_path = model_path
        self.device = self._get_device(device)
        self.model = None
        self.transform = None
        self._load_model()
    
    def _get_device(self, device: Optional[str]) -> torch.device:
        """Get the appropriate device for model execution"""
        if device is None:
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _load_model(self):
        """Load the model and setup transformations"""
        try:
            print(f"Loading model from {self.model_path}...")
            print(f"Using device: {self.device}")
            
            # Load model (更安全的低显存加载：先CPU再迁移，半精度优先)
            model = AutoModelForImageSegmentation.from_pretrained(
                self.model_path, trust_remote_code=True
            )

            # 为避免第三方权重中混合精度不兼容，强制使用 fp32 更稳妥
            self.model = model.to(self.device, dtype=torch.float32)
            self.model.eval()
            
            # Setup transformations
            self.transform = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            print("Model loaded successfully")
            
        except Exception as e:
            # 不直接退出进程，抛出异常给上层 UI 处理，避免整个服务被杀死
            raise RuntimeError(f"Error loading model from '{self.model_path}': {e}")
    
    def generate_mask(self, image) -> torch.Tensor:
        """
        Generate background mask for an image
        
        Args:
            image: PIL Image to process
            
        Returns:
            Background mask tensor
        """
        # 容错：若属性缺失或未初始化，尝试重新加载
        if not hasattr(self, 'model') or not hasattr(self, 'transform'):
            self._load_model()
        if self.model is None or self.transform is None:
            raise RuntimeError("Model not loaded")
        
        # Transform image
        input_tensor = self.transform(image).unsqueeze(0)
        # Match device and dtype with model to avoid dtype mismatch (e.g., fp16 weights vs fp32 input)
        # 模型已强制 fp32，这里同样统一到 fp32
        input_tensor = input_tensor.to(self.device, dtype=torch.float32)
        
        # Forward
        with torch.no_grad():
            result = self.model(input_tensor)

        # Normalize various output structures to logits tensor
        logits = None
        if hasattr(result, 'logits'):
            logits = result.logits
        elif isinstance(result, dict) and 'logits' in result:
            logits = result['logits']
        elif torch.is_tensor(result):
            logits = result
        elif isinstance(result, (list, tuple)) and len(result) > 0:
            logits = result[-1] if torch.is_tensor(result[-1]) else None

        if logits is None:
            raise RuntimeError('Model output format unsupported: cannot find logits tensor')

        # logits: [B, C, H, W]
        # If C==1, treat as alpha matte via sigmoid; else semantic seg, treat non-zero class as foreground
        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits)
            mask = probs[:, 0]
        else:
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1)
            # assume class 0 is background
            mask = (pred != 0).float()

        # squeeze to [H, W]
        mask = mask[0]
        return mask
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def __del__(self):
        """Cleanup on destruction"""
        self.cleanup()
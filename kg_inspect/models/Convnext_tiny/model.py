# kg_inspect/models/convnext_tiny/model.py
import torch
import torch.nn as nn
from torchvision import models

class ConvNextTiny(nn.Module):
    """
    Kiến trúc ConvNeXt-Tiny tùy chỉnh, có cấu trúc giống hệt với model
    đã được train trên Kaggle.
    
    Class này chỉ định nghĩa kiến trúc, không chứa logic load file .pth.
    Việc load trọng số sẽ được thực hiện bên ngoài (trong pipeline).
    """
    def __init__(self, num_classes: int):
        super().__init__()
        
        
        base_model = models.convnext_tiny(weights=None)

        self.features = base_model.features

        
        in_features = base_model.classifier[-1].in_features
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.LayerNorm(in_features, eps=1e-6),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Luồng forward đơn giản: qua features rồi qua classifier.
        """
        x = self.features(x)
        x = self.classifier(x)
        return x
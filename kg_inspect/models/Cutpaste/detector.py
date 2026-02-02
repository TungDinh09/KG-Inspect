# kg_inspect/models/Cutpaste/detector.py
from __future__ import annotations
import pickle
from pathlib import Path
from typing import Dict, Any
import paddle
from paddle.vision import transforms as ptransforms
from PIL import Image

from .model import ProjectionNet
from .density import GaussianDensityPaddle


class CutPasteDetector:
    """
    Wrapper dùng nội bộ cho pipeline:
    - KHÔNG đụng gì tới predict.py (vẫn giữ nguyên).
    - Tự load .pdparams + dict_train_embed.pkl theo data_type.
    """

    def __init__(
        self,
        data_type: str = "bottle",
        model_root: str = "kg_inspect/pretrained_models/Cutpaste",
        img_size: int = 256,
        seed: int = 102,
        head_layer: int = 1,
        num_classes: int = 3,
    ):
        self.data_type = data_type
        self.model_root = Path(model_root)
        self.img_size = int(img_size)

        paddle.seed(seed)

        self.model_path = self.model_root / "models" / f"model-{data_type}.pdparams"
        self.eval_dir = self.model_root / "eval" / f"model-{data_type}"
        self.train_embed_pkl = self.eval_dir / "dict_train_embed.pkl"

        # Transforms (giống predict.py)
        self.transform = ptransforms.Compose([
            ptransforms.Resize((self.img_size, self.img_size)),
            ptransforms.ToTensor(),
            ptransforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])

        # Load model
        head_layers = [512] * head_layer + [128]
        self.model = ProjectionNet(pretrained=False, head_layers=head_layers, num_classes=num_classes)
        weights = paddle.load(str(self.model_path))
        self.model.set_state_dict(weights)
        self.model.eval()

        with open(self.train_embed_pkl, "rb") as f:
            info = pickle.load(f)
        self.train_embed = paddle.to_tensor(info["train_embed"])
        self.best_threshold = float(paddle.to_tensor(info["threshold"]).item())

        self.density = GaussianDensityPaddle()
        self.density.fit(self.train_embed)

    @paddle.no_grad()
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Input: PIL.Image
        Output:
          {
            "score": float,
            "is_anomaly": bool,
            "threshold": float
          }
        """
        img = image.convert("RGB").resize((self.img_size, self.img_size))
        ten = self.transform(img)
        ten = paddle.unsqueeze(ten, axis=0)  # (1,3,H,W)
        embed, _ = self.model(ten)
        embed = paddle.nn.functional.normalize(embed, p=2, axis=1)
        distances = self.density.predict(embed)
        score = float(distances[0])
        return {
            "score": score,
            "is_anomaly": score >= self.best_threshold,
            "threshold": self.best_threshold,
        }

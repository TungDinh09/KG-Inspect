# kg_inspect/pipeline/core/vision.py

import os
import json
from typing import Optional, Dict, Any

import torch
from PIL import Image
from torchvision import transforms as T
from rich.console import Console

from kg_inspect.models.Convnext_tiny.model import ConvNextTiny
from kg_inspect.utils.device import get_torch_device
from pathlib import Path
from kg_inspect.models.Cutpaste.detector import CutPasteDetector



console = Console()


class CNNInspect:
    def __init__(self) -> None:
        
        self.device_str: str = os.getenv("DEVICE", "cpu")
        self.device = get_torch_device(self.device_str)
        
        # ConvNeXt config
        
        self.pretrained_root: str = os.getenv(
            "PRETRAIN_MODEL_PATH",
            "kg_inspect/pretrained_models",
        )

        # --------------------------------------------------
        # ConvNeXt-Tiny config
        # --------------------------------------------------
        self.model_path: str = os.path.join(
            self.pretrained_root,
            "Convnext_tiny",
            "convnext_tiny.pth",
        )

        default_labels_txt = "kg_inspect/models/Convnext_tiny/labels.txt"
        self.labels_path: str = os.getenv("LABELS_TXT", default_labels_txt)

        # --------------------------------------------------
        # CutPaste config
        # --------------------------------------------------
        self.id2type: Dict[str, str] = {}
        id2type_env = os.getenv("CUTPASTE_ID2TYPE", "").strip()
        
        
        if id2type_env:
            try:
                self.id2type = json.loads(id2type_env)
            except json.JSONDecodeError:
                console.log(
                    "[yellow][WARN][/yellow] CUTPASTE_ID2TYPE không phải JSON hợp lệ, bỏ qua."
                )

        self.default_cp_type: str = os.getenv("CUTPASTE_DATA_TYPE", "bottle")
        self.cp_model_root: str = os.path.join(
            self.pretrained_root,
            "Cutpaste",
        )
        
        self.cp_img_size: int = int(os.getenv("CUTPASTE_IMG_SIZE", "256"))

        # Runtime state
        self.cnn: Optional[ConvNextTiny] = None
        self._cp_cache: Dict[str, Any] = {}

        self.torch_transform = T.Compose(
            [
                T.Resize((64, 64)),
                T.ToTensor(),
            ]
        )

        # Load model ngay khi init
        self._load_models()

    
    def _load_models(self) -> None:
        """Load ConvNeXt-Tiny weights và labels vào bộ nhớ."""

        p = Path(self.labels_path)
        try:
            labels = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        except FileNotFoundError:
            console.log(f"[yellow][WARN][/yellow] Labels file not found: {p}")
            labels = None
    
        if not labels:
            raise FileNotFoundError(
                f"Không tìm thấy hoặc không đọc được labels.txt tại: {self.labels_path}"
            )
        self.labels = labels
        num_classes = len(labels)

        # Khởi tạo ConvNeXt-Tiny
        console.log(
            f"[INFO] Khởi tạo ConvNextTiny với [bold]{num_classes}[/bold] classes."
        )
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Không tìm thấy file trọng số model tại: {self.model_path}"
            )

        self.cnn = ConvNextTiny(num_classes=num_classes)
        self.cnn.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.cnn.to(self.device)
        self.cnn.eval()

        console.log(
            f"[INFO] ConvNeXt-Tiny đã load xong từ: [green]{self.model_path}[/green] "
            f"trên device [cyan]{self.device}[/cyan]"
        )

    
    async def run(self, image_path: str) -> Dict[str, Any]:
        """
        Chạy full pipeline trên 1 ảnh:
          - ConvNeXt phân loại object
          - CutPaste phát hiện defect tương ứng

        Args:
            image_path: đường dẫn ảnh đầu vào.

        Returns:
            {
                "convnext": { class_id, label, confidence },
                "cutpaste": { ... },
                "meta": {
                    "image_path": str,
                    "device": str,
                    "cutpaste_data_type": str,
                }
            }
        """
        if self.cnn is None:
            raise RuntimeError(
                "Models are not loaded. Gọi _load_models() trước khi chạy."
            )

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")

        img = Image.open(image_path)

        # 1) Classification (helper)
        if self.cnn is None:
            raise RuntimeError(
                "ConvNeXt model chưa được load. Gọi _load_models() trước khi sử dụng."
            )

        x = self.torch_transform(img.convert("RGB")).unsqueeze(0).to(
            self.device
        )
        logits = self.cnn(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

        class_id = pred.item()
        confidence = conf.item()
        labels = self.labels
        label = labels[class_id] if 0 <= class_id < len(labels) else None
        
        cls_out = {"class_id": class_id, "label": label, "confidence": confidence}

        
        label = cls_out.get("label")
        cid = str(cls_out.get("class_id"))
        
        if label:
            cp_type = label.strip().lower().replace(" ", "_").replace("-", "_")
        elif cid in self.id2type:
            cp_type = self.id2type[cid]
        else:
            cp_type = self.default_cp_type
        
        
        if cp_type not in self._cp_cache:
            console.log(
                f"[INFO] Load CutPaste model cho data_type: [cyan]{cp_type}[/cyan]"
            )
            detector = CutPasteDetector(
                data_type=cp_type,
                model_root=self.cp_model_root,
                img_size=self.cp_img_size,
                seed=102,
            )
            self._cp_cache[cp_type] = detector
        else:
            detector = self._cp_cache[cp_type]
        
        cpa_out = detector.predict(img)

        return {
            "convnext": cls_out,
            "cutpaste": cpa_out,
            "meta": {
                "image_path": image_path,
                "device": str(self.device),
                "cutpaste_data_type": cp_type,
            },
        }

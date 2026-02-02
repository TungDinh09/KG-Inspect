import argparse
import os
import time
from typing import Union, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class EfficientAD:
    """
    Wrapper cho EfficientAD:
      - Load model (teacher / student / AE + mean/std)
      - Chạy inference cho 1 ảnh
      - Có hàm tiện ích để tạo heatmap overlay và lưu ra file
    """

    def __init__(
        self,
        checkpoint_path: str,
        object_name: str = "leather",
        device: str = "cuda:0",
        image_size: int = 256,
    ) -> None:
        """
        Args:
            checkpoint_path: thư mục gốc chứa checkpoint cho từng object.
                             Ví dụ: ./output/1/trainings/mvtec_ad
            object_name:     tên object (subfolder trong checkpoint_path), vd: 'leather'
            device:          'cuda:0' hoặc 'cpu'
            image_size:      kích thước resize input trước khi vào model
        """
        self.checkpoint_path = checkpoint_path
        self.object_name = object_name
        self.image_size = image_size

        # Chọn device
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("[WARN] CUDA không khả dụng, fallback về CPU.")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Transform input
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        # Load model & stats
        self._load_models()

    # -----------------------------------------------------
    # Internal: load model
    # -----------------------------------------------------
    def _load_models(self) -> None:
        ckpt_dir = os.path.join(self.checkpoint_path, self.object_name)

        teacher_model_checkpoint = os.path.join(ckpt_dir, "teacher_final.pth")
        student_model_checkpoint = os.path.join(ckpt_dir, "student_final.pth")
        autoencoder_model_checkpoint = os.path.join(ckpt_dir, "autoencoder_final.pth")
        teacher_meanvalue_checkpoint = None #os.path.join(ckpt_dir, "teacher_mean.pth")
        teacher_stdvalue_checkpoint = None #os.path.join(ckpt_dir, "teacher_std.pth")

        print(f"[INFO] Loading checkpoints from: {ckpt_dir}")
        assert os.path.isfile(
            teacher_model_checkpoint
        ), f"Missing teacher checkpoint: {teacher_model_checkpoint}"
        assert os.path.isfile(
            student_model_checkpoint
        ), f"Missing student checkpoint: {student_model_checkpoint}"
        assert os.path.isfile(
            autoencoder_model_checkpoint
        # ), f"Missing AE checkpoint: {autoencoder_model_checkpoint}"
        # assert os.path.isfile(
        #     teacher_meanvalue_checkpoint
        # ), f"Missing mean stats: {teacher_meanvalue_checkpoint}"
        # assert os.path.isfile(
        #     teacher_stdvalue_checkpoint
        ), f"Missing std stats: {teacher_stdvalue_checkpoint}"

        self.teacher_net = torch.load(teacher_model_checkpoint, map_location=self.device)
        self.student_net = torch.load(student_model_checkpoint, map_location=self.device)
        self.ae_net = torch.load(
            autoencoder_model_checkpoint, map_location=self.device
        )

        self.teacher_mean_tensor = torch.load(
            teacher_meanvalue_checkpoint, map_location=self.device
        )
        self.teacher_std_tensor = torch.load(
            teacher_stdvalue_checkpoint, map_location=self.device
        )

        self.teacher_net.eval()
        self.student_net.eval()
        self.ae_net.eval()

        print("[INFO] Model loaded and set to eval mode.")

    # -----------------------------------------------------
    # Internal: preprocess / postprocess helpers
    # -----------------------------------------------------
    def _to_pil(self, image: Union[str, Image.Image, np.ndarray]) -> Image.Image:
        """Chuẩn hoá input về PIL.Image."""
        if isinstance(image, str):
            # đường dẫn
            assert os.path.isfile(image), f"Image not found: {image}"
            pil = Image.open(image).convert("RGB")
            return pil
        elif isinstance(image, Image.Image):
            return image.convert("RGB")
        elif isinstance(image, np.ndarray):
            # BGR (cv2) -> RGB
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(image)
        else:
            raise TypeError("image must be str (path), PIL.Image, hoặc np.ndarray")

    # -----------------------------------------------------
    # Public: inference cho 1 ảnh -> score map
    # -----------------------------------------------------
    @torch.no_grad()
    def infer_map(
        self,
        image: Union[str, Image.Image, np.ndarray],
        out_channels: int = 384,
        q_st_start=None,
        q_st_end=None,
        q_ae_start=None,
        q_ae_end=None,
    ) -> np.ndarray:
        """
        Chạy EfficientAD cho 1 ảnh, trả về anomaly map (resize về size gốc).

        Returns:
            map_combined: np.ndarray [H, W], giá trị (float) anomaly.
        """
        pil_img = self._to_pil(image)
        orig_w, orig_h = pil_img.size

        # Preprocess
        pil_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        # Forward
        teacher_output = self.teacher_net(pil_tensor)
        teacher_output = (teacher_output - self.teacher_mean_tensor) / self.teacher_std_tensor

        student_output = self.student_net(pil_tensor)     # [1, C, H, W]
        autoencoder_output = self.ae_net(pil_tensor)      # [1, C, H, W]

        # MSE map
        map_st = torch.mean(
            (teacher_output - student_output[:, :out_channels]) ** 2,
            dim=1,
            keepdim=True,
        )
        map_ae = torch.mean(
            (autoencoder_output - student_output[:, out_channels:]) ** 2,
            dim=1,
            keepdim=True,
        )

        # Optional quantization
        if q_st_start is not None and q_st_end is not None:
            map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
        if q_ae_start is not None and q_ae_end is not None:
            map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)

        map_combined = 0.5 * map_st + 0.5 * map_ae

        # Upscale về size gốc (pad + interpolate giống code gốc)
        map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
        map_combined = torch.nn.functional.interpolate(
            map_combined, (orig_h, orig_w), mode="bilinear"
        )
        map_combined = map_combined[0, 0].detach().cpu().numpy()

        return map_combined

    # -----------------------------------------------------
    # Public: inference + tạo heatmap overlay, trả về cả map & ảnh overlay
    # -----------------------------------------------------
    def infer_with_overlay(
        self,
        image: Union[str, Image.Image, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Chạy inference + tạo heatmap overlay.

        Returns:
            map_combined: np.ndarray [H, W] (float anomaly map)
            overlay:      np.ndarray [H, W, 3] (uint8 BGR) để lưu bằng cv2
        """
        pil_img = self._to_pil(image)
        map_combined = self.infer_map(pil_img)

        # Normalize & heatmap
        norm_map = cv2.normalize(
            map_combined,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
            cv2.CV_8UC1,
        )
        heatmap = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)

        # Merge với ảnh gốc
        img_np = np.array(pil_img)  # RGB
        out = np.float32(heatmap) / 255 + np.float32(img_np) / 255
        out = out / np.max(out)
        out = np.uint8(out * 255.0)
        out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

        return map_combined, out_bgr

    # -----------------------------------------------------
    # Public: inference cho 1 ảnh từ path + lưu ra file
    # -----------------------------------------------------
    def run_on_image_path(
        self,
        image_path: str,
        output_dir: str,
    ) -> Tuple[str, float]:
        """
        Chạy inference cho 1 ảnh từ path và lưu overlay ra output_dir.

        Returns:
            out_path: đường dẫn file output
            time_cost: thời gian chạy (giây)
        """
        os.makedirs(output_dir, exist_ok=True)

        t0 = time.time()
        _, overlay = self.infer_with_overlay(image_path)
        t1 = time.time()

        out_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(out_path, overlay)

        return out_path, (t1 - t0)


# =========================================================
# CLI (tùy chọn): chạy trực tiếp từ dòng lệnh cho 1 ảnh
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EfficientAD - Single Image via EfficientAD class")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path tới 1 ảnh PNG/JPG",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Thư mục gốc checkpoint, vd: ./output/1/trainings/mvtec_ad",
    )
    parser.add_argument(
        "--object",
        type=str,
        default="leather",
        help="Tên object (folder con trong checkpoint_path)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./output_vis_single",
        help="Folder lưu overlay output",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="cuda:0 hoặc cpu",
    )

    args = parser.parse_args()

    # Khởi tạo class
    model = EfficientAD(
        checkpoint_path=args.checkpoint_path,
        object_name=args.object,
        device=args.device,
    )

    print(f"[INFO] Running EfficientAD on: {args.image}")
    out_path, t_cost = model.run_on_image_path(
        image_path=args.image,
        output_dir=args.output_path,
    )

    print(f"[INFO] Saved overlay to: {out_path}")
    print(f"[INFO] Time cost: {t_cost:.4f} s")

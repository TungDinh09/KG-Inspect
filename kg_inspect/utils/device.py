import torch

def get_torch_device(device_str: str) -> torch.device:
    """Xác định và trả về torch.device dựa trên chuỗi đầu vào và khả năng của hệ thống."""
    if device_str.lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
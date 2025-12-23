"""
設備選擇工具模組
"""
import torch


def get_device():
    """
    獲取可用的計算設備
    
    優先級：MPS (Apple Silicon) > CUDA > CPU
    
    返回:
        device: 設備字串 ("mps", "cuda", 或 "cpu")
    """
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"使用設備: {device}")
    return device


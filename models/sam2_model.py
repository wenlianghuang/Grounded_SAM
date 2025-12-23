"""
SAM2 模型模組
負責 SAM2 模型的載入和分割功能
"""
import torch
import cv2
import numpy as np

# 嘗試導入 SAM2
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False


class SAM2Model:
    """
    SAM2 模型封裝類
    """
    
    def __init__(self, device="cpu"):
        """
        初始化 SAM2 模型
        
        參數:
            device: 計算設備 ("cpu", "cuda", "mps")
        """
        self.device = device
        self.predictor = None
        self._check_availability()
    
    def _check_availability(self):
        """檢查 SAM2 是否可用"""
        if not SAM2_AVAILABLE:
            raise ImportError("SAM2 未安裝，請先安裝: pip install sam-2")
    
    def load_model(self, model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml", 
                   checkpoint_path="/Volumes/T7_SSD/SAM/sam2.1_hiera_large.pt"):
        """
        載入 SAM2.1 模型
        
        參數:
            model_cfg: SAM2 模型配置檔案路徑
            checkpoint_path: SAM2 模型檢查點路徑
        
        返回:
            predictor: SAM2ImagePredictor 實例
        """
        try:
            print(f"載入 SAM2 模型...")
            print(f"  配置檔案: {model_cfg}")
            print(f"  檢查點: {checkpoint_path}")
            
            sam2_model = build_sam2(model_cfg, checkpoint_path, device=self.device)
            self.predictor = SAM2ImagePredictor(sam2_model)
            
            print("SAM2.1-high 模型載入完成！")
            return self.predictor
        except Exception as e:
            print(f"載入 SAM2 模型時出錯: {e}")
            print("提示: 請確保已正確安裝 SAM2 並下載模型權重")
            raise
    
    def segment(self, image, bbox):
        """
        使用 SAM2 對 bounding box 進行分割
        
        參數:
            image: numpy array (BGR 格式)
            bbox: [x1, y1, x2, y2] 格式的邊界框（可以是 list 或 tensor）
        
        返回:
            mask: 分割遮罩 (numpy array, 0-255, 二值化)
        """
        if self.predictor is None:
            raise ValueError("模型尚未載入，請先調用 load_model()")
        
        # 轉換為 RGB 給 SAM2
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 確保圖像是 uint8 格式
        if image_rgb.dtype != np.uint8:
            image_rgb = (image_rgb * 255).astype(np.uint8) if image_rgb.max() <= 1.0 else image_rgb.astype(np.uint8)
        
        # 設定圖像
        self.predictor.set_image(image_rgb)
        
        # 轉換 bbox 為 SAM2 格式 [x1, y1, x2, y2]
        if isinstance(bbox, torch.Tensor):
            x1, y1, x2, y2 = [int(coord.item() if hasattr(coord, 'item') else int(coord)) for coord in bbox]
        elif isinstance(bbox, (list, tuple, np.ndarray)):
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
        else:
            raise ValueError(f"不支援的 bbox 格式: {type(bbox)}")
        
        # 確保座標在有效範圍內
        h, w = image_rgb.shape[:2]
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        # 使用 bounding box 作為 prompt
        try:
            masks, scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=np.array([x1, y1, x2, y2]),
                multimask_output=False,
            )
            
            # 返回第一個 mask（通常是最好的），轉換為二值遮罩
            mask = (masks[0] * 255).astype(np.uint8)
            
        except Exception as e:
            print(f"分割時出錯: {e}")
            # 返回空遮罩
            mask = np.zeros((h, w), dtype=np.uint8)
        
        return mask


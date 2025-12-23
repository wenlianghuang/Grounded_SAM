"""
Grounded DINO 檢測模組
負責使用 Grounded DINO 進行物體檢測
"""
from DINO_Bounding_Box import detect_players


class GroundedDINODetector:
    """
    Grounded DINO 檢測器封裝類
    """
    
    def __init__(self):
        """初始化檢測器"""
        pass
    
    def detect(self, image, text_prompt="basketball.", 
               box_threshold=0.30, text_threshold=0.20,
               include_labels=None):
        """
        檢測圖像中的物體
        
        參數:
            image: 輸入圖像 (numpy array, BGR 格式)
            text_prompt: 文字提示（例如："basketball."）
            box_threshold: 邊界框置信度閾值
            text_threshold: 文字匹配閾值
            include_labels: 要包含的標籤列表（例如：["basketball"]）
        
        返回:
            boxes: 邊界框列表
            scores: 置信度列表
            labels: 標籤列表
        """
        boxes, scores, labels = detect_players(
            image,
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            include_labels=include_labels if include_labels else []
        )
        
        return boxes, scores, labels


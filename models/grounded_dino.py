"""
Grounded DINO 檢測模組
負責使用 Grounded DINO 進行物體檢測
"""
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from utils.device import get_device


class GroundedDINODetector:
    """
    Grounded DINO 檢測器封裝類
    """
    
    def __init__(self, device=None):
        """
        初始化檢測器
        
        參數:
            device: 計算設備（如果為 None，則自動選擇）
        """
        if device is None:
            device = get_device()
        
        self.device = device
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """載入 Grounding DINO 模型"""
        if self.model is None:
            model_id = "IDEA-Research/grounding-dino-base"
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
            self.model.eval()
    
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
            boxes: 邊界框列表 [[x1, y1, x2, y2], ...]
            scores: 置信度列表
            labels: 標籤列表
        """
        # 確保模型已載入
        if self.model is None:
            self._load_model()
        
        # 將 OpenCV BGR 格式轉換為 PIL RGB 格式
        if isinstance(image, np.ndarray):
            # OpenCV 使用 BGR，PIL 使用 RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise ValueError(f"不支援的圖像格式: {type(image)}")
        
        # 預處理
        inputs = self.processor(images=pil_image, text=text_prompt, return_tensors="pt").to(self.device)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 後處理
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            text_threshold=text_threshold,
            target_sizes=[pil_image.size[::-1]]  # [height, width]
        )[0]
        
        # 提取結果
        boxes = results["boxes"]
        scores = results["scores"]
        labels = results["labels"]
        
        # 應用 box_threshold 過濾
        if len(scores) > 0:
            mask = scores >= box_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            labels = [labels[i] for i in range(len(labels)) if mask[i]]
        
        # 如果指定了 include_labels，則只保留匹配的標籤
        if include_labels and len(labels) > 0:
            filtered_boxes = []
            filtered_scores = []
            filtered_labels = []
            for box, score, label in zip(boxes, scores, labels):
                if label.lower() in [l.lower() for l in include_labels]:
                    filtered_boxes.append(box)
                    filtered_scores.append(score)
                    filtered_labels.append(label)
            boxes = filtered_boxes
            scores = filtered_scores
            labels = filtered_labels
        
        # 轉換 boxes 為 numpy 數組格式 [x1, y1, x2, y2]
        if len(boxes) > 0:
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu().numpy()
            boxes = [box.tolist() if isinstance(box, (torch.Tensor, np.ndarray)) else list(box) for box in boxes]
            scores = [score.item() if isinstance(score, torch.Tensor) else float(score) for score in scores]
        
        return boxes, scores, labels


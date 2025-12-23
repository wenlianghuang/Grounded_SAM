"""
足球圖像處理模組
負責足球圖像的載入和替換功能
"""
import cv2
import numpy as np
import math
import torch
from pathlib import Path


class FootballImageProcessor:
    """
    足球圖像處理器
    """
    
    def __init__(self):
        """初始化處理器"""
        self.football_img = None
    
    def load_football_image(self, football_image_path=None):
        """
        載入足球圖像
        
        參數:
            football_image_path: 足球圖像路徑（如果為 None，則創建一個簡單的足球圖像）
        
        返回:
            football_img: 足球圖像 (BGR 格式, numpy array)
        """
        if football_image_path and Path(football_image_path).exists():
            # 從檔案載入
            football_img = cv2.imread(str(football_image_path))
            if football_img is None:
                raise ValueError(f"無法載入足球圖像: {football_image_path}")
            print(f"已載入足球圖像: {football_image_path}")
        else:
            # 創建一個簡單的足球圖像（白色背景，黑色五邊形圖案）
            size = 200
            football_img = np.ones((size, size, 3), dtype=np.uint8) * 255  # 白色背景
            
            # 繪製足球圖案（簡化版：黑色圓形和線條）
            center = (size // 2, size // 2)
            radius = size // 2 - 10
            
            # 繪製外圓
            cv2.circle(football_img, center, radius, (0, 0, 0), 2)
            
            # 繪製五邊形（簡化版）
            points = []
            for i in range(5):
                angle = i * 2 * math.pi / 5 - math.pi / 2
                x = int(center[0] + radius * 0.6 * math.cos(angle))
                y = int(center[1] + radius * 0.6 * math.sin(angle))
                points.append([x, y])
            points = np.array(points, np.int32)
            cv2.polylines(football_img, [points], True, (0, 0, 0), 2)
            
            # 繪製一些線條模擬足球紋理
            for i in range(5):
                angle = i * 2 * math.pi / 5 - math.pi / 2
                x1 = int(center[0] + radius * 0.3 * math.cos(angle))
                y1 = int(center[1] + radius * 0.3 * math.sin(angle))
                x2 = int(center[0] + radius * 0.9 * math.cos(angle))
                y2 = int(center[1] + radius * 0.9 * math.sin(angle))
                cv2.line(football_img, (x1, y1), (x2, y2), (0, 0, 0), 1)
            
            print("已創建預設足球圖像")
        
        self.football_img = football_img
        return football_img
    
    def replace_with_football(self, frame, mask, bbox):
        """
        將遮罩區域替換為足球圖像
        
        參數:
            frame: 原始圖像 (BGR 格式)
            mask: 分割遮罩 (0-255)
            bbox: 邊界框 [x1, y1, x2, y2]，用於確定足球圖像的大小
        
        返回:
            result_frame: 替換後的圖像
        """
        if self.football_img is None:
            raise ValueError("足球圖像尚未載入，請先調用 load_football_image()")
        
        result_frame = frame.copy()
        
        # 創建二值遮罩（大於 128 的像素視為前景）
        binary_mask = mask > 128
        
        if not np.any(binary_mask):
            return result_frame
        
        # 獲取遮罩區域的邊界框
        y_coords, x_coords = np.where(binary_mask)
        if len(y_coords) == 0 or len(x_coords) == 0:
            return result_frame
        
        min_y, max_y = y_coords.min(), y_coords.max()
        min_x, max_x = x_coords.min(), x_coords.max()
        
        # 計算需要替換的區域大小
        mask_height = max_y - min_y + 1
        mask_width = max_x - min_x + 1
        
        # 從 bbox 計算預期大小（用於縮放足球圖像）
        if isinstance(bbox, torch.Tensor):
            x1, y1, x2, y2 = [int(coord.item() if hasattr(coord, 'item') else int(coord)) for coord in bbox]
        else:
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        # 使用較大的尺寸來確保足球圖像能覆蓋整個區域
        target_size = max(bbox_width, bbox_height, mask_width, mask_height)
        
        # 縮放足球圖像到目標大小
        football_resized = cv2.resize(self.football_img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        
        # 計算在遮罩區域內放置足球圖像的位置
        # 將足球圖像的中心對齊到遮罩區域的中心
        mask_center_x = (min_x + max_x) // 2
        mask_center_y = (min_y + max_y) // 2
        
        # 計算足球圖像在原始圖像中的位置
        football_h, football_w = football_resized.shape[:2]
        start_x = mask_center_x - football_w // 2
        start_y = mask_center_y - football_h // 2
        end_x = start_x + football_w
        end_y = start_y + football_h
        
        # 確保座標在有效範圍內
        frame_h, frame_w = frame.shape[:2]
        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(frame_w, end_x)
        end_y = min(frame_h, end_y)
        
        # 調整足球圖像的大小以匹配可用區域
        available_w = end_x - start_x
        available_h = end_y - start_y
        
        if available_w > 0 and available_h > 0:
            football_cropped = cv2.resize(football_resized, (available_w, available_h), interpolation=cv2.INTER_LINEAR)
            
            # 創建遮罩區域的局部遮罩
            local_mask = binary_mask[start_y:end_y, start_x:end_x]
            
            # 只在遮罩區域內替換
            if local_mask.shape == football_cropped.shape[:2]:
                # 使用遮罩進行混合
                for c in range(3):
                    result_frame[start_y:end_y, start_x:end_x, c] = np.where(
                        local_mask,
                        football_cropped[:, :, c],
                        result_frame[start_y:end_y, start_x:end_x, c]
                    )
        
        return result_frame


"""
使用 DINO 檢測籃球 -> SAM2.1 分割籃球 -> 將籃球替換為足球圖像，生成新影片
流程：Basketball Detect -> Segment -> Transfer to Football
"""

import torch
import cv2
import numpy as np
import math
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# 導入 DINO 檢測功能
from DINO_Bounding_Box import detect_players

# 嘗試導入 SAM2
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    print("警告: SAM2 未安裝，請確保已安裝 sam-2")
    SAM2_AVAILABLE = False

# 設備選擇：優先使用 MPS (Apple Silicon)，否則使用 CPU
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"使用設備: {device}")


def load_sam2_model(model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml", 
                    checkpoint_path="/Volumes/T7_SSD/SAM/sam2.1_hiera_large.pt"):
    """
    載入 SAM2.1-high 模型
    
    參數:
        model_cfg: SAM2 模型配置檔案路徑
        checkpoint_path: SAM2 模型檢查點路徑
    """
    if not SAM2_AVAILABLE:
        raise ImportError("SAM2 未安裝，請先安裝: pip install sam-2")
    
    try:
        print(f"載入 SAM2 模型...")
        print(f"  配置檔案: {model_cfg}")
        print(f"  檢查點: {checkpoint_path}")
        
        sam2_model = build_sam2(model_cfg, checkpoint_path, device=device)
        predictor = SAM2ImagePredictor(sam2_model)
        
        print("SAM2.1-high 模型載入完成！")
        return predictor
    except Exception as e:
        print(f"載入 SAM2 模型時出錯: {e}")
        print("提示: 請確保已正確安裝 SAM2 並下載模型權重")
        raise


def segment_with_sam2(predictor, image, bbox):
    """
    使用 SAM2 對 bounding box 進行分割
    
    參數:
        predictor: SAM2ImagePredictor 實例
        image: numpy array (BGR 格式)
        bbox: [x1, y1, x2, y2] 格式的邊界框（可以是 list 或 tensor）
    
    返回:
        mask: 分割遮罩 (numpy array, 0-255, 二值化)
    """
    # 轉換為 RGB 給 SAM2
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 確保圖像是 uint8 格式
    if image_rgb.dtype != np.uint8:
        image_rgb = (image_rgb * 255).astype(np.uint8) if image_rgb.max() <= 1.0 else image_rgb.astype(np.uint8)
    
    # 設定圖像
    predictor.set_image(image_rgb)
    
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
        masks, scores, _ = predictor.predict(
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


def load_football_image(football_image_path=None):
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
    
    return football_img


def replace_with_football(frame, mask, football_img, bbox):
    """
    將遮罩區域替換為足球圖像
    
    參數:
        frame: 原始圖像 (BGR 格式)
        mask: 分割遮罩 (0-255)
        football_img: 足球圖像 (BGR 格式)
        bbox: 邊界框 [x1, y1, x2, y2]，用於確定足球圖像的大小
    
    返回:
        result_frame: 替換後的圖像
    """
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
    football_resized = cv2.resize(football_img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
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


def process_video_basketball_to_football(video_path, output_video_path=None,
                                        football_image_path=None,
                                        box_threshold=0.30, text_threshold=0.20,
                                        model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
                                        checkpoint_path="/Volumes/T7_SSD/SAM/sam2.1_hiera_large.pt",
                                        max_frames=None, frame_skip=1):
    """
    處理影片：檢測籃球 -> SAM2 分割 -> 替換為足球圖像 -> 生成新影片
    
    流程：
    1. 使用 DINO 檢測籃球
    2. 使用 SAM2 對每個籃球進行精確分割
    3. 將分割出的籃球區域替換為足球圖像
    4. 生成新影片
    
    參數:
        video_path: 輸入影片路徑
        output_video_path: 輸出影片路徑（如果為 None，則自動生成）
        football_image_path: 足球圖像路徑（如果為 None，則創建預設足球圖像）
        box_threshold: DINO 檢測的邊界框閾值
        text_threshold: DINO 檢測的文字匹配閾值
        model_cfg: SAM2 模型配置檔案路徑
        checkpoint_path: SAM2 模型檢查點路徑
        max_frames: 最多處理多少幀（None=處理全部）
        frame_skip: 每隔幾幀處理一次（1=每幀都處理）
    
    返回:
        output_video_path: 輸出影片路徑
    """
    # 載入足球圖像
    print("\n" + "="*60)
    print("步驟 1: 載入足球圖像")
    print("="*60)
    football_img = load_football_image(football_image_path)
    
    # 載入 SAM2 模型
    print("\n" + "="*60)
    print("步驟 2: 載入 SAM2.1 模型")
    print("="*60)
    predictor = load_sam2_model(model_cfg, checkpoint_path)
    
    # 開啟輸入影片
    print("\n" + "="*60)
    print("步驟 3: 開啟輸入影片")
    print("="*60)
    print(f"影片路徑: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"無法開啟影片檔案: {video_path}")
    
    # 取得影片資訊
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"影片資訊:")
    print(f"  總幀數: {total_frames}")
    print(f"  FPS: {fps:.2f}")
    print(f"  解析度: {width}x{height}")
    
    # 設定輸出影片路徑
    if output_video_path is None:
        video_name = Path(video_path).stem
        output_dir = Path(video_path).parent / "basketball_to_football_output"
        output_dir.mkdir(exist_ok=True)
        output_video_path = output_dir / f"{video_name}_basketball_to_football.mp4"
    else:
        output_video_path = Path(output_video_path)
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 設定輸出影片寫入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    print(f"\n輸出影片將保存至: {output_video_path}")
    
    # 處理統計
    total_basketballs_detected = 0
    total_basketballs_segmented = 0
    processed_frames = 0
    
    # 創建進度條
    frames_to_process = total_frames if max_frames is None else min(max_frames, total_frames)
    if frame_skip > 1:
        frames_to_process = frames_to_process // frame_skip
    
    print("\n" + "="*60)
    print("步驟 4: 開始處理影片（檢測籃球 -> SAM2 分割 -> 替換為足球）")
    print("="*60)
    pbar = tqdm(total=frames_to_process, desc="處理影片", unit="幀")
    
    try:
        frame_count = 0
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 根據 frame_skip 決定是否處理此幀
            if frame_count % frame_skip == 0:
                # 使用 DINO 檢測籃球
                boxes, scores, labels = detect_players(
                    frame,
                    text_prompt="basketball.",
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    include_labels=["basketball"]
                )
                
                # 處理檢測到的籃球
                result_frame = frame.copy()
                
                if len(boxes) > 0:
                    total_basketballs_detected += len(boxes)
                    
                    # 對每個籃球進行 SAM2 分割並替換為足球圖像
                    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                        try:
                            # 使用 SAM2 進行分割
                            mask = segment_with_sam2(predictor, frame, box)
                            
                            # 將分割區域替換為足球圖像
                            result_frame = replace_with_football(result_frame, mask, football_img, box)
                            
                            total_basketballs_segmented += 1
                            
                        except Exception as e:
                            print(f"\n警告: 處理第 {frame_count} 幀的第 {i+1} 個籃球時出錯: {e}")
                            continue
                
                # 寫入處理後的幀
                video_writer.write(result_frame)
                processed_frames += 1
                
                # 更新進度條
                pbar.set_postfix({
                    '已處理': f"{processed_frames}/{frames_to_process}",
                    '檢測到': total_basketballs_detected,
                    '已替換': total_basketballs_segmented
                })
                pbar.update(1)  # 更新進度條
                
                # 檢查是否達到最大處理幀數
                if max_frames and processed_frames >= max_frames:
                    break
            
            else:
                # 跳過的幀，直接寫入原始幀
                video_writer.write(frame)
            
            frame_count += 1
    
    finally:
        cap.release()
        video_writer.release()
        pbar.close()
    
    # 顯示統計資訊
    print("\n" + "="*60)
    print("處理完成！")
    print("="*60)
    print(f"處理了 {processed_frames} 幀")
    print(f"檢測到 {total_basketballs_detected} 個籃球")
    print(f"成功分割並替換 {total_basketballs_segmented} 個籃球為足球")
    print(f"\n輸出影片: {output_video_path}")
    
    return str(output_video_path)


# --- 主程式 ---
if __name__ == "__main__":
    # 設定輸入影片路徑
    video_path = "/Volumes/T7_SSD/Video_Test/20225_12_23_952.mp4"
    
    # 可選：設定足球圖像路徑（如果為 None，則使用預設足球圖像）
    # football_image_path = "/path/to/football_image.png"
    football_image_path = None
    
    # 執行處理
    output_path = process_video_basketball_to_football(
        video_path=video_path,
        output_video_path=None,  # 自動生成輸出路徑
        football_image_path=football_image_path,  # 足球圖像路徑（None=使用預設）
        box_threshold=0.30,  # DINO 檢測閾值（降低可減少漏檢）
        text_threshold=0.20,  # DINO 文字匹配閾值（降低可減少漏檢）
        model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
        checkpoint_path="/Volumes/T7_SSD/SAM/sam2.1_hiera_large.pt",
        max_frames=None,  # 處理全部幀，或設為數字如 100 來測試
        frame_skip=1  # 每幀都處理（設為 2 可每隔一幀處理，加快速度）
    )
    
    print(f"\n{'='*60}")
    print(f"完成！籃球轉換為足球的影片已生成")
    print(f"{'='*60}")
    print(f"輸出路徑: {output_path}")


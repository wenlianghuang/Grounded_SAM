"""
將 SAM-2.1 分割的 drone 對象插入到下載的背景圖片中
每個背景生成 5 張不同的合成結果（不同位置、大小、旋轉等）
"""
import cv2
import numpy as np
import json
import requests
import random
import math
import time
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageEnhance
import shutil

from models.sam2_model import SAM2Model
from models.grounded_dino import GroundedDINODetector
from utils.device import get_device
import torch


def download_background_images(
    output_dir="backgrounds",
    num_images=50,
    keywords=["sky", "outdoor", "nature", "city", "field", "mountain", "beach", "forest"],
    max_retries=3,
    retry_delay=2
):
    """
    從免費圖片網站下載背景圖片
    
    參數:
        output_dir: 輸出目錄
        num_images: 下載圖片數量
        keywords: 搜索關鍵詞列表
        max_retries: 最大重試次數
        retry_delay: 重試延遲（秒）
    
    返回:
        下載的圖片路徑列表
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n" + "="*60)
    print("下載背景圖片")
    print("="*60)
    
    downloaded_images = []
    
    # 方法 1: 使用 Picsum Photos (更穩定，無需 API key)
    print("使用 Picsum Photos API 下載圖片...")
    
    for i in tqdm(range(num_images), desc="下載背景圖片"):
        # Picsum Photos - 提供隨機圖片，無需 API key
        # 使用不同的圖片 ID 確保多樣性
        image_id = random.randint(1, 1000)
        url = f"https://picsum.photos/1920/1080?random={image_id}"
        
        success = False
        for attempt in range(max_retries):
            try:
                # 添加請求頭，模擬瀏覽器
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                response = requests.get(url, timeout=15, stream=True, headers=headers, allow_redirects=True)
                
                if response.status_code == 200:
                    # 檢查內容類型
                    content_type = response.headers.get('content-type', '')
                    if 'image' in content_type:
                        image_path = output_path / f"background_{i:04d}.jpg"
                        with open(image_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        
                        # 驗證圖片是否有效
                        try:
                            img = cv2.imread(str(image_path))
                            if img is not None and img.size > 0:
                                downloaded_images.append(image_path)
                                success = True
                                break
                            else:
                                image_path.unlink()  # 刪除無效圖片
                        except:
                            if image_path.exists():
                                image_path.unlink()
                else:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))  # 指數退避
                        continue
                    
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    print(f"\n警告: 下載圖片 {i+1} 失敗: {e}")
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    print(f"\n警告: 處理圖片 {i+1} 時出錯: {e}")
        
        if not success:
            # 如果下載失敗，嘗試使用備用方法
            try:
                # 備用方法：使用不同的圖片服務
                backup_url = f"https://picsum.photos/id/{image_id}/1920/1080"
                response = requests.get(backup_url, timeout=15, stream=True, headers=headers)
                if response.status_code == 200:
                    image_path = output_path / f"background_{i:04d}.jpg"
                    with open(image_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    img = cv2.imread(str(image_path))
                    if img is not None and img.size > 0:
                        downloaded_images.append(image_path)
                        success = True
            except:
                pass
        
        # 添加延遲以避免請求過於頻繁
        if i < num_images - 1:  # 最後一張不需要延遲
            time.sleep(0.5)  # 每次請求間隔 0.5 秒
    
    print(f"\n成功下載 {len(downloaded_images)} 張背景圖片")
    
    if len(downloaded_images) < num_images:
        print(f"警告: 只下載了 {len(downloaded_images)}/{num_images} 張圖片")
        print("建議：可以手動添加更多背景圖片到目錄中，或重新運行下載")
    
    return downloaded_images


def load_segmented_drones(segmented_dir, original_image_dir=None, detections_json=None):
    """
    從 prepare_yolo_dataset 的輸出中載入分割好的 drone 對象
    如果找不到，可以從原始數據重新分割
    
    參數:
        segmented_dir: 包含分割對象的目錄（應該是 prepare_yolo_dataset 的輸出）
        original_image_dir: 原始圖像目錄（用於重新分割）
        detections_json: 檢測結果 JSON 文件（用於重新分割）
    
    返回:
        drone_images: 載入的 drone 圖像列表（帶透明通道）
    """
    segmented_path = Path(segmented_dir)
    
    # 查找分割對象目錄（檢查更多可能的路徑）
    possible_dirs = [
        segmented_path / "segmented_objects",
        segmented_path / "segmented_objects" / "train",  # 可能在 train 子目錄
        segmented_path / "segmented_objects" / "val",    # 可能在 val 子目錄
        segmented_path.parent / "segmented_objects",
        segmented_path / "objects",
        Path(segmented_dir) if Path(segmented_dir).is_dir() else None,
    ]
    
    # 過濾 None 值
    possible_dirs = [d for d in possible_dirs if d is not None]
    
    drone_dir = None
    for d in possible_dirs:
        if d.exists() and d.is_dir():
            # 檢查目錄中是否有圖片文件
            files = list(d.glob("*.png")) + list(d.glob("*.jpg"))
            if len(files) > 0:
                drone_dir = d
                print(f"找到分割對象目錄: {drone_dir}")
                break
    
    if drone_dir is None:
        print(f"\n未找到分割對象目錄")
        print(f"已檢查的路徑:")
        for d in possible_dirs:
            print(f"  - {d} ({'存在' if d.exists() else '不存在'})")
        
        # 如果提供了原始數據，嘗試重新分割
        if original_image_dir and detections_json:
            print(f"\n嘗試從原始數據重新分割...")
            return extract_drones_from_original_data(original_image_dir, detections_json)
        else:
            print(f"\n提示: 請先運行 prepare_yolo_dataset.py 生成分割對象")
            print(f"或者提供 original_image_dir 和 detections_json 參數以重新分割")
            return None
    
    # 載入所有分割的 drone 圖像
    drone_files = list(drone_dir.glob("*.png")) + list(drone_dir.glob("*.jpg"))
    
    if len(drone_files) == 0:
        print(f"在 {drone_dir} 中未找到分割的 drone 圖像")
        return None
    
    print(f"\n找到 {len(drone_files)} 個分割的 drone 對象")
    
    drone_images = []
    for drone_file in drone_files:
        # 載入圖像（保持透明通道）
        img = cv2.imread(str(drone_file), cv2.IMREAD_UNCHANGED)
        if img is not None:
            # 如果是 3 通道，添加 alpha 通道
            if len(img.shape) == 3 and img.shape[2] == 3:
                # 創建 alpha 通道（基於非黑色像素）
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, alpha = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                img[:, :, 3] = alpha
            elif len(img.shape) == 2:
                # 如果是灰度圖，轉換為 BGRA
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
            drone_images.append(img)
    
    print(f"成功載入 {len(drone_images)} 個 drone 對象")
    return drone_images


def extract_drones_from_original_data(original_image_dir, detections_json, sam2_model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml", sam2_checkpoint="/Volumes/T7_SSD/SAM/sam2.1_hiera_large.pt"):
    """
    從原始圖像和檢測結果中提取 drone 對象
    
    參數:
        original_image_dir: 原始圖像目錄
        detections_json: 檢測結果 JSON 文件
        sam2_model_cfg: SAM-2.1 模型配置
        sam2_checkpoint: SAM-2.1 檢查點路徑
    
    返回:
        drone_images: 提取的 drone 圖像列表（帶透明通道）
    """
    print("\n" + "="*60)
    print("從原始數據提取 Drone 對象")
    print("="*60)
    
    image_path = Path(original_image_dir)
    detections_json_path = Path(detections_json)
    
    if not image_path.exists():
        print(f"錯誤: 圖像目錄不存在: {image_path}")
        return None
    
    if not detections_json_path.exists():
        print(f"錯誤: 檢測結果文件不存在: {detections_json_path}")
        return None
    
    # 載入檢測結果
    with open(detections_json_path, 'r', encoding='utf-8') as f:
        detection_list = json.load(f)
    
    print(f"載入了 {len(detection_list)} 個檢測結果")
    
    # 初始化 SAM-2.1
    device = get_device()
    if torch.backends.mps.is_available():
        device = "mps"
    
    print("載入 SAM-2.1 模型...")
    sam2_model = SAM2Model(device=device)
    sam2_model.load_model(sam2_model_cfg, sam2_checkpoint)
    
    drone_images = []
    
    # 處理每個檢測結果
    for det in tqdm(detection_list[:50], desc="提取 Drone 對象"):  # 限制處理前 50 個
        image_file = image_path / det['image_file']
        if not image_file.exists():
            continue
        
        image = cv2.imread(str(image_file))
        if image is None:
            continue
        
        # 處理每個 bounding box
        for box in det['boxes']:
            try:
                # 使用 SAM-2.1 分割
                mask = sam2_model.segment(image, box)
                
                # 提取對象
                extracted = extract_drone_from_image(image, mask)
                
                # 裁剪到 bounding box 區域
                x1, y1, x2, y2 = [int(coord) for coord in box]
                x1 = max(0, min(x1, image.shape[1]))
                y1 = max(0, min(y1, image.shape[0]))
                x2 = max(0, min(x2, image.shape[1]))
                y2 = max(0, min(y2, image.shape[0]))
                
                if x2 > x1 and y2 > y1:
                    cropped = extracted[y1:y2, x1:x2]
                    if cropped.size > 0:
                        drone_images.append(cropped)
            except Exception as e:
                continue
    
    print(f"\n成功提取 {len(drone_images)} 個 drone 對象")
    return drone_images


def extract_drone_from_image(image, mask):
    """
    從圖像中提取 drone 對象（使用遮罩）
    
    參數:
        image: 原始圖像 (BGR)
        mask: 分割遮罩 (0-255)
    
    返回:
        extracted: 提取的 drone 圖像（帶透明通道，BGRA）
    """
    # 創建 4 通道圖像
    extracted = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    
    # 將遮罩轉換為 alpha 通道
    mask_binary = mask > 128
    extracted[:, :, 3] = np.where(mask_binary, 255, 0).astype(np.uint8)
    
    # 將背景設為透明
    extracted[~mask_binary] = [0, 0, 0, 0]
    
    return extracted


def resize_drone(drone_img, min_scale=0.3, max_scale=1.5):
    """
    隨機縮放 drone 圖像
    
    參數:
        drone_img: drone 圖像（帶透明通道）
        min_scale: 最小縮放比例
        max_scale: 最大縮放比例
    
    返回:
        縮放後的圖像
    """
    scale = random.uniform(min_scale, max_scale)
    h, w = drone_img.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    
    if new_h < 10 or new_w < 10:
        new_h, new_w = max(10, new_h), max(10, new_w)
    
    resized = cv2.resize(drone_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized


def rotate_drone(drone_img, max_angle=30):
    """
    隨機旋轉 drone 圖像
    
    參數:
        drone_img: drone 圖像（帶透明通道）
        max_angle: 最大旋轉角度（度）
    
    返回:
        旋轉後的圖像
    """
    angle = random.uniform(-max_angle, max_angle)
    
    if abs(angle) < 0.1:
        return drone_img
    
    h, w = drone_img.shape[:2]
    center = (w // 2, h // 2)
    
    # 獲取旋轉矩陣
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 計算新的邊界框
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # 調整旋轉矩陣以包含整個圖像
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    # 旋轉圖像
    rotated = cv2.warpAffine(drone_img, M, (new_w, new_h), 
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_TRANSPARENT)
    
    return rotated


def place_drone_on_background(background, drone_img, position=None):
    """
    將 drone 放置到背景圖像上
    
    參數:
        background: 背景圖像 (BGR)
        drone_img: drone 圖像（帶透明通道，BGRA）
        position: 放置位置 (x, y)，如果為 None 則隨機
    
    返回:
        result: 合成後的圖像
        bbox: bounding box [x1, y1, x2, y2]
    """
    bg_h, bg_w = background.shape[:2]
    drone_h, drone_w = drone_img.shape[:2]
    
    # 確保 drone 不會超出背景
    max_x = max(0, bg_w - drone_w)
    max_y = max(0, bg_h - drone_h)
    
    if position is None:
        x = random.randint(0, max_x) if max_x > 0 else 0
        y = random.randint(0, max_y) if max_y > 0 else 0
    else:
        x, y = position
        x = max(0, min(x, max_x))
        y = max(0, min(y, max_y))
    
    # 創建結果圖像
    result = background.copy()
    
    # 提取 alpha 通道
    alpha = drone_img[:, :, 3] / 255.0
    
    # 計算放置區域
    end_x = min(x + drone_w, bg_w)
    end_y = min(y + drone_h, bg_h)
    
    # 裁剪 drone 圖像以適應邊界
    drone_crop_w = end_x - x
    drone_crop_h = end_y - y
    drone_cropped = drone_img[:drone_crop_h, :drone_crop_w]
    alpha_cropped = alpha[:drone_crop_h, :drone_crop_w]
    
    # Alpha 混合
    for c in range(3):
        result[y:end_y, x:end_x, c] = (
            alpha_cropped * drone_cropped[:, :, c] +
            (1 - alpha_cropped) * result[y:end_y, x:end_x, c]
        )
    
    # 計算 bounding box
    bbox = [x, y, end_x, end_y]
    
    return result, bbox


def add_shadow_effect(result, bbox, shadow_intensity=0.3):
    """
    添加陰影效果（可選）
    
    參數:
        result: 合成後的圖像
        bbox: bounding box
        shadow_intensity: 陰影強度
    """
    if random.random() > 0.5:  # 50% 概率添加陰影
        x1, y1, x2, y2 = bbox
        shadow_offset = 5
        
        # 創建陰影遮罩
        shadow_mask = np.zeros((result.shape[0], result.shape[1]), dtype=np.uint8)
        cv2.ellipse(shadow_mask, 
                  ((x1 + x2) // 2 + shadow_offset, y2 + shadow_offset),
                  ((x2 - x1) // 2, 10), 0, 0, 360, 255, -1)
        
        # 應用陰影
        shadow_mask = shadow_mask.astype(float) / 255.0 * shadow_intensity
        for c in range(3):
            result[:, :, c] = np.clip(
                result[:, :, c] * (1 - shadow_mask),
                0, 255
            ).astype(np.uint8)
    
    return result


def convert_bbox_to_yolo_format(bbox, img_width, img_height):
    """將 bbox 轉換為 YOLO 格式"""
    x1, y1, x2, y2 = bbox
    
    center_x = (x1 + x2) / 2.0 / img_width
    center_y = (y1 + y2) / 2.0 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    center_x = max(0.0, min(1.0, center_x))
    center_y = max(0.0, min(1.0, center_y))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))
    
    return [center_x, center_y, width, height]


def insert_segmented_drones(
    segmented_drone_dir=None,
    original_image_dir=None,
    detections_json=None,
    background_dir="backgrounds",
    output_dir="synthetic_drone_dataset",
    num_per_background=5,
    class_name="drone",
    download_backgrounds=True,
    num_backgrounds=50
):
    """
    將分割的 drone 插入到背景圖片中，生成合成數據集
    
    參數:
        segmented_drone_dir: 分割的 drone 對象目錄（從 prepare_yolo_dataset 輸出）
        original_image_dir: 原始圖像目錄（如果找不到分割對象，用於重新分割）
        detections_json: 檢測結果 JSON 文件（如果找不到分割對象，用於重新分割）
        background_dir: 背景圖片目錄
        output_dir: 輸出數據集目錄
        num_per_background: 每個背景生成多少張合成圖像
        class_name: 類別名稱
        download_backgrounds: 是否下載背景圖片
        num_backgrounds: 下載的背景圖片數量
    
    返回:
        輸出數據集目錄路徑
    """
    output_path = Path(output_dir)
    
    # 創建輸出目錄結構
    images_dir = output_path / "images"
    labels_dir = output_path / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("生成合成 Drone 數據集")
    print("="*60)
    
    # 步驟 1: 下載或載入背景圖片
    if download_backgrounds:
        background_images = download_background_images(
            output_dir=background_dir,
            num_images=num_backgrounds
        )
    else:
        background_path = Path(background_dir)
        background_images = list(background_path.glob("*.jpg")) + list(background_path.glob("*.png"))
        print(f"\n從 {background_dir} 載入 {len(background_images)} 張背景圖片")
    
    if len(background_images) == 0:
        raise ValueError("沒有找到背景圖片")
    
    # 步驟 2: 載入分割的 drone 對象
    drone_images = []
    
    if segmented_drone_dir:
        drone_images = load_segmented_drones(
            segmented_drone_dir,
            original_image_dir=original_image_dir,
            detections_json=detections_json
        )
    
    # 如果沒有找到分割對象，嘗試從原始數據重新分割
    if drone_images is None or len(drone_images) == 0:
        if original_image_dir and detections_json:
            print("\n嘗試從原始數據重新分割...")
            drone_images = extract_drones_from_original_data(original_image_dir, detections_json)
        
        if drone_images is None or len(drone_images) == 0:
            raise ValueError(
                "無法載入 drone 對象！\n"
                "請確保：\n"
                "1. 已運行 prepare_yolo_dataset.py 生成分割對象，或\n"
                "2. 提供 original_image_dir 和 detections_json 參數以重新分割"
            )
    
    print(f"\n使用 {len(drone_images)} 個 drone 對象進行合成")
    
    # 步驟 3: 生成合成圖像
    total_generated = 0
    class_id = 0
    
    print("\n" + "="*60)
    print("開始生成合成圖像")
    print("="*60)
    
    pbar = tqdm(total=len(background_images) * num_per_background, desc="生成合成圖像", unit="張")
    
    for bg_idx, bg_path in enumerate(background_images):
        background = cv2.imread(str(bg_path))
        if background is None:
            continue
        
        bg_h, bg_w = background.shape[:2]
        
        # 每個背景生成 num_per_background 張合成圖像
        for synth_idx in range(num_per_background):
            # 隨機選擇一個 drone
            drone_img = random.choice(drone_images).copy()
            
            # 隨機變換
            # 1. 縮放
            scale = random.uniform(0.2, 1.0)
            drone_img = resize_drone(drone_img, min_scale=scale, max_scale=scale)
            
            # 2. 旋轉
            if random.random() > 0.3:  # 70% 概率旋轉
                drone_img = rotate_drone(drone_img, max_angle=45)
            
            # 3. 亮度調整（可選）
            if random.random() > 0.5:
                brightness = random.uniform(0.8, 1.2)
                drone_img[:, :, :3] = np.clip(drone_img[:, :, :3] * brightness, 0, 255).astype(np.uint8)
            
            # 4. 放置到背景上
            result, bbox = place_drone_on_background(background.copy(), drone_img)
            
            # 5. 添加陰影（可選）
            if random.random() > 0.5:
                result = add_shadow_effect(result, bbox)
            
            # 保存合成圖像
            image_filename = f"synth_{bg_idx:04d}_{synth_idx:02d}.jpg"
            image_path = images_dir / image_filename
            cv2.imwrite(str(image_path), result)
            
            # 生成 YOLO 格式標籤
            yolo_bbox = convert_bbox_to_yolo_format(bbox, bg_w, bg_h)
            label_path = labels_dir / f"{Path(image_filename).stem}.txt"
            
            with open(label_path, 'w') as f:
                f.write(f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")
            
            total_generated += 1
            pbar.update(1)
    
    pbar.close()
    
    # 創建數據集配置文件
    dataset_yaml = f"""# YOLO 合成數據集配置文件
path: {output_path.absolute()}
train: images
val: images

# 類別
names:
  0: {class_name}
"""
    
    with open(output_path / "dataset.yaml", 'w') as f:
        f.write(dataset_yaml.strip())
    
    # 保存類別文件
    with open(output_path / "classes.txt", 'w') as f:
        f.write(class_name)
    
    print(f"\n" + "="*60)
    print("合成數據集生成完成！")
    print("="*60)
    print(f"生成圖像: {total_generated} 張")
    print(f"使用背景: {len(background_images)} 張")
    print(f"輸出目錄: {output_path}")
    print(f"配置文件: {output_path / 'dataset.yaml'}")
    
    return str(output_path)


def main():
    """主函數"""
    # 配置參數
    segmented_drone_dir = "/Volumes/T7_SSD/Object_Image/Drone_YOLO_Dataset"  # prepare_yolo_dataset 的輸出目錄
    original_image_dir = "/Volumes/T7_SSD/Object_Image/Drone"  # 原始圖像目錄（備用）
    detections_json = "/Volumes/T7_SSD/Object_Image/Drone/detections.json"  # 檢測結果（備用）
    background_dir = "backgrounds"
    output_dir = "synthetic_drone_dataset"
    
    output_path = insert_segmented_drones(
        segmented_drone_dir=segmented_drone_dir,
        original_image_dir=original_image_dir,  # 如果找不到分割對象，使用原始數據重新分割
        detections_json=detections_json,  # 如果找不到分割對象，使用檢測結果重新分割
        background_dir=background_dir,
        output_dir=output_dir,
        num_per_background=5,  # 每個背景生成 5 張
        class_name="drone",
        download_backgrounds=True,
        num_backgrounds=50  # 下載 50 張背景圖片
    )
    
    print(f"\n合成數據集已生成: {output_path}")


if __name__ == "__main__":
    main()


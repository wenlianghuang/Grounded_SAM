"""
準備 YOLO 訓練數據集
流程：讀取圖像和檢測結果 -> SAM-2.1 精確分割 -> 轉換為 YOLO 格式
此腳本在本地運行，生成數據集後可上傳到 Colab 進行訓練
"""
import cv2
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import shutil

from models.sam2_model import SAM2Model
from models.grounded_dino import GroundedDINODetector
from utils.device import get_device


def convert_bbox_to_yolo_format(bbox, img_width, img_height):
    """
    將 [x1, y1, x2, y2] 格式轉換為 YOLO 格式 [center_x, center_y, width, height] (歸一化)
    
    參數:
        bbox: [x1, y1, x2, y2]
        img_width: 圖像寬度
        img_height: 圖像高度
    
    返回:
        [center_x, center_y, width, height] (歸一化到 0-1)
    """
    x1, y1, x2, y2 = bbox
    
    # 計算中心點和寬高
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1
    
    # 歸一化
    center_x /= img_width
    center_y /= img_height
    width /= img_width
    height /= img_height
    
    # 確保值在有效範圍內
    center_x = max(0.0, min(1.0, center_x))
    center_y = max(0.0, min(1.0, center_y))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))
    
    return [center_x, center_y, width, height]


def refine_bbox_with_mask(bbox, mask):
    """
    使用 SAM-2.1 分割遮罩來精確調整 bounding box
    
    參數:
        bbox: 原始 bounding box [x1, y1, x2, y2]
        mask: 分割遮罩 (0-255)
    
    返回:
        精確的 bounding box [x1, y1, x2, y2]
    """
    # 找到遮罩區域
    mask_binary = mask > 128
    y_coords, x_coords = np.where(mask_binary)
    
    if len(y_coords) == 0 or len(x_coords) == 0:
        return bbox  # 如果沒有遮罩，返回原始 bbox
    
    # 計算精確的邊界
    min_x = max(0, int(x_coords.min()))
    min_y = max(0, int(y_coords.min()))
    max_x = min(mask.shape[1], int(x_coords.max()))
    max_y = min(mask.shape[0], int(y_coords.max()))
    
    return [min_x, min_y, max_x, max_y]


def prepare_yolo_dataset(
    image_dir="/Volumes/T7_SSD/Object_Image/Drone",
    detections_json=None,
    output_dir="/Volumes/T7_SSD/Object_Image/Drone_YOLO_Dataset",
    class_name="drone",
    sam2_model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
    sam2_checkpoint="/Volumes/T7_SSD/SAM/sam2.1_hiera_large.pt",
    use_sam_refinement=True,
    train_ratio=0.8
):
    """
    準備 YOLO 格式的訓練數據集
    
    參數:
        image_dir: 圖像目錄
        detections_json: 檢測結果 JSON 文件路徑（如果為 None，則重新檢測）
        output_dir: 輸出數據集目錄
        class_name: 類別名稱
        sam2_model_cfg: SAM-2.1 模型配置
        sam2_checkpoint: SAM-2.1 檢查點路徑
        use_sam_refinement: 是否使用 SAM-2.1 精確調整 bbox
        train_ratio: 訓練集比例（剩餘為驗證集）
    
    返回:
        數據集目錄路徑
    """
    image_path = Path(image_dir)
    output_path = Path(output_dir)
    
    # 創建 YOLO 數據集目錄結構
    yolo_images_dir = output_path / "images"
    yolo_labels_dir = output_path / "labels"
    yolo_images_dir.mkdir(parents=True, exist_ok=True)
    yolo_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # 創建訓練/驗證分割
    train_images_dir = yolo_images_dir / "train"
    train_labels_dir = yolo_labels_dir / "train"
    val_images_dir = yolo_images_dir / "val"
    val_labels_dir = yolo_labels_dir / "val"
    
    # 創建分割對象目錄（用於 insert_segment_drone.py）
    segmented_objects_dir = output_path / "segmented_objects"
    segmented_objects_dir.mkdir(parents=True, exist_ok=True)
    
    for d in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # 初始化設備和模型
    device = get_device()
    if torch.backends.mps.is_available():
        device = "mps"
        print(f"✓ 使用 MPS GPU 加速")
    
    # 載入 SAM-2.1（如果需要精確調整）
    sam2_model = None
    if use_sam_refinement:
        print("\n" + "="*60)
        print("載入 SAM-2.1 模型進行精確分割")
        print("="*60)
        sam2_model = SAM2Model(device=device)
        sam2_model.load_model(sam2_model_cfg, sam2_checkpoint)
    
    # 載入檢測結果
    dino_detector = None
    detections = {}
    
    if detections_json and Path(detections_json).exists():
        print(f"\n從 {detections_json} 載入檢測結果")
        with open(detections_json, 'r', encoding='utf-8') as f:
            detection_list = json.load(f)
        for det in detection_list:
            frame_num = int(Path(det['image_file']).stem.split('_')[1])
            detections[frame_num] = det
        print(f"載入了 {len(detections)} 個檢測結果")
    else:
        print("\n重新進行 DINO 檢測...")
        dino_detector = GroundedDINODetector(device=device)
    
    # 獲取所有圖像文件
    image_files = sorted(image_path.glob("frame_*.jpg"))
    print(f"\n找到 {len(image_files)} 張圖像")
    
    if len(image_files) == 0:
        raise ValueError(f"在 {image_path} 中沒有找到圖像文件")
    
    # 類別映射（YOLO 格式需要類別 ID）
    class_id = 0  # 單一類別，ID 為 0
    class_names = [class_name]
    
    # 保存類別名稱文件
    with open(output_path / "classes.txt", 'w') as f:
        f.write('\n'.join(class_names))
    print(f"類別: {class_names}")
    
    print("\n" + "="*60)
    print("開始準備 YOLO 數據集")
    print("="*60)
    
    total_annotations = 0
    train_count = 0
    val_count = 0
    skipped_count = 0
    
    pbar = tqdm(image_files, desc="處理圖像", unit="張")
    
    for idx, img_file in enumerate(pbar):
        frame_num = int(img_file.stem.split('_')[1])
        image = cv2.imread(str(img_file))
        
        if image is None:
            skipped_count += 1
            continue
        
        img_height, img_width = image.shape[:2]
        
        # 獲取檢測結果
        if frame_num in detections:
            boxes = detections[frame_num]['boxes']
            scores = detections[frame_num]['scores']
            labels = detections[frame_num]['labels']
        elif dino_detector:
            boxes, scores, labels = dino_detector.detect(
                image,
                text_prompt="Drone",
                box_threshold=0.30,
                text_threshold=0.20
            )
        else:
            skipped_count += 1
            continue
        
        if len(boxes) == 0:
            skipped_count += 1
            continue
        
        # 決定是訓練集還是驗證集
        is_train = (idx % (1 / (1 - train_ratio)) >= 1) if train_ratio < 1.0 else True
        # 簡化：每 10 張中 8 張訓練，2 張驗證
        is_train = (idx % 10 < 8)
        
        if is_train:
            images_dst = train_images_dir
            labels_dst = train_labels_dir
            train_count += 1
        else:
            images_dst = val_images_dir
            labels_dst = val_labels_dir
            val_count += 1
        
        # 複製圖像
        dst_image_path = images_dst / img_file.name
        shutil.copy2(img_file, dst_image_path)
        
        # 準備標籤文件
        label_lines = []
        
        for obj_idx, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            try:
                mask = None
                # 使用 SAM-2.1 精確調整 bbox（可選）
                if use_sam_refinement and sam2_model:
                    mask = sam2_model.segment(image, box)
                    refined_bbox = refine_bbox_with_mask(box, mask)
                else:
                    refined_bbox = box
                
                # 保存分割的對象圖像（帶透明通道）
                if mask is not None:
                    # 提取對象
                    mask_binary = mask > 128
                    extracted_obj = image.copy()
                    extracted_obj[~mask_binary] = [0, 0, 0]  # 背景設為黑色
                    
                    # 裁剪到 bounding box 區域
                    x1, y1, x2, y2 = [int(coord) for coord in refined_bbox]
                    x1 = max(0, min(x1, img_width))
                    y1 = max(0, min(y1, img_height))
                    x2 = max(0, min(x2, img_width))
                    y2 = max(0, min(y2, img_height))
                    
                    cropped_obj = extracted_obj[y1:y2, x1:x2]
                    cropped_mask = mask[y1:y2, x1:x2]
                    
                    # 創建帶透明通道的圖像
                    if cropped_obj.size > 0 and cropped_mask.size > 0:
                        obj_bgra = cv2.cvtColor(cropped_obj, cv2.COLOR_BGR2BGRA)
                        obj_bgra[:, :, 3] = (cropped_mask > 128).astype(np.uint8) * 255
                        
                        # 保存分割對象
                        obj_filename = segmented_objects_dir / f"{img_file.stem}_obj{obj_idx:02d}.png"
                        cv2.imwrite(str(obj_filename), obj_bgra)
                
                # 轉換為 YOLO 格式
                yolo_bbox = convert_bbox_to_yolo_format(
                    refined_bbox, img_width, img_height
                )
                
                # YOLO 格式：class_id center_x center_y width height
                label_line = f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n"
                label_lines.append(label_line)
                total_annotations += 1
                
            except Exception as e:
                print(f"\n警告: 處理 {img_file.name} 的對象時出錯: {e}")
                continue
        
        # 保存標籤文件
        if label_lines:
            label_file = labels_dst / f"{img_file.stem}.txt"
            with open(label_file, 'w') as f:
                f.writelines(label_lines)
        
        pbar.set_postfix({
            '訓練': train_count,
            '驗證': val_count,
            '標註': total_annotations,
            '跳過': skipped_count
        })
    
    # 創建數據集配置文件（YOLO 格式）
    dataset_yaml = f"""# YOLO 數據集配置文件
# 路徑相對於此文件，或使用絕對路徑
path: {output_path.absolute()}
train: images/train
val: images/val

# 類別
names:
  0: {class_name}
"""
    
    with open(output_path / "dataset.yaml", 'w') as f:
        f.write(dataset_yaml.strip())
    
    # 創建 README 說明文件
    readme_content = f"""# YOLO 訓練數據集

## 數據集信息
- 類別: {class_name}
- 訓練圖像: {train_count} 張
- 驗證圖像: {val_count} 張
- 總標註數: {total_annotations} 個
- 跳過圖像: {skipped_count} 張

## 目錄結構
```
{output_path.name}/
├── images/
│   ├── train/  # 訓練圖像
│   └── val/    # 驗證圖像
├── labels/
│   ├── train/  # 訓練標籤（YOLO 格式）
│   └── val/    # 驗證標籤
├── segmented_objects/  # 分割的對象圖像（用於合成數據集）
├── dataset.yaml  # 數據集配置文件
└── classes.txt   # 類別名稱
```

## 使用說明
1. 將整個數據集目錄上傳到 Google Colab
2. 使用 train_yolo_colab.py 進行訓練
3. 或使用 ultralytics YOLO 直接訓練：
   ```python
   from ultralytics import YOLO
   model = YOLO('yolov8n.pt')
   model.train(data='dataset.yaml', epochs=100)
   ```
"""
    
    with open(output_path / "README.md", 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"\n" + "="*60)
    print("數據集準備完成！")
    print("="*60)
    print(f"訓練圖像: {train_count} 張")
    print(f"驗證圖像: {val_count} 張")
    print(f"總標註數: {total_annotations} 個")
    print(f"跳過圖像: {skipped_count} 張")
    print(f"數據集目錄: {output_path}")
    print(f"配置文件: {output_path / 'dataset.yaml'}")
    print(f"\n下一步：將數據集上傳到 Colab 並使用 train_yolo_colab.py 進行訓練")
    
    return str(output_path)


def main():
    """主函數"""
    # 準備 YOLO 數據集
    dataset_dir = prepare_yolo_dataset(
        image_dir="/Volumes/T7_SSD/Object_Image/Drone",
        detections_json="/Volumes/T7_SSD/Object_Image/Drone/detections.json",
        output_dir="/Volumes/T7_SSD/Object_Image/Drone_YOLO_Dataset",
        class_name="drone",
        use_sam_refinement=True,  # 使用 SAM-2.1 精確調整
        train_ratio=0.8
    )
    
    print(f"\n數據集已準備完成，目錄: {dataset_dir}")


if __name__ == "__main__":
    main()


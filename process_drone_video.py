"""
處理 Drone.mp4 影片
使用 DINO 檢測 Drone，繪製 bounding box，並將每一幀保存為 JPEG
"""
import cv2
import numpy as np
import torch
import json
from pathlib import Path
from tqdm import tqdm

from models.grounded_dino import GroundedDINODetector
from utils.device import get_device


def process_drone_video(
    video_path="Drone.mp4",
    output_dir="/Volumes/T7_SSD/Object_Image/Drone",
    box_threshold=0.30,
    text_threshold=0.20,
    frame_skip=1
):
    """
    處理 Drone 影片：檢測 -> 繪製 bounding box -> 保存為 JPEG
    
    參數:
        video_path: 輸入影片路徑
        output_dir: 輸出目錄路徑
        box_threshold: DINO 檢測的邊界框閾值
        text_threshold: DINO 檢測的文字匹配閾值
        frame_skip: 每隔幾幀處理一次（1=每幀都處理）
    """
    # 創建輸出目錄
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"輸出目錄: {output_path}")
    
    # 初始化 DINO 檢測器
    print("\n" + "="*60)
    print("載入 DINO 模型")
    print("="*60)
    device = get_device()
    
    # 確認使用 MPS（如果可用）
    if torch.backends.mps.is_available():
        if device != "mps":
            print(f"警告: MPS 可用但當前使用 {device}，切換到 MPS")
            device = "mps"
        print(f"✓ 使用 MPS (Metal Performance Shaders) 進行 GPU 加速")
    elif device == "mps":
        print(f"警告: MPS 不可用，回退到 {device}")
    else:
        print(f"使用設備: {device}")
    
    dino_detector = GroundedDINODetector(device=device)
    
    # 開啟輸入影片
    print("\n" + "="*60)
    print("開啟輸入影片")
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
    
    # 處理統計
    total_drones_detected = 0
    saved_frames = 0  # 保存的幀數（所有幀）
    detected_frames = 0  # 進行檢測的幀數
    detection_results = []  # 存儲所有檢測結果
    
    print("\n" + "="*60)
    print("開始處理影片（檢測 Drone -> 繪製 bounding box -> 保存 JPEG）")
    print("="*60)
    print(f"將保存所有 {total_frames} 幀為 JPEG")
    if frame_skip > 1:
        print(f"每 {frame_skip} 幀進行一次檢測，其他幀保存原始圖像")
    pbar = tqdm(total=total_frames, desc="處理影片", unit="幀")
    
    try:
        frame_count = 0
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 根據 frame_skip 決定是否進行檢測
            should_detect = (frame_count % frame_skip == 0)
            
            if should_detect:
                # 使用 DINO 檢測 Drone
                boxes, scores, labels = dino_detector.detect(
                    frame,
                    text_prompt="Drone",
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    include_labels=None  # 不限制標籤，因為可能檢測到 "Drone" 或其他相關標籤
                )
                
                # 繪製 bounding box
                result_frame = frame.copy()
                
                if len(boxes) > 0:
                    total_drones_detected += len(boxes)
                    
                    # 保存檢測結果
                    detection_results.append({
                        'frame': frame_count,
                        'image_file': f"frame_{frame_count:06d}.jpg",
                        'boxes': [[float(coord) for coord in box] for box in boxes],
                        'scores': [float(score) for score in scores],
                        'labels': labels
                    })
                    
                    # 對每個檢測到的 Drone 繪製 bounding box
                    for box, score, label in zip(boxes, scores, labels):
                        # 轉換 box 座標為整數
                        x1, y1, x2, y2 = [int(coord) for coord in box]
                        
                        # 繪製邊界框（綠色，線寬 2）
                        cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # 繪製標籤和置信度
                        label_text = f"{label}: {score:.2f}"
                        # 計算文字大小
                        (text_width, text_height), baseline = cv2.getTextSize(
                            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                        )
                        # 繪製文字背景
                        cv2.rectangle(
                            result_frame,
                            (x1, y1 - text_height - baseline - 5),
                            (x1 + text_width, y1),
                            (0, 255, 0),
                            -1
                        )
                        # 繪製文字
                        cv2.putText(
                            result_frame,
                            label_text,
                            (x1, y1 - baseline - 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 0),
                            1
                        )
                
                detected_frames += 1
                frame_to_save = result_frame
            else:
                # 跳過檢測的幀，直接保存原始幀
                frame_to_save = frame
            
            # 保存所有幀為 JPEG（無論是否進行檢測）
            frame_filename = output_path / f"frame_{frame_count:06d}.jpg"
            cv2.imwrite(str(frame_filename), frame_to_save, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_frames += 1
            
            # 更新進度條
            pbar.set_postfix({
                '已保存': f"{saved_frames}/{total_frames}",
                '已檢測': detected_frames,
                '檢測到': total_drones_detected
            })
            pbar.update(1)
            
            frame_count += 1
    
    finally:
        cap.release()
        pbar.close()
    
    # 保存檢測結果為 JSON
    if detection_results:
        json_path = output_path / "detections.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(detection_results, f, indent=2, ensure_ascii=False)
        print(f"\n檢測結果已保存至: {json_path}")
    
    # 顯示統計資訊
    print("\n" + "="*60)
    print("處理完成！")
    print("="*60)
    print(f"保存了 {saved_frames} 幀 JPEG 圖像（所有幀）")
    print(f"進行了 {detected_frames} 次檢測")
    print(f"檢測到 {total_drones_detected} 個 Drone")
    print(f"輸出目錄: {output_path}")
    
    return str(output_path)


def main():
    """主函數"""
    video_path = "Drone.mp4"
    output_dir = "/Volumes/T7_SSD/Object_Image/Drone"
    
    process_drone_video(
        video_path=video_path,
        output_dir=output_dir,
        box_threshold=0.30,
        text_threshold=0.20,
        frame_skip=1  # 每幀都處理
    )
    
    print(f"\n{'='*60}")
    print(f"完成！Drone 檢測結果已保存")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()


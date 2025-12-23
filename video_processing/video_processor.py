"""
影片處理模組
負責影片處理的主流程：檢測 -> 分割 -> 替換 -> 生成新影片
"""
import cv2
from pathlib import Path
from tqdm import tqdm

from models.sam2_model import SAM2Model
from models.grounded_dino import GroundedDINODetector
from image_processing.football_image import FootballImageProcessor
from utils.device import get_device


class VideoProcessor:
    """
    影片處理器
    負責整個影片處理流程
    """
    
    def __init__(self, device=None):
        """
        初始化影片處理器
        
        參數:
            device: 計算設備（如果為 None，則自動選擇）
        """
        if device is None:
            device = get_device()
        
        self.device = device
        self.sam2_model = SAM2Model(device=device)
        self.dino_detector = GroundedDINODetector()
        self.football_processor = FootballImageProcessor()
    
    def process_video(self, video_path, output_video_path=None,
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
        self.football_processor.load_football_image(football_image_path)
        
        # 載入 SAM2 模型
        print("\n" + "="*60)
        print("步驟 2: 載入 SAM2.1 模型")
        print("="*60)
        self.sam2_model.load_model(model_cfg, checkpoint_path)
        
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
                    boxes, scores, labels = self.dino_detector.detect(
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
                                mask = self.sam2_model.segment(frame, box)
                                
                                # 將分割區域替換為足球圖像
                                result_frame = self.football_processor.replace_with_football(
                                    result_frame, mask, box
                                )
                                
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


"""
主程式入口
使用 DINO 檢測籃球 -> SAM2.1 分割籃球 -> 將籃球替換為足球圖像，生成新影片
流程：Basketball Detect -> Segment -> Transfer to Football
"""
from video_processing.video_processor import VideoProcessor


def main():
    """主函數"""
    # 設定輸入影片路徑
    video_path = "/Volumes/T7_SSD/Video_Test/20225_12_23_952.mp4"
    
    # 可選：設定足球圖像路徑（如果為 None，則使用預設足球圖像）
    # football_image_path = "/path/to/football_image.png"
    football_image_path = None
    
    # 創建影片處理器
    processor = VideoProcessor()
    
    # 執行處理
    output_path = processor.process_video(
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


if __name__ == "__main__":
    main()


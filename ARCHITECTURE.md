# 專案架構說明

## 目錄結構

```
Grounded SAM/
├── main.py                          # 主程式入口
├── utils/                           # 工具模組
│   ├── __init__.py
│   └── device.py                    # 設備選擇工具
├── models/                          # 模型模組
│   ├── __init__.py
│   ├── sam2_model.py                # SAM2 模型封裝
│   └── grounded_dino.py             # Grounded DINO 檢測器封裝
├── image_processing/                # 圖像處理模組
│   ├── __init__.py
│   └── football_image.py            # 足球圖像載入和替換
├── video_processing/                # 影片處理模組
│   ├── __init__.py
│   └── video_processor.py           # 影片處理主流程
└── Basketball_Purple_Segmentation.py # 原始檔案（保留作為參考）
```

## 模組說明

### 1. `utils/device.py`
- **功能**: 自動選擇計算設備（MPS/CUDA/CPU）
- **主要函數**: `get_device()`

### 2. `models/sam2_model.py`
- **功能**: SAM2 模型的載入和分割功能
- **主要類**: `SAM2Model`
  - `load_model()`: 載入 SAM2 模型
  - `segment()`: 對圖像進行分割

### 3. `models/grounded_dino.py`
- **功能**: Grounded DINO 物體檢測
- **主要類**: `GroundedDINODetector`
  - `detect()`: 檢測圖像中的物體

### 4. `image_processing/football_image.py`
- **功能**: 足球圖像的載入和替換
- **主要類**: `FootballImageProcessor`
  - `load_football_image()`: 載入或創建足球圖像
  - `replace_with_football()`: 將遮罩區域替換為足球圖像

### 5. `video_processing/video_processor.py`
- **功能**: 影片處理的主流程
- **主要類**: `VideoProcessor`
  - `process_video()`: 處理整個影片流程

### 6. `main.py`
- **功能**: 程式入口，配置參數並執行處理

## 使用方式

### 基本使用

```bash
python main.py
```

### 自訂配置

編輯 `main.py` 中的參數：

```python
processor.process_video(
    video_path="your_video.mp4",
    output_video_path=None,  # 自動生成
    football_image_path=None,  # 使用預設足球圖像
    box_threshold=0.30,
    text_threshold=0.20,
    model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
    checkpoint_path="/path/to/sam2.1_hiera_large.pt",
    max_frames=None,  # 處理全部幀
    frame_skip=1  # 每幀都處理
)
```

## 處理流程

1. **初始化**: 創建 `VideoProcessor` 實例
2. **載入資源**: 
   - 載入足球圖像
   - 載入 SAM2 模型
3. **處理影片**:
   - 逐幀讀取影片
   - 使用 DINO 檢測籃球
   - 使用 SAM2 分割籃球
   - 將籃球替換為足球圖像
   - 寫入輸出影片
4. **完成**: 輸出統計資訊和影片路徑

## 依賴項

- `torch`: PyTorch
- `cv2`: OpenCV
- `numpy`: NumPy
- `sam2`: SAM2 庫
- `DINO_Bounding_Box`: Grounded DINO 檢測模組（需要單獨提供）

# Grounded SAM - 籃球轉足球影片處理

## 專案重構說明

原檔案 `Basketball_Purple_Segmentation.py` 已重構為模組化架構，程式碼按功能分離到不同模組中。

## 快速開始

### 執行主程式

```bash
python main.py
```

### 專案結構

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
└── video_processing/                # 影片處理模組
    ├── __init__.py
    └── video_processor.py           # 影片處理主流程
```

## 模組說明

### 1. `utils/device.py`
- 自動選擇計算設備（MPS/CUDA/CPU）
- 函數：`get_device()`

### 2. `models/sam2_model.py`
- SAM2 模型的載入和分割功能
- 類：`SAM2Model`
  - `load_model()`: 載入 SAM2 模型
  - `segment()`: 對圖像進行分割

### 3. `models/grounded_dino.py`
- Grounded DINO 物體檢測
- 類：`GroundedDINODetector`
  - `detect()`: 檢測圖像中的物體

### 4. `image_processing/football_image.py`
- 足球圖像的載入和替換
- 類：`FootballImageProcessor`
  - `load_football_image()`: 載入或創建足球圖像
  - `replace_with_football()`: 將遮罩區域替換為足球圖像

### 5. `video_processing/video_processor.py`
- 影片處理的主流程
- 類：`VideoProcessor`
  - `process_video()`: 處理整個影片流程

### 6. `main.py`
- 程式入口，配置參數並執行處理

## 使用範例

### 基本使用

```python
from video_processing.video_processor import VideoProcessor

# 創建處理器
processor = VideoProcessor()

# 處理影片
output_path = processor.process_video(
    video_path="input_video.mp4",
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

### 單獨使用各個模組

```python
# 使用 SAM2 模型
from models.sam2_model import SAM2Model
from utils.device import get_device

sam2 = SAM2Model(device=get_device())
sam2.load_model(model_cfg="...", checkpoint_path="...")
mask = sam2.segment(image, bbox)

# 使用 DINO 檢測器
from models.grounded_dino import GroundedDINODetector

detector = GroundedDINODetector()
boxes, scores, labels = detector.detect(image, text_prompt="basketball.")

# 使用圖像處理器
from image_processing.football_image import FootballImageProcessor

processor = FootballImageProcessor()
processor.load_football_image("football.png")
result = processor.replace_with_football(frame, mask, bbox)
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

## 注意事項

1. 確保已安裝所有依賴項
2. 確保 `DINO_Bounding_Box.py` 檔案存在並可導入
3. 確保 SAM2 模型檔案路徑正確
4. 影片路徑和輸出路徑需要正確設定

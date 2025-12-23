"""
範例程式碼：使用 IDEA-Research/grounding-dino-base 模型
從網路下載圖片並偵測其中的物體

這個範例展示了如何：
1. 從網路下載圖片
2. 使用 Grounding DINO 模型進行物體偵測
3. 視覺化偵測結果
"""

import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO

# 1. 設定設備（優先使用 GPU，否則使用 CPU）
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"  # Apple Silicon (Mac)
else:
    device = "cpu"
print(f"使用設備: {device}")

# 2. 載入模型和處理器
print("正在載入 Grounding DINO 模型...")
model_id = "IDEA-Research/grounding-dino-base"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
model.eval()
print("模型載入完成！")

# 3. 從網路下載圖片
def download_image_from_url(url):
    """
    從 URL 下載圖片
    
    參數:
        url: 圖片的 URL 地址
    
    返回:
        PIL Image 物件
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        print(f"成功下載圖片: {url}")
        return image
    except Exception as e:
        print(f"下載圖片失敗: {e}")
        raise

# 4. 物體偵測函數
def detect_objects(image, text_prompt, box_threshold=0.3, text_threshold=0.25, debug=False):
    """
    偵測圖片中的物體
    
    參數:
        image: PIL Image 物件
        text_prompt: 要偵測的物體描述（例如："person. dog. cat. car. bicycle."）
        box_threshold: 邊界框置信度閾值
        text_threshold: 文字匹配閾值
        debug: 是否顯示除錯資訊（顯示所有候選偵測結果）
    
    返回:
        boxes: 邊界框列表 [[x1, y1, x2, y2], ...]
        scores: 置信度列表
        labels: 標籤列表
    """
    # 預處理
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    
    # 推理
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 後處理 - 先使用很低的 text_threshold 來獲取所有候選結果
    if debug:
        debug_results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            text_threshold=0.0,  # 設定為 0 來查看所有結果
            target_sizes=[image.size[::-1]]
        )[0]
        
        all_boxes = debug_results["boxes"]
        all_scores = debug_results["scores"]
        all_labels = debug_results["labels"]
        
        print(f"\n=== 除錯資訊：所有候選偵測結果 ===")
        print(f"總共偵測到 {len(all_boxes)} 個候選物體（未應用任何閾值）:")
        for i, (box, score, label) in enumerate(zip(all_boxes, all_scores, all_labels), 1):
            print(f"  {i}. {label}: 置信度={score:.3f}")
        print()
    
    # 應用 text_threshold 進行後處理
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]]  # [height, width]
    )[0]
    
    # 過濾低置信度的結果
    boxes = results["boxes"]
    scores = results["scores"]
    labels = results["labels"]
    
    if debug and len(boxes) > 0:
        print(f"應用 text_threshold={text_threshold} 後，剩餘 {len(boxes)} 個物體")
    
    # 應用 box_threshold 過濾
    if len(scores) > 0:
        mask = scores >= box_threshold
        if debug:
            before_count = len(boxes)
            after_count = int(mask.sum().item() if isinstance(mask, torch.Tensor) else mask.sum())
            print(f"應用 box_threshold={box_threshold} 前: {before_count} 個物體")
            print(f"應用 box_threshold={box_threshold} 後: {after_count} 個物體")
        boxes = boxes[mask]
        scores = scores[mask]
        labels = [labels[i] for i in range(len(labels)) if mask[i]]
    
    return boxes, scores, labels

# 5. 視覺化偵測結果
def visualize_detections(image, boxes, scores, labels):
    """
    在圖片上繪製偵測結果
    
    參數:
        image: PIL Image 物件
        boxes: 邊界框列表
        scores: 置信度列表
        labels: 標籤列表
    """
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    # 繪製每個偵測框
    for box, score, label in zip(boxes, scores, labels):
        # 轉換座標
        if isinstance(box, torch.Tensor):
            x1, y1, x2, y2 = box.cpu().numpy()
        else:
            x1, y1, x2, y2 = box
        
        # 繪製矩形框
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        
        # 添加標籤和置信度
        label_text = f"{label}: {score:.2f}"
        ax.text(
            x1, y1 - 5, label_text,
            bbox=dict(boxstyle='round', facecolor='red', alpha=0.7),
            fontsize=10, color='white', weight='bold'
        )
    
    ax.axis('off')
    plt.tight_layout()
    return fig

# 6. 主程式範例
if __name__ == "__main__":
    # 範例 1: 偵測人物和動物
    print("\n" + "="*50)
    print("範例 1: 偵測人物和動物")
    print("="*50)
    
    # 從網路下載範例圖片（可以替換為任何圖片 URL）
    image_url = "https://images.unsplash.com/photo-1511884642898-4c92249e20b6?w=800"
    # 或者使用其他圖片 URL，例如：
    # image_url = "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=800"
    
    try:
        image = download_image_from_url(image_url)
        
        # 定義要偵測的物體（用英文描述，用句號分隔）
        # 添加多種石頭相關的詞彙以提高偵測率
        text_prompt = "person. dog. cat. bicycle. car. building. tree. rock. stone. rocks. stones."
        
        print(f"\n偵測提示: {text_prompt}")
        print("正在進行物體偵測...")
        
        # 執行偵測 - 使用較低的閾值來偵測 rock
        boxes, scores, labels = detect_objects(
            image,
            text_prompt,
            box_threshold=0.20,  # 降低邊界框閾值以偵測更多物體（包括 rock）
            text_threshold=0.15,  # 降低文字匹配閾值
            debug=True  # 啟用除錯模式，顯示所有候選偵測結果
        )
        
        print(f"\n偵測結果:")
        print(f"  找到 {len(boxes)} 個物體")
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels), 1):
            print(f"  {i}. {label} (置信度: {score:.2f})")
        
        # 視覺化結果
        fig = visualize_detections(image, boxes, scores, labels)
        plt.savefig("detection_result_example1.png", dpi=150, bbox_inches='tight')
        print("\n偵測結果已儲存為: detection_result_example1.png")
        plt.show()
        
    except Exception as e:
        print(f"處理失敗: {e}")
    
    # 範例 2: 偵測特定物體（例如：籃球相關）
    print("\n" + "="*50)
    print("範例 2: 偵測籃球相關物體")
    print("="*50)
    
    # 另一個範例圖片 URL（籃球相關）
    basketball_url = "https://images.unsplash.com/photo-1546519638-68e109498ffc?w=800"
    
    try:
        image2 = download_image_from_url(basketball_url)
        
        # 偵測籃球相關物體
        text_prompt2 = "basketball player. basketball. person. court. hoop."
        
        print(f"\n偵測提示: {text_prompt2}")
        print("正在進行物體偵測...")
        
        boxes2, scores2, labels2 = detect_objects(
            image2,
            text_prompt2,
            box_threshold=0.3,
            text_threshold=0.25
        )
        
        print(f"\n偵測結果:")
        print(f"  找到 {len(boxes2)} 個物體")
        for i, (box, score, label) in enumerate(zip(boxes2, scores2, labels2), 1):
            print(f"  {i}. {label} (置信度: {score:.2f})")
        
        # 視覺化結果
        fig2 = visualize_detections(image2, boxes2, scores2, labels2)
        plt.savefig("detection_result_example2.png", dpi=150, bbox_inches='tight')
        print("\n偵測結果已儲存為: detection_result_example2.png")
        plt.show()
        
    except Exception as e:
        print(f"處理失敗: {e}")
    
    print("\n" + "="*50)
    print("範例程式碼執行完成！")
    print("="*50)
    print("\n提示:")
    print("1. 可以修改 image_url 來偵測不同的圖片")
    print("2. 可以修改 text_prompt 來偵測不同的物體")
    print("3. 可以調整 box_threshold 和 text_threshold 來控制偵測的敏感度")
    print("4. 降低閾值可以偵測更多物體，但可能增加誤檢")
    print("5. 提高閾值可以減少誤檢，但可能漏掉一些物體")


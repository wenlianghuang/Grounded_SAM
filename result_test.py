from ultralytics import YOLO

# 模型路徑（請替換為您的訓練好的模型路徑）
best_model_path = "Drone.pt"

model = YOLO(str(best_model_path))

# 測試圖像路徑（請替換為您的測試圖像）
test_image_path = "test2.jpg"

# 進行預測
results = model.predict(test_image_path, save=True, conf=0.25)

# 顯示結果
for result in results:
    result.show()  # 顯示圖像和檢測結果
    print(f"檢測到 {len(result.boxes)} 個對象")

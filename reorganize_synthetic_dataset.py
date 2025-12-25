"""
重新組織合成數據集，將現有的圖像和標籤分配到 train/val 目錄
"""
import shutil
from pathlib import Path
import random

def reorganize_synthetic_dataset(
    dataset_dir="synthetic_drone_dataset",
    train_ratio=0.8
):
    """
    重新組織數據集，添加 train/val 分割
    
    參數:
        dataset_dir: 數據集目錄
        train_ratio: 訓練集比例（默認 0.8）
    """
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        raise ValueError(f"數據集目錄不存在: {dataset_path}")
    
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"
    
    if not images_dir.exists() or not labels_dir.exists():
        raise ValueError("找不到 images 或 labels 目錄")
    
    # 檢查是否已經有 train/val 分割
    if (images_dir / "train").exists() and (images_dir / "val").exists():
        print("數據集已經有 train/val 分割，跳過重組")
        return
    
    print("\n" + "="*60)
    print("重新組織數據集（添加 train/val 分割）")
    print("="*60)
    
    # 創建 train/val 目錄
    train_images_dir = images_dir / "train"
    train_labels_dir = labels_dir / "train"
    val_images_dir = images_dir / "val"
    val_labels_dir = labels_dir / "val"
    
    for d in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # 獲取所有圖像文件
    image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    
    if len(image_files) == 0:
        print("未找到圖像文件")
        return
    
    print(f"找到 {len(image_files)} 張圖像")
    
    # 隨機打亂
    random.seed(42)  # 固定種子以確保可重現
    random.shuffle(image_files)
    
    # 分割訓練集和驗證集
    split_idx = int(len(image_files) * train_ratio)
    train_images = image_files[:split_idx]
    val_images = image_files[split_idx:]
    
    print(f"\n分割結果:")
    print(f"  訓練集: {len(train_images)} 張 ({len(train_images)/len(image_files)*100:.1f}%)")
    print(f"  驗證集: {len(val_images)} 張 ({len(val_images)/len(image_files)*100:.1f}%)")
    
    # 移動訓練集
    print("\n移動訓練集文件...")
    for img_file in train_images:
        # 移動圖像
        dst_img = train_images_dir / img_file.name
        shutil.move(str(img_file), str(dst_img))
        
        # 移動對應的標籤
        label_file = labels_dir / f"{img_file.stem}.txt"
        if label_file.exists():
            dst_label = train_labels_dir / label_file.name
            shutil.move(str(label_file), str(dst_label))
    
    # 移動驗證集
    print("移動驗證集文件...")
    for img_file in val_images:
        # 移動圖像
        dst_img = val_images_dir / img_file.name
        shutil.move(str(img_file), str(dst_img))
        
        # 移動對應的標籤
        label_file = labels_dir / f"{img_file.stem}.txt"
        if label_file.exists():
            dst_label = val_labels_dir / label_file.name
            shutil.move(str(label_file), str(dst_label))
    
    # 更新 dataset.yaml
    dataset_yaml = f"""# YOLO 合成數據集配置文件
path: {dataset_path.absolute()}
train: images/train
val: images/val

# 類別
names:
  0: drone
"""
    
    with open(dataset_path / "dataset.yaml", 'w') as f:
        f.write(dataset_yaml.strip())
    
    print("\n" + "="*60)
    print("數據集重組完成！")
    print("="*60)
    print(f"訓練集: {len(train_images)} 張")
    print(f"驗證集: {len(val_images)} 張")
    print(f"配置文件已更新: {dataset_path / 'dataset.yaml'}")


def main():
    """主函數"""
    reorganize_synthetic_dataset(
        dataset_dir="synthetic_drone_dataset",
        train_ratio=0.8
    )


if __name__ == "__main__":
    main()


import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from src.config import Config
from src.model import TrafficNet
from src.trainer import ModelTrainer
from src.data_loader import TrafficDataset

def collate_fn(batch):
    """自定義批次整理函數"""
    images = []
    boxes = []
    labels = []
    
    for item in batch:
        images.append(item['image'])
        boxes.append(item['boxes'])
        labels.append(item['labels'])
    
    images = torch.stack(images)
    
    return {
        'images': images,
        'boxes': boxes,  # 保持為列表，因為每個樣本的框數量可能不同
        'labels': labels
    }

def main():
    config = Config()
    
    # 建立必要的目錄
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    # 定義數據轉換
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 準備訓練集和驗證集
    print("正在載入 UA-DETRAC 數據集...")
    print(f"數據集路徑: {config.UA_DETRAC_PATH}")  # 添加這行來調試路徑
    
    train_dataset = TrafficDataset(
        root_dir=config.UA_DETRAC_PATH,
        dataset_type='UA-DETRAC',
        transform=transform,
        mode='train'
    )
    
    val_dataset = TrafficDataset(
        root_dir=config.UA_DETRAC_PATH,
        dataset_type='UA-DETRAC',
        transform=transform,
        mode='val'
    )
    
    print(f"訓練集大小: {len(train_dataset)}")
    print(f"驗證集大小: {len(val_dataset)}")
    
    # 創建數據加載器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    # 初始化模型和訓練器
    print("初始化模型...")
    model = TrafficNet(config)
    trainer = ModelTrainer(model, config)
    
    # 開始訓練
    print("開始訓練...")
    trainer.train(train_loader, val_loader)
    
    print("訓練完成！")

if __name__ == "__main__":
    main()

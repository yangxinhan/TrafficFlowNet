import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from PIL import Image
import logging
import torch

logger = logging.getLogger(__name__)

class TrafficDataset(Dataset):
    def __init__(self, root_dir, dataset_type='UA-DETRAC', transform=None, mode='train'):
        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.transform = transform
        self.mode = mode
        self.samples = []
        self.labels = []
        self.class_map = {}  # 用於映射車輛類型到數字標籤
        
        # 檢查目錄結構
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"數據集根目錄不存在: {root_dir}")
        
        logger.info(f"載入數據集從: {root_dir}")
        logger.info(f"目錄內容: {os.listdir(root_dir)}")
        
        if dataset_type == 'UA-DETRAC':
            self._load_ua_detrac_samples()
    
    def _load_ua_detrac_samples(self):
        img_dir = os.path.join(self.root_dir, 'DETRAC_Upload', 'images', self.mode)
        label_dir = os.path.join(self.root_dir, 'DETRAC_Upload', 'labels', self.mode)
        
        if not os.path.exists(img_dir) or not os.path.exists(label_dir):
            raise FileNotFoundError(f"找不到圖片或標籤目錄: {img_dir} 或 {label_dir}")
        
        logger.info(f"載入 {self.mode} 數據從: {img_dir}")
        
        # 過濾有效的圖片和標籤對
        valid_pairs = []
        for img_file in os.listdir(img_dir):
            if not img_file.endswith(('.jpg', '.png')):
                continue
                
            img_path = os.path.join(img_dir, img_file)
            label_file = os.path.join(label_dir, img_file.rsplit('.', 1)[0] + '.txt')
            
            if os.path.exists(label_file):
                try:
                    # 驗證標籤是否可讀
                    label = self._read_label(label_file)
                    if label['boxes'].size(0) > 0:  # 確保有有效的標註
                        valid_pairs.append((img_path, label_file))
                except Exception as e:
                    logger.warning(f"跳過無效的標籤文件 {label_file}: {str(e)}")
                    continue
        
        logger.info(f"找到 {len(valid_pairs)} 個有效的圖片-標籤對")
        self.samples = [p[0] for p in valid_pairs]
        self.label_files = [p[1] for p in valid_pairs]

    def _read_label(self, label_file):
        boxes = []
        classes = []
        with open(label_file, 'r') as f:
            for line in f:
                data = line.strip().split()
                if len(data) >= 5:  # class x y w h 格式
                    cls = int(data[0])
                    x, y, w, h = map(float, data[1:5])
                    boxes.append([x, y, w, h])
                    classes.append(cls)
        return {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(classes, dtype=torch.long)
        }
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # 讀取並處理標籤
        label = self._read_label(self.label_files[idx])
        
        return {
            'image': image,
            'boxes': label['boxes'],
            'labels': label['labels']
        }

import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

class TrafficDetector:
    def __init__(self, model_path, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
    def _load_model(self, model_path):
        # 載入訓練好的模型
        checkpoint = torch.load(model_path, map_location=self.device)
        from src.model import TrafficNet
        model = TrafficNet(self.config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model
    
    def detect_vehicles(self, frame):
        # 將 OpenCV 格式轉換為 PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        # 預處理圖像
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        # 執行偵測
        with torch.no_grad():
            predictions = self.model(image)
            probabilities = torch.softmax(predictions, dim=1)
            scores, classes = torch.max(probabilities, dim=1)
        
        return {
            'classes': classes.cpu().numpy(),
            'scores': scores.cpu().numpy(),
        }
    
    def analyze_traffic_flow(self, detections):
        # 分析車流量
        flow_data = {
            'vehicle_count': len(detections),
            'traffic_density': self._calculate_density(detections)
        }
        return flow_data
    
    def _calculate_density(self, detections):
        # 計算交通密度
        return len(detections) / 100  # 簡化示例

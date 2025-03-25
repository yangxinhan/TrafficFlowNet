import torch
import cv2
import numpy as np

class TrafficDetector:
    def __init__(self, model_path, config):
        self.config = config
        self.model = self._load_model(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _load_model(self, model_path):
        # 載入預訓練模型
        model = torch.load(model_path)
        model.to(self.device)
        return model
    
    def detect_vehicles(self, frame):
        # 執行車輛檢測
        predictions = self.model(frame)
        # 處理檢測結果
        return predictions
    
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

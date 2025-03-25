import cv2
import numpy as np
import matplotlib.pyplot as plt

class TrafficVisualizer:
    def __init__(self, config):
        self.config = config
    
    def draw_detections(self, frame, detections):
        # 在影像上標示檢測結果
        output = frame.copy()
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        return output
    
    def plot_traffic_flow(self, flow_data):
        # 繪製車流量統計圖表
        plt.figure(figsize=(10, 5))
        plt.plot(flow_data['timestamps'], flow_data['vehicle_count'])
        plt.title('車流量分析')
        plt.xlabel('時間')
        plt.ylabel('車輛數量')
        return plt

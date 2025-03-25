from src.config import Config
from src.detector import TrafficDetector
from src.visualizer import TrafficVisualizer
import cv2

def main():
    config = Config()
    detector = TrafficDetector('models/yolov5.pt', config)
    visualizer = TrafficVisualizer(config)
    
    # 讀取視頻流
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 執行檢測
        detections = detector.detect_vehicles(frame)
        
        # 分析車流
        flow_data = detector.analyze_traffic_flow(detections)
        
        # 視覺化結果
        output_frame = visualizer.draw_detections(frame, detections)
        
        # 顯示結果
        cv2.imshow('Traffic Analysis', output_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

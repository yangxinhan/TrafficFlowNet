from src.config import Config
from src.detector import TrafficDetector
from src.visualizer import TrafficVisualizer
import cv2
import os

def ensure_dir(directory):
    """確保目錄存在"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    config = Config()
    
    # 確保所需目錄存在
    ensure_dir(config.VIDEO_DIR)
    ensure_dir(config.VIDEO_OUTPUT_DIR)
    ensure_dir(config.CHECKPOINT_DIR)
    
    # 檢查模型檔案
    checkpoints = [f for f in os.listdir(config.CHECKPOINT_DIR) if f.endswith('.pt')]
    if not checkpoints:
        raise FileNotFoundError("找不到訓練好的模型檔案")
    
    latest_checkpoint = sorted(checkpoints)[-1]
    model_path = os.path.join(config.CHECKPOINT_DIR, latest_checkpoint)
    print(f"使用模型: {model_path}")
    
    # 初始化檢測器和視覺化器
    detector = TrafficDetector(model_path, config)
    visualizer = TrafficVisualizer(config)
    
    # 檢查輸入視頻
    video_files = [f for f in os.listdir(config.VIDEO_DIR) if f.endswith(('.mp4', '.avi', '.mov'))]
    if not video_files:
        print("請在 videos 目錄中放入要處理的視頻檔案")
        return
    
    for video_file in video_files:
        input_path = os.path.join(config.VIDEO_DIR, video_file)
        output_path = os.path.join(config.VIDEO_OUTPUT_DIR, f'output_{video_file}')
        
        print(f"處理視頻: {video_file}")
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            print(f"無法開啟視頻: {video_file}")
            continue
        
        # 獲取視頻參數
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # 創建視頻寫入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 執行檢測和分析
            results = detector.detect_vehicles(frame)
            flow_data = detector.analyze_traffic_flow(results)
            
            # 視覺化
            output_frame = visualizer.draw_detections(frame, results)
            cv2.putText(output_frame, 
                       f"Vehicles: {flow_data['vehicle_count']}", 
                       (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       1, 
                       (0, 255, 0), 
                       2)
            
            # 儲存和顯示
            out.write(output_frame)
            cv2.imshow('Traffic Analysis', output_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:  # 每30幀顯示一次進度
                print(f"已處理 {frame_count} 幀")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        out.release()
        print(f"完成視頻處理: {output_path}")
    
    cv2.destroyAllWindows()
    print("所有視頻處理完成")

if __name__ == "__main__":
    main()

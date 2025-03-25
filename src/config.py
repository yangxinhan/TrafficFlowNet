import os

class Config:
    # 數據集路徑
    UA_DETRAC_PATH = "/Users/yangxinhan/Downloads/UA-DETRAC"
    BDD_PATH = "data/BDD"
    
    # 模型參數
    MODEL_PATH = "models"
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 50
    
    # 檢測參數
    CONFIDENCE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.4
    
    # 輸出路徑
    OUTPUT_PATH = "output"
    
    # 訓練參數
    NUM_CLASSES = 4  # 車輛類型數量
    INPUT_CHANNELS = 3
    HIDDEN_LAYERS = [64, 128, 256]
    DROPOUT_RATE = 0.3
    NUM_WORKERS = 4
    TRAIN_VAL_SPLIT = 0.8
    
    # 訓練記錄
    LOG_DIR = "logs"
    CHECKPOINT_DIR = "checkpoints"

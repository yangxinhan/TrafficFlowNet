import os

class Config:
    # 數據集根目錄
    DATA_ROOT = "/Users/yangxinhan/Downloads/data"
    UA_DETRAC_PATH = f"{DATA_ROOT}/UA-DETRAC"  # 添加這行
    
    # UA-DETRAC 數據集路徑
    UA_DETRAC = {
        'images': {
            'train': f"{DATA_ROOT}/UA-DETRAC/images/train",
            'val': f"{DATA_ROOT}/UA-DETRAC/images/val",
            'test': f"{DATA_ROOT}/UA-DETRAC/images/test"
        },
        'labels': {
            'train': f"{DATA_ROOT}/UA-DETRAC/labels/train",
            'val': f"{DATA_ROOT}/UA-DETRAC/labels/val",
            'test': f"{DATA_ROOT}/UA-DETRAC/labels/test"
        }
    }
    
    # BDD100K 數據集路徑
    BDD100K = {
        'images': {
            'train': f"{DATA_ROOT}/BDD100K/bdd100k/images/100k/train",
            'val': f"{DATA_ROOT}/BDD100K/bdd100k/images/100k/val",
            'test': f"{DATA_ROOT}/BDD100K/bdd100k/images/100k/test"
        }
    }
    
    # YOLO 相關參數
    YOLO_CONFIG = {
        'img_size': 640,
        'conf_thres': 0.25,
        'iou_thres': 0.45,
        'max_det': 1000,
        'classes': ['car', 'bus', 'van', 'others']
    }
    
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
    
    # 視頻相關配置
    VIDEO_DIR = "videos"  # 輸入視頻目錄
    VIDEO_OUTPUT_DIR = "output/videos"  # 輸出視頻目錄
    DEFAULT_VIDEO_WIDTH = 1280
    DEFAULT_VIDEO_HEIGHT = 720
    DEFAULT_FPS = 30

import torch
import torch.nn as nn
import torch.nn.functional as F

class TrafficNet(nn.Module):
    def __init__(self, config):
        super(TrafficNet, self).__init__()
        self.config = config
        
        # 特徵提取層 - 使用更小的卷積核和更多的池化層
        self.features = nn.Sequential(
            # 第一層: 256x256 -> 128x128
            nn.Conv2d(config.INPUT_CHANNELS, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第二層: 128x128 -> 64x64
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三層: 64x64 -> 32x32
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第四層: 32x32 -> 16x16
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第五層: 16x16 -> 8x8
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 計算特徵圖的尺寸: 512 x 8 x 8 = 32768
        self.feature_size = 512 * 8 * 8
        
        # 分類層 - 使用更多的中間層來逐步降維
        self.classifier = nn.Sequential(
            nn.Dropout(p=config.DROPOUT_RATE),
            nn.Linear(self.feature_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.DROPOUT_RATE),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, config.NUM_CLASSES)
        )
        
        # 初始化權重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        batch_size = x.size(0)
        # 添加維度檢查和日誌
        print(f"Input shape: {x.shape}")
        
        x = self.features(x)
        print(f"After features shape: {x.shape}")
        
        x = x.view(batch_size, -1)
        print(f"After flatten shape: {x.shape}")
        
        x = self.classifier(x)
        print(f"Output shape: {x.shape}")
        
        return x

class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, img_size):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.nc = nc  # 類別數
        self.img_size = img_size
        
        self.conv = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, len(anchors) * (5 + nc), 1)
        )
    
    def forward(self, x):
        return self.conv(x)

class YOLO(nn.Module):
    def __init__(self, config):
        super(YOLO, self).__init__()
        self.config = config
        nc = len(config.YOLO_CONFIG['classes'])
        
        # Backbone
        self.backbone = nn.Sequential(
            # Layer 1: 640 x 640 x 3 -> 320 x 320 x 32
            self._conv_block(3, 32, 3, 2),
            # Layer 2: 320 x 320 x 32 -> 160 x 160 x 64
            self._conv_block(32, 64, 3, 2),
            # 繼續添加更多層...
        )
        
        # Detection heads
        anchors = [[10,13, 16,30, 33,23],  # P3/8
                   [30,61, 62,45, 59,119],  # P4/16
                   [116,90, 156,198, 373,326]]  # P5/32
                   
        self.yolo_layers = nn.ModuleList([
            YOLOLayer(anchors[0], nc, config.YOLO_CONFIG['img_size']),
            YOLOLayer(anchors[1], nc, config.YOLO_CONFIG['img_size']),
            YOLOLayer(anchors[2], nc, config.YOLO_CONFIG['img_size'])
        ])
    
    def _conv_block(self, in_ch, out_ch, k, s):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, padding=k//2),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return [yolo(features) for yolo in self.yolo_layers]

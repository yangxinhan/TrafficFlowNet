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

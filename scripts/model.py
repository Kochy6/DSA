import torch
import torch.nn as nn
from torchvision import models

class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Tanh(),
            nn.Linear(in_dim // 2, 1)
        )

    def forward(self, x):
        # x shape: (B, T, D)
        weights = self.attention(x) # (B, T, 1)
        weights = torch.softmax(weights, dim=1)
        # 加权求和: (B, T, D) * (B, T, 1) -> (B, D)
        output = torch.sum(x * weights, dim=1)
        return output, weights

class DSATemporalModel(nn.Module):
    def __init__(self, num_classes=2, d_model=512, nhead=8, num_layers=2):
        super().__init__()
        # 1. Backbone: ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(*(list(resnet.children())[:-1]))
        
        # 2. 时序特征投影
        self.feature_projection = nn.Linear(512, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 16, d_model)) 
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, 
            batch_first=True, dropout=0.3
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. 【核心改进】注意力池化
        self.att_pooling = AttentionPooling(d_model)
        
        # 5. 分类器
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.5), # 增加 Dropout 防止过拟合
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        
        features = self.backbone(x) # (B*T, 512, 1, 1)
        features = features.view(b, t, -1) 
        
        x = self.feature_projection(features)
        x = x + self.pos_embedding
        x = self.transformer(x) 
        
        # 使用注意力机制聚合
        global_feature, _ = self.att_pooling(x)
        
        return self.classifier(global_feature)
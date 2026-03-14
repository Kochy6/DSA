import torch
import torch.nn as nn
from torchvision import models

class DSATemporalModel(nn.Module):
    def __init__(self, num_classes=2, d_model=512, nhead=8, num_layers=3):
        super(DSATemporalModel, self).__init__()
        
        # 1. 空间特征提取器 (Backbone)
        # 使用轻量级的 ResNet18 提取单帧图像特征
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # 修改第一层卷积以适配单通道 (Gray-scale) 的 DSA 图像
        # 原始 ResNet 期待 3 通道 (RGB)，我们将其改为 1 通道
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 移除最后的全连接层 (fc)，保留特征提取部分
        # 输出特征维度通常是 512 (对于 ResNet18)
        self.backbone = nn.Sequential(*(list(resnet.children())[:-1]))
        
        # 2. 维度对齐层 (如果 Backbone 输出不是 d_model)
        self.feature_projection = nn.Linear(512, d_model)
        
        # 3. 位置编码 (Positional Encoding)
        # Transformer 本身不理解顺序，必须告诉它哪帧在前，哪帧在后
        self.pos_embedding = nn.Parameter(torch.randn(1, 32, d_model))
        
        # 4. 时序聚合器 (Transformer Encoder)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 5. 分类头 (Classifier)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        输入 x 的形状: $$(B, 1, 32, 512, 512)$$ (Batch, Channel, Time, H, W)
        """
        b, c, t, h, w = x.shape
        
        # 步骤 A: 将时间维合并到 Batch 维，利用 2D CNN 提取每一帧特征
        # 形状变为 (B*T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        
        # 通过 CNN 获取特征: (B*T, 512, 1, 1)
        features = self.backbone(x)
        features = features.view(b * t, -1) # (B*T, 512)
        
        # 步骤 B: 恢复时间维度
        # 形状变为 (B, T, 512)
        features = features.view(b, t, -1)
        features = self.feature_projection(features) # (B, T, d_model)
        
        # 步骤 C: 加入位置编码
        features = features + self.pos_embedding
        
        # 步骤 D: Transformer 聚合时序信息
        # 输出形状仍为 (B, T, d_model)
        output = self.transformer(features)
        
        # 步骤 E: 池化获取整个序列的全局特征 (Global Average Pooling over Time)
        # 取 32 帧特征的平均值作为该病例的最终特征
        global_feature = output.mean(dim=1)
        
        # 步骤 F: 分类
        logits = self.classifier(global_feature)
        
        return logits
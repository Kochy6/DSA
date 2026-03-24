import os
import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd
from sklearn.model_selection import train_test_split
from monai.networks.nets import DenseNet121
from monai.metrics import ROCAUCMetric
from monai.transforms import (
    Compose, 
    LoadImaged, 
    EnsureChannelFirstd, 
    Resized, 
    ScaleIntensityd, 
    EnsureTyped,
    RandFlipd,           # 如果你加了数据增强
    RandRotate90d,       # 如果你加了数据增强
    RandGaussianNoised,   # 如果你加了数据增强
    RandAdjustContrastd
)
from monai.data import CacheDataset, DataLoader

class Pretrained25DModel(nn.Module):
    def __init__(self, out_channels=1):
        super(Pretrained25DModel, self).__init__()
        # 1. 加载 2D 预训练 ResNet18 (也可以选 DenseNet121)
        # ResNet18 更轻量，适合 100 个样本的小数据集
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        
        # 2. 适配单通道输入 (DSA 是灰度图，ImageNet 是 3 通道)
        original_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # 将原始 3 通道权重的均值初始化给新层，保留部分预训练特征
        with torch.no_grad():
            self.backbone.conv1.weight[:] = original_conv.weight.mean(dim=1, keepdim=True)

        # 3. 移除原始全连接层，准备提取特征
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # 4. 最终分类头
        self.classifier = nn.Linear(512, out_channels)

    def forward(self, x):
        # x shape: [Batch, 1, Depth, Height, Width] -> [B, 1, 48, 512, 512]
        b, c, d, h, w = x.shape
        
        # 将深度维度合并到 Batch 维度，方便 2D 骨干处理
        # x -> [B*D, C, H, W]
        x = x.transpose(1, 2).flatten(0, 1) 
        
        # 提取每一帧的特征 -> [B*D, 512, 1, 1]
        features = self.feature_extractor(x)
        features = features.view(b, d, -1) # -> [B, 48, 512]
        
        # 时间维度聚合：对 48 帧特征求平均 (Temporal Pooling)
        # 这一步将变长的 40-50 帧信息浓缩为固定维度的特征向量
        combined_features = features.mean(dim=1) # -> [B, 512]
        
        # 输出 Logits
        return self.classifier(combined_features)


# --- 1. 环境与路径配置，定义多源数据路径 --- ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 数据源 A
DATA_DIR_A = "/mnt/pro/DSA/data/ori_data/DICOM1"
CSV_PATH_A = "/mnt/pro/DSA/csv/label_sorted.csv"

# 数据源 B
DATA_DIR_B = "/autodl-fs/data/Pro/DSA/dicom_all"
CSV_PATH_B = "/autodl-fs/data/Pro/DSA/label.csv"

# --- 2. 分别读取并构建数据字典 ---
def build_data_dicts(csv_path, data_dir):
    """验证并构建单一数据源的字典列表"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"标签文件未找到: {csv_path}")
        
    df = pd.read_csv(csv_path, dtype={'filename': str})
    
    # 强制检查列名一致性，确保两批数据的格式对齐
    if 'filename' not in df.columns or 'label' not in df.columns:
        raise ValueError(f"CSV文件 {csv_path} 缺少必要的 'filename' 或 'label' 列。")

    return [
        {"file": os.path.join(data_dir, str(row['filename'])), "label": int(row['label'])} 
        for _, row in df.iterrows()
    ]

data_dicts_A = build_data_dicts(CSV_PATH_A, DATA_DIR_A)
data_dicts_B = build_data_dicts(CSV_PATH_B, DATA_DIR_B)

# --- 3. 整合多源数据 ---
merged_data_dicts = data_dicts_A + data_dicts_B
print(f"数据整合完毕。数据源A: {len(data_dicts_A)}例, 数据源B: {len(data_dicts_B)}例, 总计: {len(merged_data_dicts)}例。")

# --- 4. 稳健的数据集切分 (分层抽样) ---
# 提取所有标签用于分层，确保训练集和验证集的正负样本比例与总体一致
all_labels = [d["label"] for d in merged_data_dicts]

train_dicts, val_dicts = train_test_split(
    merged_data_dicts, 
    test_size=0.2,       # 设定 20% 的数据作为验证集
    random_state=42,     # 设定随机种子以保证实验的可重复性
    stratify=all_labels  # 关键逻辑：基于标签分布进行分层抽样
)
print(f"切分完成。训练集: {len(train_dicts)}例, 验证集: {len(val_dicts)}例。")
# print(data_dicts)  

# --- 3. 定义预处理管线 (Transforms) ---
# 注意：须降采样，否则可能显存溢出
train_transforms = Compose([
    LoadImaged(keys=["file"], reader="PydicomReader"),
    EnsureChannelFirstd(keys=["file"]),
    Resized(keys=["file"], spatial_size=(48, 512, 512), mode="trilinear"),
    
    # --- 新增：数据增强策略 ---
    # 1. 空间增强：随机翻转和旋转（血管的结构具有对称性）
    RandFlipd(keys=["file"], prob=0.5, spatial_axis=1), # 水平翻转
    RandRotate90d(keys=["file"], prob=0.5, spatial_axes=[1, 2]), # 90度旋转
    
    # 2. 强度增强：模拟造影剂浓度差异和机器噪声
    RandAdjustContrastd(keys=["file"], prob=0.5, gamma=(0.7, 1.3)),
    RandGaussianNoised(keys=["file"], prob=0.3, mean=0.0, std=0.05),
    
    ScaleIntensityd(keys=["file"]),
    EnsureTyped(keys=["file", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["file"], reader="PydicomReader"),
    EnsureChannelFirstd(keys=["file"]),
    Resized(keys=["file"], spatial_size=(48, 512, 512), mode="trilinear"),
    ScaleIntensityd(keys=["file"]),
    EnsureTyped(keys=["file", "label"]),
])


# --- 4. 构建带缓存的 Dataset (CacheDataset) ---
# cache_rate=1.0 表示 100 个文件全部缓存到内存 (约占 8-10GB RAM)
# 注意：数据量增加后，cache_rate=1.0 可能会导致显存/内存溢出。
# 如果物理内存不足，需降低 cache_rate 或改用常规 Dataset。
train_ds = CacheDataset(data=train_dicts, transform=train_transforms, cache_rate=1.0)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)

val_ds = CacheDataset(data=val_dicts, transform=val_transforms, cache_rate=1.0)
val_loader = DataLoader(val_ds, batch_size=1)

# --- 5. 定义 3D 网络、损失函数与优化器 ---

model = Pretrained25DModel(out_channels=1).to(device)
loss_function = torch.nn.BCEWithLogitsLoss() 
# 将参数分为两组，给予不同的学习率
optimizer = torch.optim.Adam([
    # feature_extractor 已经包含了 conv1 以及所有 ResNet 的中间层
    {'params': model.feature_extractor.parameters(), 'lr': 1e-4},
    # 随机初始化的分类头使用更高的学习率
    {'params': model.classifier.parameters(), 'lr': 1e-3}
])
auc_metric = ROCAUCMetric()
# acc_metric = AccuracyMetric(reduction="mean") # 初始化准确率指标

# --- 6. 训练循环 ---
print("开始训练（第一个 Epoch 正在缓存数据，请稍候...）")
for epoch in range(20):
    model.train()
    epoch_loss = 0
    for batch_data in train_loader:
        inputs = batch_data["file"].to(device)
        labels = batch_data["label"].float().to(device).view(-1, 1)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # 验证环节
    model.eval()
    with torch.no_grad():
        y_pred = []
        y_true = []
        for val_data in val_loader:
            val_images = val_data["file"].to(device)
            val_labels = val_data["label"].float().to(device).view(-1) 
            
            output = model(val_images)
            pred = torch.sigmoid(output).view(-1) 
            
            y_pred.append(pred)
            y_true.append(val_labels)
        
        y_pred = torch.cat(y_pred)
        y_true = torch.cat(y_true)
        
        # --- 计算 AUC ---
        auc_metric(y_pred, y_true)
        auc_value = auc_metric.aggregate()
        auc_metric.reset()

        # 2. 手动计算 Accuracy (完全不依赖 monai.metrics)
        # 将概率转为 0/1 类别
        y_pred_class = (y_pred >= 0.5).float()
        # 计算预测正确的个数
        correct = (y_pred_class == y_true).float().sum()
        # 计算百分比
        acc_value = (correct / len(y_true)).item()

    print(f"Epoch {epoch+1}: Loss = {epoch_loss/len(train_loader):.4f}, "
          f"Val AUC = {auc_value:.4f}, Val Acc = {acc_value:.4f}")
torch.save(model.state_dict(), "pre_based_dsa_prognosis_model_313bs4.pth")
import os
import torch
import pandas as pd
from monai.networks.nets import DenseNet121
from monai.metrics import ROCAUCMetric
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Resized, ScaleIntensityd, EnsureTyped
from monai.data import CacheDataset, DataLoader

# --- 1. 环境与路径配置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "/mnt/pro/DSA/data/ori_data/DICOM1"
CSV_PATH = "/mnt/pro/DSA/csv/label_sorted.csv"

# --- 2. 准备数据字典列表 ---
df = pd.read_csv(CSV_PATH, dtype={'filename': str})
# print(df.head(),type(df))  # 检查数据读取是否正确

data_dicts = [
    {"file": os.path.join(DATA_DIR, str(row['filename'])), "label": row['label']} 
    for _, row in df.iterrows()
]
# print(data_dicts)  

# --- 3. 定义预处理管线 (Transforms) ---
# 注意：须降采样，否则可能显存溢出
train_transforms = Compose([
    LoadImaged(keys=["file"], reader="PydicomReader"),
    EnsureChannelFirstd(keys=["file"]),
    # 核心修改：统一帧数为 48（应对 40-50 帧的波动），高宽设为 512
    Resized(
        keys=["file"], 
        spatial_size=(48, 512, 512), 
        mode="trilinear"
    ),
    ScaleIntensityd(keys=["file"]),
    EnsureTyped(keys=["file", "label"]),
])

# --- 4. 构建带缓存的 Dataset (CacheDataset) ---
# cache_rate=1.0 表示 100 个文件全部缓存到内存 (约占 8-10GB RAM)
train_ds = CacheDataset(data=data_dicts[:80], transform=train_transforms, cache_rate=1.0)
print(train_ds[0]["file"].shape,train_ds[1]["file"].shape,train_ds[2]["file"].shape,train_ds[3]["file"].shape,train_ds[4]["file"].shape,train_ds[5]["file"].shape) 
# train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)

# val_ds = CacheDataset(data=data_dicts[80:], transform=train_transforms, cache_rate=1.0)
# val_loader = DataLoader(val_ds, batch_size=1)

# # --- 5. 定义 3D 网络、损失函数与优化器 ---
# # 使用 DenseNet121 处理 3D 时空数据
# model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=1).to(device)
# loss_function = torch.nn.BCEWithLogitsLoss() 
# optimizer = torch.optim.Adam(model.parameters(), 1e-4)
# auc_metric = ROCAUCMetric()
# # acc_metric = AccuracyMetric(reduction="mean") # 初始化准确率指标

# # --- 6. 训练循环 ---
# print("开始训练（第一个 Epoch 正在缓存数据，请稍候...）")
# for epoch in range(20):
#     model.train()
#     epoch_loss = 0
#     for batch_data in train_loader:
#         inputs = batch_data["file"].to(device)
#         labels = batch_data["label"].float().to(device).view(-1, 1)
        
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = loss_function(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()

#     # 验证环节
#     model.eval()
#     with torch.no_grad():
#         y_pred = []
#         y_true = []
#         for val_data in val_loader:
#             val_images = val_data["file"].to(device)
#             val_labels = val_data["label"].float().to(device).view(-1) 
            
#             output = model(val_images)
#             pred = torch.sigmoid(output).view(-1) 
            
#             y_pred.append(pred)
#             y_true.append(val_labels)
        
#         y_pred = torch.cat(y_pred)
#         y_true = torch.cat(y_true)
        
#         # --- 计算 AUC ---
#         auc_metric(y_pred, y_true)
#         auc_value = auc_metric.aggregate()
#         auc_metric.reset()

#         # 2. 手动计算 Accuracy (完全不依赖 monai.metrics)
#         # 将概率转为 0/1 类别
#         y_pred_class = (y_pred >= 0.5).float()
#         # 计算预测正确的个数
#         correct = (y_pred_class == y_true).float().sum()
#         # 计算百分比
#         acc_value = (correct / len(y_true)).item()

#     print(f"Epoch {epoch+1}: Loss = {epoch_loss/len(train_loader):.4f}, "
#           f"Val AUC = {auc_value:.4f}, Val Acc = {acc_value:.4f}")
# torch.save(model.state_dict(), "dsa_prognosis_model.pth")
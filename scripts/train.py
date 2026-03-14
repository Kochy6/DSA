import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import random

from dataset import DSADataset
from model import DSATemporalModel

# --- 1. 超参数与全局配置 ---
cfg = {
    "csv_path": r"/mnt/pro/DSA/cleansed_list.csv",
    "batch_size": 2,          # 物理显存限制
    "accumulation_steps": 8,  # 梯度累积，等效 Batch Size = 16
    "epochs": 50,
    "freeze_epochs": 10,      # 前10轮冻结Backbone
    "lr": 5e-5,
    "num_folds": 5,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "save_dir": "./checkpoints"
}

def seed_everything(seed=42):
    """固定随机种子以保证实验可复现性"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- 2. 核心训练逻辑 ---
def train_one_epoch(model, loader, optimizer, criterion, device, epoch, scaler):
    model.train()
    running_loss = 0.0
    total_preds = []
    total_labels = []
    
    os.makedirs('debug_images', exist_ok=True)
    optimizer.zero_grad()

    for batch_idx, (images, labels) in enumerate(loader):
        # 调试可视化：每个 Epoch 的第一组数据
        if batch_idx == 0:
            with torch.no_grad():
                # 兼容 3D [B, C, D, H, W] 或 2D [B, C, H, W]
                img_plot = images[0, 0, images.shape[2]//2].cpu().numpy() if images.ndim == 5 else images[0, 0].cpu().numpy()
                plt.figure(figsize=(6, 6))
                plt.imshow(img_plot, cmap='gray')
                plt.title(f"Epoch_{epoch}_Label_{labels[0].item()}")
                plt.savefig(f"debug_images/epoch_{epoch}.png")
                plt.close()

        images, labels = images.to(device), labels.to(device)

        # 混合精度前向传播
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / cfg["accumulation_steps"] 

        # 梯度缩放与反向传播
        scaler.scale(loss).backward()
        
        # 梯度累积更新
        if (batch_idx + 1) % cfg["accumulation_steps"] == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * cfg["accumulation_steps"] * images.size(0)
        
        with torch.no_grad():
            _, predicted = torch.max(outputs, 1)
            total_preds.extend(predicted.cpu().numpy())
            total_labels.extend(labels.cpu().numpy())

    # 处理剩余梯度
    if (len(loader) % cfg["accumulation_steps"]) != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, np.array(total_preds), np.array(total_labels)

# --- 3. 验证逻辑 ---
def validate(model, loader, criterion, device):
    model.eval()
    all_probs = []
    all_labels = []
    running_loss = 0.0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    val_auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
    val_acc = accuracy_score(all_labels, np.array(all_probs) > 0.5)
    return running_loss / len(loader.dataset), val_auc, val_acc

# --- 4. 主程序 ---
def main():
    seed_everything(42)
    os.makedirs(cfg["save_dir"], exist_ok=True)
    
    # 【新增】定义日志文件路径
    log_path = os.path.join(cfg["save_dir"], "training_results.txt")
    
    # 【新增】辅助函数：同时打印到控制台和写入文件
    def log(message):
        print(message)
        with open(log_path, "a") as f:
            f.write(message + "\n")

    # 初始化日志文件头
    with open(log_path, "w") as f:
        f.write("=== Training Log ===\n")

    df = pd.read_csv(cfg["csv_path"])
    full_dataset = DSADataset(cfg["csv_path"])
    
    # 计算类别权重 (负样本数 / 正样本数)
    neg_c, pos_c = len(df[df['label']==0]), len(df[df['label']==1])
    class_weights = torch.tensor([1.0, neg_c / pos_c], dtype=torch.float).to(cfg["device"])
    log(f"[INFO] 权重分配: {class_weights.cpu().numpy()}") # 【修改】使用log

    skf = StratifiedKFold(n_splits=cfg["num_folds"], shuffle=True, random_state=42)
    
    fold_auc_scores = [] # 【新增】记录每折结果

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
        log(f"\n{'='*10} Fold {fold+1}/{cfg['num_folds']} {'='*10}") # 【修改】使用log
        
        train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=cfg["batch_size"], 
                                  shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        # ...existing code...
        val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=cfg["batch_size"], 
                                num_workers=2, pin_memory=True)
        
        model = DSATemporalModel(num_classes=2).to(cfg["device"])
        
        # 初始状态：冻结 Backbone
        for param in model.backbone.parameters():
            param.requires_grad = False
        
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["lr"])
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        scaler = torch.amp.GradScaler('cuda') # 修正旧版调用方式
        
        best_auc = 0.0
        
        for epoch in range(cfg["epochs"]):
            # 第 10 轮解冻 Backbone
            if epoch == cfg["freeze_epochs"]:
                for param in model.backbone.parameters():
                    param.requires_grad = True
                torch.cuda.empty_cache()
                optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"] * 0.2)
                scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"] - epoch)
                log(f"[INFO] Epoch {epoch}: Backbone已解冻，开始微调。") # 【修改】使用log

            # 调用训练和验证
            # ...existing code...
            train_loss, t_preds, t_labels, = train_one_epoch(model, train_loader, optimizer, criterion, cfg["device"], epoch, scaler)
            val_loss, val_auc, val_acc = validate(model, val_loader, criterion, cfg["device"])
            
            scheduler.step()

            # 打印与保存
            save_msg = ""
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), f"{cfg['save_dir']}/best_fold{fold+1}.pth")
                save_msg = " [*] Best Saved"
            
            # 【修改】构建日志信息并记录
            epoch_msg = (f"Epoch {epoch:02d} | Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f} | "
                         f"Acc: {val_acc:.4f} | 1/0: {np.sum(t_preds==1)}/{np.sum(t_preds==0)}{save_msg}")
            log(epoch_msg)
        
        fold_auc_scores.append(best_auc)
        log(f"Fold {fold+1} Best AUC: {best_auc:.4f}")

    log(f"\n=== Training Finished. Mean AUC: {np.mean(fold_auc_scores):.4f} ===")

if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import random

from scripts.core.dataset import DSADataset
from scripts.core.model import DSATemporalModel

# --- 1. 超参数与全局配置 ---
cfg = {
    "csv_path": r"/mnt/pro/DSA/cleansed_list.csv",
    "batch_size": 4,          # 物理显存限制
    "accumulation_steps": 8,  # 梯度累积，等效 Batch Size = 32
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
    total_probs = []  # 【修改】改为收集概率值而不是硬标签 (0/1)
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
        
        # 【修改】获取类别1的预测概率，用于计算 Train AUC
        with torch.no_grad():
            probs = torch.softmax(outputs, dim=1)[:, 1]
            total_probs.extend(probs.cpu().numpy())
            total_labels.extend(labels.cpu().numpy())

    # 处理剩余梯度
    if (len(loader) % cfg["accumulation_steps"]) != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, np.array(total_probs), np.array(total_labels)

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
    
    # 定义日志文件路径
    log_path = os.path.join(cfg["save_dir"], "training_results_batchsize_4.txt")
    
    # 辅助函数：同时打印到控制台和写入文件
    def log(message):
        print(message)
        with open(log_path, "a") as f:
            f.write(message + "\n")

    # 初始化日志文件头
    with open(log_path, "w") as f:
        f.write("=== Training Log ===\n")

    df = pd.read_csv(cfg["csv_path"])
    # 计算类别权重 (负样本数 / 正样本数)
    neg_c, pos_c = len(df[df['label']==0]), len(df[df['label']==1])
    class_weights = torch.tensor([1.0, neg_c / pos_c], dtype=torch.float).to(cfg["device"])
    log(f"[INFO] 权重分配: {class_weights.cpu().numpy()}") 

    skf = StratifiedKFold(n_splits=cfg["num_folds"], shuffle=True, random_state=42)
    fold_auc_scores = [] 

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
        log(f"\n{'='*10} Fold {fold+1}/{cfg['num_folds']} {'='*10}") 
        
        train_dataset = DSADataset(cfg["csv_path"], mode='train')
        val_dataset = DSADataset(cfg["csv_path"], mode='val')

        train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=cfg["batch_size"], shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(Subset(val_dataset, val_idx), batch_size=cfg["batch_size"], shuffle=False, num_workers=2, pin_memory=True)
        
        model = DSATemporalModel(num_classes=2).to(cfg["device"])
        
        # 初始状态：冻结 Backbone
        for param in model.backbone.parameters():
            param.requires_grad = False
        
        # 1. 解冻前的优化器 (只练顶层)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["lr"])
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        scaler = torch.amp.GradScaler('cuda') 
        
        best_auc = 0.0
        
        for epoch in range(cfg["epochs"]):
            # 第 10 轮解冻 Backbone
            if epoch == cfg["freeze_epochs"]:
                for param in model.backbone.parameters():
                    param.requires_grad = True
                torch.cuda.empty_cache()
                optimizer = optim.AdamW([
                {'params': model.backbone.parameters(), 'lr': cfg["lr"] * 0.1}, 
                {'params': model.transformer.parameters(), 'lr': cfg["lr"]},
                {'params': model.att_pooling.parameters(), 'lr': cfg["lr"]},
                {'params': model.classifier.parameters(), 'lr': cfg["lr"]}
                                        ], weight_decay=0.05) # 显式增加权重衰减
                scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"] - epoch)
                log(f"[INFO] Epoch {epoch}: Backbone已解冻，开始微调。") 

            # 调用训练
            train_loss, t_probs, t_labels = train_one_epoch(model, train_loader, optimizer, criterion, cfg["device"], epoch, scaler)
            
            # 【新增】计算 Train AUC 
            try:
                train_auc = roc_auc_score(t_labels, t_probs) if len(np.unique(t_labels)) > 1 else 0.5
            except ValueError:
                train_auc = 0.5
                
            # 调用验证
            val_loss, val_auc, val_acc = validate(model, val_loader, criterion, cfg["device"])
            
            scheduler.step()

            # 打印与保存
            save_msg = ""
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), f"{cfg['save_dir']}/best_fold{fold+1}.pth")
                save_msg = " [*] Best Saved"
            
            # 【修改】将 Train 概率转为二值预测，用于计算1/0分布
            t_preds_binary = (t_probs > 0.5).astype(int)
            pred_1_count = np.sum(t_preds_binary == 1)
            pred_0_count = np.sum(t_preds_binary == 0)

            # 【修改】构建带 Train AUC 的日志信息
            epoch_msg = (f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | "
                         f"Val AUC: {val_auc:.4f} | Acc: {val_acc:.4f} | 1/0: {pred_1_count}/{pred_0_count}{save_msg}")
            log(epoch_msg)
        
        fold_auc_scores.append(best_auc)
        log(f"Fold {fold+1} Best AUC: {best_auc:.4f}")

    log(f"\n=== Training Finished. Mean AUC: {np.mean(fold_auc_scores):.4f} ===")

if __name__ == "__main__":
    main()
"""
train.py — Training pipeline for DSATemporalModel
==================================================
Previous fixes retained:
  [FIX-1]  Data leakage eliminated: single base DSADataset + AugmentedSubset.
  [FIX-3]  seq_len passed to model constructor from cfg["target_t"].
  [FIX-4]  Gradient accumulation double-step at epoch boundary fixed.
  [FIX-5]  Scheduler continuity: backbone added as param group to existing
           optimizer; no scheduler reset on unfreeze.
  [FIX-9]  Symmetric class weights: total/(2*count[c]).
  [FIX-10] Gradient clipping before every optimizer step.

New fix applied:
  [FIX-A]  Stronger regularization strategy for the unfreeze transition,
           validated against the fold 1/2 training log showing severe
           overfitting (train AUC 0.69 vs val AUC 0.55 by epoch 49).

           Three sub-changes:

           [FIX-A1] Delayed backbone unfreeze: epoch 10 → epoch 25.
           Rationale: the log shows val AUC was already at its lifetime peak
           (0.5603) by epoch 1, before the head had learned anything useful.
           The backbone unfreeze at epoch 10 immediately widened the train/val
           gap from ~0.00 to ~0.05 and continued to widen it. The head needs
           more time to converge to a stable feature extractor before the
           backbone parameters are allowed to adapt. Delaying to epoch 25
           (halfway through training) gives the head 25 epochs on frozen,
           stable ResNet-18 features before fine-tuning begins.

           [FIX-A2] Aggressive backbone weight decay at unfreeze: 0.05 → 0.2.
           Rationale: with ~435 training samples, an 11M-parameter backbone
           can fit the training set almost perfectly once unfrozen. A higher
           L2 penalty shrinks backbone weights toward zero, preventing the
           backbone from straying far from its ImageNet initialization (which
           already encodes useful edge and texture priors). The top-layer
           weight decay is also increased from 0.05 → 0.1 to match.

           [FIX-A3] ReduceLROnPlateau replaces CosineAnnealingLR.
           Rationale: with a flat val AUC signal (0.51–0.56 plateau), a
           cosine schedule has no way to react to the plateau — it decays
           regardless of validation performance. ReduceLROnPlateau monitors
           val_auc and halves the LR after 7 epochs without improvement.
           This gives the optimizer a chance to find a lower-loss basin when
           the current LR is too large to escape a local flat region.
           On this dataset size, an adaptive LR schedule consistently
           outperforms fixed schedules.

           [FIX-A4] Early stopping with patience=15.
           Rationale: fold 1 showed no improvement after epoch 1. Without
           early stopping, 49 more epochs were wasted purely overfitting.
           Early stopping halts the fold when val AUC has not improved for
           15 consecutive epochs, freeing GPU time for other folds.
"""

import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
import torchvision.transforms.v2 as v2

from dataset import DSADataset
from model import DSATemporalModel


# ---------------------------------------------------------------------------
# 1. Hyper-parameters and global configuration
# ---------------------------------------------------------------------------
cfg = {
    "csv_path":            r"/mnt/pro/DSA/cleansed_list.csv",
    "batch_size":          4,        # Physical VRAM limit
    "accumulation_steps":  8,        # Effective batch size = 4 × 8 = 32
    "epochs":              60,       # Increased headroom; early stopping will
                                     # terminate most folds well before epoch 60
    # --- [FIX-A1] Delayed unfreeze ---
    "freeze_epochs":       25,       # was 10; head trains for 25 epochs first
    # --- Learning rates ---
    "lr":                  5e-5,     # head LR (all epochs)
    "backbone_lr_frac":    0.1,      # backbone LR = lr * backbone_lr_frac
    # --- [FIX-A2] Regularization ---
    "head_weight_decay":   0.1,      # was 0.05; stronger head regularization
    "backbone_weight_decay": 0.2,    # was 0.05; strong backbone regularization
    # --- [FIX-A3] ReduceLROnPlateau ---
    "lr_factor":           0.5,      # LR multiplied by this on plateau
    "lr_patience":         7,        # epochs without improvement before reduce
    "lr_min":              1e-7,     # floor on LR
    # --- [FIX-A4] Early stopping ---
    "early_stop_patience": 15,       # stop fold if val AUC flat for N epochs
    # --- Other ---
    "max_grad_norm":       1.0,      # [FIX-10] gradient clipping
    "num_folds":           5,
    "target_t":            32,       # temporal frames; shared with dataset & model
    "device":              torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # 动态生成 save_dir：基础路径 + 时间戳
    "save_dir":            f"./checkpoints/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",  # directory to save best models
}


# ---------------------------------------------------------------------------
# 2. Reproducibility
# ---------------------------------------------------------------------------
def seed_everything(seed: int = 42):
    """Fix all random seeds for reproducible experiments."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# 3. [FIX-1] Augmentation wrapper
# ---------------------------------------------------------------------------
class RandomGaussianNoise(torch.nn.Module):
    """自定义的高斯噪声注入模块，模拟 X 射线硬件的本底噪声"""
    def __init__(self, mean=0., std=0.02, p=0.5):
        super().__init__()
        self.mean = mean
        self.std = std
        self.p = p

    def forward(self, img):
        if torch.rand(1).item() < self.p:
            noise = torch.randn_like(img) * self.std + self.mean
            return img + noise
        return img

_TRAIN_TRANSFORM = v2.Compose(
    [
        # 微调空间仿射：不翻转！仅模拟患者体位的一点点摆偏，或者C臂机的微小缩放
        v2.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.95, 1.05), fill=0),
        # 光度学抖动：模拟 X 线曝光剂量和造影剂浓度的差异
        v2.ColorJitter(brightness=0.1, contrast=0.1),
        # 高斯噪声注入：模拟探测器的电子噪声/量子斑点
        RandomGaussianNoise(std=0.02, p=0.3),
        # 随机小面积遮罩：挖掉 1%~5% 的小区域，防过拟合
        v2.RandomErasing(p=0.2, scale=(0.01, 0.05), ratio=(0.5, 2.0), value=0)
    ]
)


class AugmentedSubset(torch.utils.data.Dataset):
    """
    Wraps a Subset and applies a training-time transform on-the-fly.

    A single base DSADataset (mode='val', no augmentation) is created once.
    Train subsets are wrapped with this class; val subsets use the base
    dataset directly. This makes the K-Fold split the single source of truth
    and prevents data leakage that would occur if two separate DSADataset
    instances were created and Subset applied to both.
    """

    def __init__(self, subset: Subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        images, label = self.subset[idx]
        return self.transform(images), label


# ---------------------------------------------------------------------------
# 4. Training logic (one epoch)
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, epoch, scaler):
    model.train() # 训练模式启用 Dropout 和 BatchNorm 更新
    running_loss   = 0.0 # 累积损失，用于计算 epoch 平均损失
    total_probs    = [] # 累积预测概率，用于 epoch 结束时计算 AUC
    total_labels   = [] # 累积真实标签，与 total_probs 对齐

    os.makedirs("debug_images", exist_ok=True) # 存储调试图像的目录
    optimizer.zero_grad() # [FIX-4] zero_grad at epoch start; step() will zero_grad after every update
    last_step_batch_idx = -1  # [FIX-4] track last optimizer step

    for batch_idx, (images, labels) in enumerate(loader): # 遍历 DataLoader 的每个 batch
        # ----- [FIX-10] Debug: save mid-frame of first batch each epoch ----

        # Debug: save mid-frame of first batch each epoch for visual sanity check
        if batch_idx == 0:
            with torch.no_grad():
                img_plot = (
                    images[0, 0, images.shape[2] // 2].cpu().numpy()
                    if images.ndim == 5
                    else images[0, 0].cpu().numpy()
                )
                plt.figure(figsize=(6, 6))
                plt.imshow(img_plot, cmap="gray")
                plt.title(f"Epoch_{epoch}_Label_{labels[0].item()}")
                plt.savefig(f"debug_images/epoch_{epoch}.png")
                plt.close()

        images, labels = images.to(device), labels.to(device)

        with torch.amp.autocast("cuda"):
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss    = loss / cfg["accumulation_steps"]

        scaler.scale(loss).backward()

        if (batch_idx + 1) % cfg["accumulation_steps"] == 0: # 每 accumulation_steps 个 batch 更新一次
            # [FIX-10] Unscale first so clip_grad_norm operates on true grads
            scaler.unscale_(optimizer) # 取消缩放，恢复原始梯度值，以便正确应用梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"]) # 梯度裁剪，防止爆炸
            scaler.step(optimizer) # 更新参数
            scaler.update() # 更新 scaler 状态
            optimizer.zero_grad() # 清零梯度，为下一轮累积准备
            last_step_batch_idx = batch_idx  # [FIX-4] 记录最后一次更新的 batch 索引

        running_loss += loss.item() * cfg["accumulation_steps"] * images.size(0) # 累积损失，乘回 accumulation_steps 和 batch 大小以得到正确的总损失

        with torch.no_grad():
            probs = torch.softmax(outputs, dim=1)[:, 1]
            total_probs.extend(probs.cpu().numpy())
            total_labels.extend(labels.cpu().numpy())

    # [FIX-4] Flush remaining gradient only if last batch did not already step
    if last_step_batch_idx != len(loader) - 1:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, np.array(total_probs), np.array(total_labels)


# ---------------------------------------------------------------------------
# 5. Validation logic
# ---------------------------------------------------------------------------
def validate(model, loader, criterion, device):
    model.eval()
    all_probs  = []
    all_labels = []
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss    = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    val_auc = (
        roc_auc_score(all_labels, all_probs)
        if len(np.unique(all_labels)) > 1
        else 0.5
    )
    val_acc = accuracy_score(all_labels, all_probs > 0.5)
    return running_loss / len(loader.dataset), val_auc, val_acc


# ---------------------------------------------------------------------------
# 6. Main training loop
# ---------------------------------------------------------------------------
def main():
    seed_everything(42)
    os.makedirs(cfg["save_dir"], exist_ok=True)

    log_path = os.path.join(cfg["save_dir"], "training_log.txt")

    def log(msg: str):
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    with open(log_path, "w") as f:
        f.write("=== Training Log ===\n")

    # -----------------------------------------------------------------------
    # Class weights [FIX-9]: symmetric, proportional to class frequency
    # -----------------------------------------------------------------------
    df    = pd.read_csv(cfg["csv_path"])
    neg_c = len(df[df["label"] == 0])
    pos_c = len(df[df["label"] == 1])
    total = neg_c + pos_c
    w0    = total / (2.0 * neg_c)
    w1    = total / (2.0 * pos_c)
    class_weights = torch.tensor([w0, w1], dtype=torch.float).to(cfg["device"])
    log(f"[INFO] Class distribution   neg={neg_c}, pos={pos_c}, total={total}")
    log(f"[INFO] Class weights        w0={w0:.4f}, w1={w1:.4f}")
    log(f"[INFO] freeze_epochs        {cfg['freeze_epochs']}  [FIX-A1]")
    log(f"[INFO] backbone_weight_decay {cfg['backbone_weight_decay']}  [FIX-A2]")
    log(f"[INFO] scheduler            ReduceLROnPlateau(patience={cfg['lr_patience']})  [FIX-A3]")
    log(f"[INFO] early_stop_patience  {cfg['early_stop_patience']}  [FIX-A4]")

    # -----------------------------------------------------------------------
    # [FIX-1] Single base dataset — augmentation injected via AugmentedSubset
    # -----------------------------------------------------------------------
    base_dataset = DSADataset(
        cfg["csv_path"], mode="val", target_t=cfg["target_t"]
    )

    skf = StratifiedKFold(n_splits=cfg["num_folds"], shuffle=True, random_state=42)
    fold_auc_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label"])):
        log(f"\n{'='*10} Fold {fold+1}/{cfg['num_folds']} {'='*10}")

        # [FIX-1] Wrap train subset with augmentation; val subset uses base dataset
        train_subset = AugmentedSubset(Subset(base_dataset, train_idx), _TRAIN_TRANSFORM)
        val_subset   = Subset(base_dataset, val_idx)

        train_loader = DataLoader(
            train_subset, batch_size=cfg["batch_size"], shuffle=True,
            # Windows(开发)使用0线程防内存暴跌，Linux(训练)使用4线程并开启持久化
            num_workers=0 if os.name == 'nt' else 4, 
            pin_memory=True, 
            persistent_workers=False if os.name == 'nt' else True,
        )
        val_loader = DataLoader(
            val_subset, batch_size=cfg["batch_size"], shuffle=False,
            num_workers=0 if os.name == 'nt' else 2, 
            pin_memory=True,
        )

        # [FIX-3] seq_len passed from cfg so positional structures match target_t
        model = DSATemporalModel(
            num_classes=2, seq_len=cfg["target_t"]
        ).to(cfg["device"])

        # -----------------------------------------------------------------
        # Freeze backbone for the first freeze_epochs epochs [FIX-A1]
        # -----------------------------------------------------------------
        for param in model.backbone.parameters():
            param.requires_grad = False

        # Optimizer covers only the unfrozen top layers initially
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg["lr"],
            weight_decay=cfg["head_weight_decay"],  # [FIX-A2] stronger head WD
        )

        # -----------------------------------------------------------------
        # [FIX-A3] ReduceLROnPlateau: reacts to val AUC, not epoch count.
        # mode='max' because higher AUC is better.
        # factor=0.5 halves the LR on plateau.
        # patience=7: wait 7 epochs of no improvement before reducing.
        # -----------------------------------------------------------------
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=cfg["lr_factor"],
            patience=cfg["lr_patience"],
            min_lr=cfg["lr_min"],
        )

        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        scaler    = torch.amp.GradScaler("cuda")
        best_auc  = 0.0

        # [FIX-A4] Early stopping state
        epochs_no_improve = 0

        for epoch in range(cfg["epochs"]):

            # -------------------------------------------------------------
            # [FIX-A1 / FIX-A2] Unfreeze backbone at freeze_epochs.
            #
            # Delayed to epoch 25 (was 10) so the head stabilizes on fixed
            # ResNet features before backbone fine-tuning begins.
            #
            # [FIX-5] Backbone added as a NEW PARAM GROUP to the existing
            # optimizer instead of creating a new optimizer + scheduler.
            # This preserves the LR history for the head param group and
            # avoids resetting the ReduceLROnPlateau plateau counter.
            # -------------------------------------------------------------
            if epoch == cfg["freeze_epochs"]:
                # [阶梯式解冻优化] 仅解冻 ResNet 的高级层 (layer3, layer4)
                # 保持底层 (conv1, layer1, layer2) 永远冻结，防止医疗小样本破坏底层普适特征
                backbone_children = list(model.backbone.children())
                layer3 = backbone_children[6]
                layer4 = backbone_children[7]
                
                for param in layer3.parameters():
                    param.requires_grad = True
                for param in layer4.parameters():
                    param.requires_grad = True
                    
                torch.cuda.empty_cache()

                backbone_lr = cfg["lr"] * cfg["backbone_lr_frac"]
                # 仅将解冻的 layer3 和 layer4 的参数传给优化器
                optimizer.add_param_group(
                    {
                        "params":       list(layer3.parameters()) + list(layer4.parameters()),
                        "lr":           backbone_lr,
                        # [FIX-A2] High weight decay prevents backbone from
                        # straying far from its ImageNet initialization
                        "weight_decay": cfg["backbone_weight_decay"],
                    }
                )
                log(
                    f"[INFO] Epoch {epoch}: Backbone阶梯式解冻 (仅Layer3 & Layer4) — "
                    f"lr={backbone_lr:.2e}, wd={cfg['backbone_weight_decay']} "
                    f"[阶梯解冻优化]"
                )

            # ----- Train one epoch ----------------------------------------
            train_loss, t_probs, t_labels = train_one_epoch(
                model, train_loader, optimizer, criterion,
                cfg["device"], epoch, scaler,
            )

            try:
                train_auc = (
                    roc_auc_score(t_labels, t_probs)
                    if len(np.unique(t_labels)) > 1
                    else 0.5
                )
            except ValueError:
                train_auc = 0.5

            # ----- Validate -----------------------------------------------
            val_loss, val_auc, val_acc = validate(
                model, val_loader, criterion, cfg["device"]
            )

            # ----- [FIX-A3] Step scheduler on val AUC (not epoch count) ---
            scheduler.step(val_auc)

            # Retrieve current LR for logging (first param group)
            current_lr = optimizer.param_groups[0]["lr"]

            # ----- Checkpoint on improvement ------------------------------
            save_msg = ""
            if val_auc > best_auc:
                best_auc         = val_auc
                epochs_no_improve = 0  # [FIX-A4] reset counter
                ckpt_path = os.path.join(cfg["save_dir"], f"best_fold{fold+1}.pth")
                torch.save(model.state_dict(), ckpt_path)
                save_msg = " [*] Best Saved"
            else:
                epochs_no_improve += 1  # [FIX-A4]

            # ----- Log ----------------------------------------------------
            t_preds_bin   = (t_probs > 0.5).astype(int)
            pred_1_count  = int(np.sum(t_preds_bin == 1))
            pred_0_count  = int(np.sum(t_preds_bin == 0))

            epoch_msg = (
                f"Epoch {epoch:02d} | "
                f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | "
                f"Val AUC: {val_auc:.4f} | Acc: {val_acc:.4f} | "
                f"1/0: {pred_1_count}/{pred_0_count} | "
                f"LR: {current_lr:.2e}"
                f"{save_msg}"
            )
            log(epoch_msg)

            # ----- [FIX-A4] Early stopping --------------------------------
            if epochs_no_improve >= cfg["early_stop_patience"]:
                log(
                    f"[INFO] Early stopping triggered — val AUC has not "
                    f"improved for {cfg['early_stop_patience']} consecutive "
                    f"epochs. Best val AUC this fold: {best_auc:.4f}."
                )
                break

        fold_auc_scores.append(best_auc)
        log(f"Fold {fold+1} Best AUC: {best_auc:.4f}")

    log(
        f"\n=== Training Finished. "
        f"Mean AUC: {np.mean(fold_auc_scores):.4f} ± "
        f"{np.std(fold_auc_scores):.4f} ==="
    )


if __name__ == "__main__":
    main()
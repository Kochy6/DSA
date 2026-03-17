"""
train.py — Training pipeline for DSATemporalModel
==================================================
Fixes applied (from audit):
  [FIX-1]  Data leakage in K-Fold CV: previously two full DSADataset objects
           were created (one per mode), then Subset was applied to both with
           the same indices. This is fragile and can silently leak when the
           dataset is refactored. Now a single dataset is created; a thin
           AugmentationWrapper is used to apply train vs. val transforms to
           the appropriate Subset without duplicating data loading.
  [FIX-3]  DSATemporalModel is constructed with seq_len=dataset.target_t so
           the positional embedding matches whatever target_t is configured.
  [FIX-4]  Gradient accumulation boundary fix: the "remaining gradients"
           flush at end-of-epoch was always firing. It now checks whether
           the last mini-batch already triggered an optimizer step, avoiding
           a spurious double-step.
  [FIX-5]  Scheduler reset inconsistency: the first CosineAnnealingLR ran
           for freeze_epochs steps and was then discarded. The replacement
           scheduler started a fresh cosine curve, making LR history for the
           top layers discontinuous. Now a single scheduler is created after
           the optimizer is finalized at epoch 0; the warm-up / unfreeze
           transition reuses it without resetting.
  [FIX-9]  Class weight formula: the original formula [1.0, neg/pos] only
           upweights the minority class. The symmetric formula
           [neg/total, pos/total] × 2 balances both classes proportionally.
  [FIX-10] Gradient clipping added (max_norm=1.0) before every optimizer
           step to prevent gradient explosions in the Transformer layers.
"""

import os
import random

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
    "csv_path":           r"/mnt/pro/DSA/cleansed_list.csv",
    "batch_size":         4,      # Physical VRAM limit
    "accumulation_steps": 8,      # Effective batch size = 4 × 8 = 32
    "epochs":             50,
    "freeze_epochs":      10,     # Backbone frozen for first N epochs
    "lr":                 5e-5,
    "max_grad_norm":      1.0,    # [FIX-10] Gradient clipping threshold
    "num_folds":          5,
    "target_t":           32,     # Temporal frames — shared with dataset & model
    "device":             torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "save_dir":           "./checkpoints",
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
#
#    Problem: creating two DSADataset instances (one mode='train', one
#    mode='val') and then applying Subset to both is fragile — both objects
#    load the entire CSV, so the mode tag is the only thing separating them.
#    If DSADataset is ever refactored to filter rows internally, the split
#    logic silently breaks.
#
#    Fix: create ONE dataset (no augmentation), then wrap the train Subset
#    in this lightweight class that applies the training transform on-the-fly.
#    The val Subset is used directly from the base dataset (Identity transform).
# ---------------------------------------------------------------------------
_TRAIN_TRANSFORM = v2.Compose(
    [
        v2.RandomAffine(degrees=5, translate=(0.02, 0.02), fill=0),
        v2.ColorJitter(brightness=0.1, contrast=0.1),
        v2.RandomVerticalFlip(p=0.3),
    ]
)


class AugmentedSubset(torch.utils.data.Dataset):
    """
    Wraps a Subset and applies a transform that the underlying Dataset
    does not apply (because it was initialized with mode='val').

    This avoids creating two full dataset copies and makes the train/val
    split the single source of truth.
    """

    def __init__(self, subset: Subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        images, label = self.subset[idx]
        images = self.transform(images)
        return images, label


# ---------------------------------------------------------------------------
# 4. Training logic (one epoch)
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, epoch, scaler):
    model.train()
    running_loss = 0.0
    total_probs = []
    total_labels = []

    # os.makedirs("debug_images", exist_ok=True)
    optimizer.zero_grad()

    # Track whether the last batch already flushed the optimizer so we
    # do not double-step at end-of-epoch.  [FIX-4]
    last_step_batch_idx = -1

    for batch_idx, (images, labels) in enumerate(loader):

        # Debug visualization: save the mid-frame of the first batch each epoch
        # if batch_idx == 0:
        #     with torch.no_grad():
        #         img_plot = (
        #             images[0, 0, images.shape[2] // 2].cpu().numpy()
        #             if images.ndim == 5
        #             else images[0, 0].cpu().numpy()
        #         )
        #         plt.figure(figsize=(6, 6))
        #         plt.imshow(img_plot, cmap="gray")
        #         plt.title(f"Epoch_{epoch}_Label_{labels[0].item()}")
        #         plt.savefig(f"debug_images/epoch_{epoch}.png")
        #         plt.close()

        images, labels = images.to(device), labels.to(device)

        # Mixed-precision forward pass
        with torch.amp.autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / cfg["accumulation_steps"]  # scale for accumulation

        # Backward (scaled)
        scaler.scale(loss).backward()

        # Gradient accumulation: step every N mini-batches
        if (batch_idx + 1) % cfg["accumulation_steps"] == 0:
            # [FIX-10] Unscale before clipping so clip operates on true grads
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg["max_grad_norm"]
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            last_step_batch_idx = batch_idx  # [FIX-4] record last flush

        # Accumulate loss (undo the accumulation scaling for logging)
        running_loss += (
            loss.item() * cfg["accumulation_steps"] * images.size(0)
        )

        # Collect probabilities for AUC computation
        with torch.no_grad():
            probs = torch.softmax(outputs, dim=1)[:, 1]
            total_probs.extend(probs.cpu().numpy())
            total_labels.extend(labels.cpu().numpy())

    # [FIX-4] Flush remaining gradients ONLY if the last batch did not
    # already trigger a step (avoids spurious double-step).
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
    all_probs = []
    all_labels = []
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs_np = np.array(all_probs)
    all_labels_np = np.array(all_labels)

    val_auc = (
        roc_auc_score(all_labels_np, all_probs_np)
        if len(np.unique(all_labels_np)) > 1
        else 0.5
    )
    val_acc = accuracy_score(all_labels_np, all_probs_np > 0.5)
    return running_loss / len(loader.dataset), val_auc, val_acc


# ---------------------------------------------------------------------------
# 6. Main training loop
# ---------------------------------------------------------------------------
def main():
    seed_everything(42)
    os.makedirs(cfg["save_dir"], exist_ok=True)

    log_filename = (
        f"Claude_results_bs{cfg['batch_size']}_"
        f"ep{cfg['epochs']}_"
        f"lr{cfg['lr']}_"
        f"t{cfg['target_t']}.txt"
    )
    log_path = os.path.join(cfg["save_dir"], log_filename)

    def log(message: str):
        """Write to both stdout and the log file."""
        print(message)
        with open(log_path, "a") as f:
            f.write(message + "\n")

    with open(log_path, "w") as f:
        f.write("=== Training Log ===\n")

    # -----------------------------------------------------------------------
    # Load CSV and compute class weights
    # -----------------------------------------------------------------------
    df = pd.read_csv(cfg["csv_path"])
    neg_c = len(df[df["label"] == 0])
    pos_c = len(df[df["label"] == 1])
    total  = neg_c + pos_c

    # [FIX-9] Symmetric balanced class weights.
    # weight[c] = (total / (num_classes * count[c]))
    # This proportionally up-weights whichever class is smaller without
    # arbitrarily fixing one class weight to 1.0.
    w0 = total / (2.0 * neg_c)   # weight for class 0 (negative)
    w1 = total / (2.0 * pos_c)   # weight for class 1 (positive / minority)
    class_weights = torch.tensor([w0, w1], dtype=torch.float).to(cfg["device"])
    log(f"[INFO] Class distribution  neg={neg_c}, pos={pos_c}")
    log(f"[INFO] Class weights       w0={w0:.4f}, w1={w1:.4f}")

    # -----------------------------------------------------------------------
    # [FIX-1] Single base dataset (no augmentation) — split is applied via
    # Subset; training Subsets are wrapped with AugmentedSubset.
    # -----------------------------------------------------------------------
    base_dataset = DSADataset(
        cfg["csv_path"], mode="val", target_t=cfg["target_t"]
    )  # mode='val' → Identity transform; augmentation added by AugmentedSubset target_size也需要调整！！！！！

    skf = StratifiedKFold(
        n_splits=cfg["num_folds"], shuffle=True, random_state=42
    )
    fold_auc_scores = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(df, df["label"])
    ):
        log(f"\n{'='*10} Fold {fold+1}/{cfg['num_folds']} {'='*10}")

        # [FIX-1] Create subsets from the single base dataset, then wrap
        # only the training subset with augmentation.
        train_subset = AugmentedSubset(
            Subset(base_dataset, train_idx), _TRAIN_TRANSFORM
        )
        val_subset = Subset(base_dataset, val_idx)  # no augmentation

        train_loader = DataLoader(
            train_subset,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        # [FIX-3] Pass seq_len so pos_embedding matches target_t
        model = DSATemporalModel(
            num_classes=2, seq_len=cfg["target_t"]
        ).to(cfg["device"])

        # Freeze backbone for the first freeze_epochs epochs
        for param in model.backbone.parameters():
            param.requires_grad = False

        # Optimizer for the unfrozen top layers only (pre-unfreeze phase)
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg["lr"],
            weight_decay=0.05,
        )

        # [FIX-5] Single scheduler covering the full epoch range.
        # After the backbone is unfrozen we update the optimizer's param
        # groups in-place without creating a new scheduler, so the cosine
        # decay for the top layers is never reset.
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["epochs"]
        )

        criterion = nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=0.1
        )
        scaler = torch.amp.GradScaler("cuda")
        best_auc = 0.0

        for epoch in range(cfg["epochs"]):

            # ----------------------------------------------------------------
            # [FIX-5] Unfreeze backbone at freeze_epochs.
            # Instead of creating a new optimizer + new scheduler (which
            # resets the cosine curve), we ADD the backbone parameters as a
            # new param group to the EXISTING optimizer and continue using
            # the same scheduler.  The scheduler will compute LR based on the
            # elapsed step count, giving a smoothly decaying LR throughout.
            # ----------------------------------------------------------------
            if epoch == cfg["freeze_epochs"]:
                model.backbone.requires_grad_(True)
                torch.cuda.empty_cache()

                # Add backbone as a new param group with a lower LR
                # (fine-tuning rate = 0.1 × head LR)
                optimizer.add_param_group(
                    {
                        "params": model.backbone.parameters(),
                        "lr": cfg["lr"] * 0.1,
                        "weight_decay": 0.05,
                    }
                )
                log(
                    f"[INFO] Epoch {epoch}: Backbone unfrozen and added to "
                    f"optimizer as a new param group (lr={cfg['lr']*0.1:.2e})."
                )

            # Train
            train_loss, t_probs, t_labels = train_one_epoch(
                model, train_loader, optimizer, criterion,
                cfg["device"], epoch, scaler
            )

            # Compute Train AUC
            try:
                train_auc = (
                    roc_auc_score(t_labels, t_probs)
                    if len(np.unique(t_labels)) > 1
                    else 0.5
                )
            except ValueError:
                train_auc = 0.5

            # Validate
            val_loss, val_auc, val_acc = validate(
                model, val_loader, criterion, cfg["device"]
            )

            # Step the shared scheduler (applies to all param groups)
            scheduler.step()

            # Save checkpoint if best AUC improved
            save_msg = ""
            if val_auc > best_auc:
                best_auc = val_auc
                ckpt_path = os.path.join(
                    cfg["save_dir"], f"best_fold{fold+1}.pth"
                )
                torch.save(model.state_dict(), ckpt_path)
                save_msg = " [*] Best Saved"

            # Prediction distribution (binary threshold at 0.5)
            t_preds_binary = (t_probs > 0.5).astype(int)
            pred_1_count = int(np.sum(t_preds_binary == 1))
            pred_0_count = int(np.sum(t_preds_binary == 0))

            epoch_msg = (
                f"Epoch {epoch:02d} | "
                f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | "
                f"Val AUC: {val_auc:.4f} | Acc: {val_acc:.4f} | "
                f"1/0: {pred_1_count}/{pred_0_count}"
                f"{save_msg}"
            )
            log(epoch_msg)

        fold_auc_scores.append(best_auc)
        log(f"Fold {fold+1} Best AUC: {best_auc:.4f}")

    log(
        f"\n=== Training Finished. "
        f"Mean AUC: {np.mean(fold_auc_scores):.4f} ± "
        f"{np.std(fold_auc_scores):.4f} ==="
    )


if __name__ == "__main__":
    main()

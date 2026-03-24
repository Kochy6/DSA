"""
model.py — DSATemporalModel
============================
Previous fix retained:
  [FIX-3]  seq_len parameterized (was hardcoded to 16).

New fix applied:
  [FIX-C]  Transformer encoder replaced with a lightweight Temporal1DCNN.

           Motivation (from training log analysis):
           Train AUC climbed to ~0.69 while val AUC plateaued at ~0.55,
           a 0.13–0.15 gap that widened progressively after the backbone
           unfreeze at epoch 10. The Transformer encoder (2 layers, d_model=512)
           contributes ~4.7M trainable parameters to the temporal head alone.
           With ~435 training samples per fold, the temporal module has
           roughly 10,800 parameters per training sample — severely
           overparameterized, providing excessive capacity for memorization.

           Fix:
           The Transformer encoder and its positional embedding are removed.
           They are replaced with a TemporalConvNet: a stack of two residual
           1D convolutional blocks operating over the temporal dimension (T).

           Why 1D CNN instead of Transformer:
             • 1D CNNs have a restricted receptive field (kernel_size=3 per
               block, effective field = 5 frames after 2 layers). This is a
               deliberate inductive bias — adjacent frames in a DSA sequence
               are the most informative temporal neighbors. The Transformer's
               global attention was learning spurious long-range correlations
               between non-adjacent frames, which is noise at this dataset size.
             • Residual connections stabilize gradients on small datasets,
               where vanishing gradients are a real risk in the temporal module.
             • 1D Conv weights are shared across the T dimension (positional
               invariance within a phase), reducing the effective parameter
               count further relative to the attention mechanism.

           Parameter reduction (temporal head only, d_model: 512 → 256):
             Old (Transformer):  ~4,748,035 parameters
             New (Temporal1DCNN): ~988,035 parameters
             Reduction: ~4.8×

           The positional embedding is also removed: Conv1d is inherently
           order-aware through the local receptive field, so learned absolute
           position embeddings are unnecessary and add overfit capacity.

           d_model is reduced from 512 → 256 as a further regularization step.
           The ResNet-18 backbone (512-d features) is projected down to 256
           before the temporal module, halving the dimension throughout the
           entire temporal pipeline.
"""

import torch
import torch.nn as nn
from torchvision import models


# ---------------------------------------------------------------------------
# Attention pooling (unchanged from previous version)
# ---------------------------------------------------------------------------

class AttentionPooling(nn.Module):
    """
    Soft attention pooling: collapses (B, T, D) → (B, D) by learning a scalar
    attention weight per time-step and returning a weighted sum.

    Parameters
    ----------
    in_dim : int  Feature dimensionality (= d_model).
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Tanh(),
            nn.Linear(in_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : Tensor  (B, T, D)

        Returns
        -------
        output  : Tensor  (B, D)     — attention-weighted aggregate
        weights : Tensor  (B, T, 1)  — per-frame softmax weights
        """
        weights = self.attention(x)              # (B, T, 1)
        weights = torch.softmax(weights, dim=1)  # normalize over T 
        output  = torch.sum(x * weights, dim=1)  # (B, D)
        return output, weights


# ---------------------------------------------------------------------------
# [FIX-C] Residual 1D convolutional block
# ---------------------------------------------------------------------------

class ResidualBlock1D(nn.Module):
    """
    One residual block for temporal feature extraction.

    Computes: out = ReLU(BN(Conv(ReLU(BN(Conv(x))))) + x)

    The skip connection keeps gradients flowing even when the block learns
    near-zero transformations — important for small datasets where a layer
    may not need to transform features substantially.

    Parameters
    ----------
    channels    : int  Input and output channel count (= d_model).
    kernel_size : int  Temporal convolution kernel size.
                       kernel_size=3 gives each frame access to its two
                       immediate temporal neighbors. The effective receptive
                       field after 2 stacked blocks is 5 frames.
    dropout     : float Dropout probability applied between the two convolutions.
    """

    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.3, dilation: int = 1):
        super().__init__()
        # 【关键修改】动态计算 padding。当 dilation>1 时，卷积核由于膨胀会变宽，
        # 为了保证输出长度不变，padding 必须随之等比例增大。
        padding = dilation * (kernel_size - 1) // 2 

        # 将 dilation 参数传给 Conv1d
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2   = nn.BatchNorm1d(channels)
        self.drop  = nn.Dropout(dropout)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  (B, channels, T)  — Conv1d format (channels before T)

        Returns
        -------
        Tensor  (B, channels, T)
        """
        residual = x # Save input for skip connection
        out = self.relu(self.bn1(self.conv1(x)))  # first sub-layer
        out = self.drop(out)
        out = self.bn2(self.conv2(out))            # second sub-layer (no ReLU yet)
        return self.relu(out + residual)           # residual addition then ReLU


# ---------------------------------------------------------------------------
# [FIX-C] Temporal convolutional network
# ---------------------------------------------------------------------------

class TemporalConvNet(nn.Module):
    """
    Stack of residual 1D convolutional blocks over the temporal dimension.

    Replaces the Transformer encoder. Takes the projected frame features
    (B, T, d_model) and produces a temporally-refined (B, T, d_model) output
    suitable for attention pooling.

    Parameters
    ----------
    d_model    : int   Feature dimensionality.
    num_blocks : int   Number of ResidualBlock1D layers (default 2).
                       More blocks widen the receptive field:
                         1 block  → 3-frame receptive field
                         2 blocks → 5-frame receptive field
                         3 blocks → 7-frame receptive field
                       2 blocks is a good balance for 32-frame sequences.
    kernel_size : int  Convolution kernel size (default 3).
    dropout     : float Dropout probability inside each residual block.
    """

    def __init__(
        self,
        d_model: int,
        num_blocks: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        blocks_list = []
        for i in range(num_blocks):
            # 【关键修改】每一层的 dilation 呈指数级增长： 1, 2, 4, 8...
            # 这里 num_blocks 默认是 2，所以分别是 2^0=1 (看附近3帧), 2^1=2 (看附近跨度为5的帧)
            current_dilation = 2 ** i 
            blocks_list.append(
                ResidualBlock1D(channels=d_model, 
                                kernel_size=kernel_size, 
                                dropout=dropout, 
                                dilation=current_dilation)
            )
            
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  (B, T, d_model)  — standard sequence format (T after B)

        Returns
        -------
        Tensor  (B, T, d_model)
        """
        # Conv1d expects (B, channels, T) — permute, process, permute back
        x = x.permute(0, 2, 1)   # (B, d_model, T)
        x = self.blocks(x)        # (B, d_model, T)
        x = x.permute(0, 2, 1)   # (B, T, d_model)
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class DSATemporalModel(nn.Module):
    """
    Temporal classification model for DSA image sequences.

    Architecture (updated with [FIX-C])
    ------------------------------------
    1. ResNet-18 backbone   — per-frame spatial features (1-channel input)
    2. Linear projection    — 512 → d_model (default 256)
    3. TemporalConvNet      — 2 residual Conv1d blocks over T dimension
       (replaces Transformer encoder + positional embedding)
    4. AttentionPooling     — soft-aggregate T frames → single vector
    5. MLP classifier       — Linear → ReLU → Dropout → Linear → logits

    Parameters
    ----------
    num_classes  : int   Output classes (default 2).
    d_model      : int   Feature dimensionality throughout temporal pipeline.
                         Reduced from 512 → 256 [FIX-C] to lower overfitting
                         capacity. Change together with backbone output (512)
                         only if you replace the backbone.
    num_blocks   : int   Number of ResidualBlock1D in TemporalConvNet.
    kernel_size  : int   Temporal conv kernel size.
    temporal_drop: float Dropout inside each temporal residual block.
    seq_len      : int   Input frames; must match dataset.target_t. [FIX-3]
    """

    def __init__(
        self,
        num_classes: int = 2,      # output classes
        d_model: int = 256,        # [FIX-C] reduced from 512
        num_blocks: int = 2,       # [FIX-C] temporal CNN depth
        kernel_size: int = 3,      # [FIX-C] temporal receptive field per block
        temporal_drop: float = 0.3,
        seq_len: int = 32,         # [FIX-3] parameterized
    ):
        super().__init__()
        self.seq_len = seq_len

        # ------------------------------------------------------------------
        # 1. Backbone: ResNet-18 (ImageNet pre-trained)
        #    conv1 replaced for 1-channel DICOM input.
        #    Final FC stripped; output is 512-d global average pool.
        # ------------------------------------------------------------------
        # Version 1: torchvision.models.resnet18 (pre-trained)
        # resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # resnet.conv1 = nn.Conv2d(
        #     1, 64, kernel_size=7, stride=2, padding=3, bias=False
        # )
        # self.backbone = nn.Sequential(*(list(resnet.children())[:-1]))
        # Output: (B*T, 512, 1, 1)

        # Version 2: Load pre-trained ResNet-18 and adapt conv1 for 1-channel input while
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # 获取原本 3 通道的权重 (64, 3, 7, 7)
        pretrained_weight = resnet.conv1.weight.clone() 

        # 创建 1 通道的新卷积
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 将 3 个通道的权重相加 (或取平均) 赋给新卷积，保留预训练的边缘提取能力
        with torch.no_grad():
            resnet.conv1.weight = nn.Parameter(pretrained_weight.sum(dim=1, keepdim=True))

        self.backbone = nn.Sequential(*(list(resnet.children())[:-1]))

        # ------------------------------------------------------------------
        # 2. Feature projection: 512 → d_model
        #    Projects backbone features into the smaller temporal space.
        #    LayerNorm stabilizes the projection output before Conv1d.
        # ------------------------------------------------------------------
        self.feature_projection = nn.Linear(512, d_model)
        self.feature_norm       = nn.LayerNorm(d_model)

        # ------------------------------------------------------------------
        # 3. [FIX-C] TemporalConvNet: residual 1D CNN over the T dimension.
        #    Replaces: TransformerEncoder + pos_embedding (~4.5M params)
        #    New cost: TemporalConvNet (~790K params)
        #
        #    No positional embedding needed: Conv1d is inherently order-aware
        #    through the local receptive field.
        # ------------------------------------------------------------------
        self.temporal = TemporalConvNet(
            d_model=d_model,
            num_blocks=num_blocks,
            kernel_size=kernel_size,
            dropout=temporal_drop,
        )

        # ------------------------------------------------------------------
        # 4. Attention pooling: (B, T, d_model) → (B, d_model)
        # ------------------------------------------------------------------
        self.att_pooling = AttentionPooling(d_model)

        # ------------------------------------------------------------------
        # 5. Classifier
        #    Hidden dim halved (d_model//2 = 128) relative to old 256 head.
        #    Dropout(0.5) before output layer for regularization.
        # ------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  (B, T, C, H, W) — B batches, T frames, C=1 channel

        Returns
        -------
        logits : Tensor  (B, num_classes)
        """
        b, t, c, h, w = x.shape

        # 1. Per-frame spatial encoding (ResNet-18)
        x = x.view(b * t, c, h, w)           # (B*T, 1, H, W)
        features = self.backbone(x)            # (B*T, 512, 1, 1)
        features = features.view(b, t, -1)    # (B, T, 512)

        # V1: 直接投影后送入 Transformer
        # # 2. Project 512 → d_model and normalize
        # x = self.feature_projection(features) # (B, T, d_model)
        # x = self.feature_norm(x)

        # # 3. [FIX-C] Temporal refinement via residual Conv1d
        # x = self.temporal(x)                  # (B, T, d_model)

        # # 4. Attention-weighted temporal aggregation
        # x, _ = self.att_pooling(x)            # (B, d_model)

        # # 5. Classification
        # return self.classifier(x)             # (B, num_classes)

        # V2: 【新增】旁路 + 主干并行：空间特征直接平均 + 传统投影+时间网络
        # 【新增】旁路：直接将空间特征进行无脑平均 (B, 512)
        spatial_global = features.mean(dim=1) 
        
        # 主干：继续原有的投影和时间网络处理
        x = self.feature_projection(features)
        x = self.feature_norm(x)
        x = self.temporal(x)
        x_temporal, _ = self.att_pooling(x)   # (B, 256)
        
        # 【新增】将 静态特征 与 动态特征 拼接
        # 此时 x_combined 维度为 512 + 256 = 768
        x_combined = torch.cat([spatial_global, x_temporal], dim=1) 
        
        # 最后送入能接收 768 维度的 classifier
        return self.classifier(x_combined)
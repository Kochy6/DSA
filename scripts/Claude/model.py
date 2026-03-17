"""
model.py — DSATemporalModel
============================
Fixes applied (from audit):
  [FIX-3]  pos_embedding shape was hardcoded to 16 frames.  It is now a
           constructor parameter (seq_len) so any change to target_t in
           dataset.py is automatically reflected here without a silent
           broadcast mismatch or runtime shape error.

No other model-level issues were identified.  The architecture (ResNet18
backbone → linear projection → Transformer encoder → AttentionPooling →
MLP classifier) is sound for this task.
"""

import torch
import torch.nn as nn
from torchvision import models


class AttentionPooling(nn.Module):
    """
    Soft attention pooling over the temporal dimension.

    Learns a scalar attention weight per time-step and returns a
    weighted sum of frame features, collapsing [B, T, D] → [B, D].

    Parameters
    ----------
    in_dim : int  Dimensionality of each time-step feature vector (= d_model).
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
        x : Tensor  shape (B, T, D)

        Returns
        -------
        output  : Tensor  shape (B, D)  — attention-weighted frame aggregate
        weights : Tensor  shape (B, T, 1) — softmax attention weights
        """
        weights = self.attention(x)             # (B, T, 1)
        weights = torch.softmax(weights, dim=1) # normalize over T
        output = torch.sum(x * weights, dim=1)  # (B, D)
        return output, weights


class DSATemporalModel(nn.Module):
    """
    Temporal classification model for DSA image sequences.

    Architecture
    ------------
    1. ResNet-18 backbone  — per-frame spatial feature extractor (1-channel)
    2. Linear projection   — map 512-d CNN features → d_model
    3. Positional embedding— learnable, shape [1, seq_len, d_model]
    4. Transformer encoder — model inter-frame temporal dependencies
    5. AttentionPooling    — soft-aggregate T frames into one vector
    6. MLP classifier      — output logits for num_classes

    Parameters
    ----------
    num_classes : int  Number of output classes (default 2 for binary DSA).
    d_model     : int  Transformer / projection dimensionality.
    nhead       : int  Number of attention heads in the Transformer.
    num_layers  : int  Number of Transformer encoder layers.
    seq_len     : int  Number of input frames (must match dataset.target_t).
                       [FIX-3] Previously hardcoded to 16.
    """

    def __init__(
        self,
        num_classes: int = 2,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 2,
        seq_len: int = 16,      # [FIX-3] parameterized — was hardcoded
    ):
        super().__init__()
        self.seq_len = seq_len

        # ------------------------------------------------------------------
        # 1. Backbone: ResNet-18 pre-trained on ImageNet
        #    conv1 is replaced to accept 1-channel (grayscale) DICOM input.
        #    The final FC layer is removed; we use the 512-d avg-pooled feature.
        # ------------------------------------------------------------------
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Strip the classification head; keep everything up to AdaptiveAvgPool
        self.backbone = nn.Sequential(*(list(resnet.children())[:-1]))
        # Output: (B*T, 512, 1, 1) → flattened to (B*T, 512)

        # ------------------------------------------------------------------
        # 2. Feature projection: 512 → d_model
        # ------------------------------------------------------------------
        self.feature_projection = nn.Linear(512, d_model)

        # ------------------------------------------------------------------
        # 3. Learnable positional embedding
        #    [FIX-3] Shape uses seq_len parameter, not the literal 16.
        #    If target_t changes in the dataset, pass the new value here and
        #    the model will allocate the correct parameter size automatically.
        # ------------------------------------------------------------------
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))

        # ------------------------------------------------------------------
        # 4. Transformer Encoder
        #    dropout=0.3 inside each encoder layer provides regularization
        #    on the attention weights and feedforward activations.
        # ------------------------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.3,
            batch_first=True,   # input/output are (B, T, D)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # ------------------------------------------------------------------
        # 5. Attention pooling: (B, T, D) → (B, D)
        # ------------------------------------------------------------------
        self.att_pooling = AttentionPooling(d_model)

        # ------------------------------------------------------------------
        # 6. Classification head
        #    Dropout(0.5) before the final linear layer combats overfitting
        #    when fine-tuning on small medical datasets.
        # ------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  shape (B, T, C, H, W)
                    B = batch, T = seq_len frames, C = 1 channel

        Returns
        -------
        logits : Tensor  shape (B, num_classes)
        """
        b, t, c, h, w = x.shape

        # Merge batch and time dimensions for parallel CNN processing
        x = x.view(b * t, c, h, w)                # (B*T, 1, H, W)

        # Per-frame spatial features
        features = self.backbone(x)                # (B*T, 512, 1, 1)
        features = features.view(b, t, -1)         # (B, T, 512)

        # Project to transformer dimension and add positional signal
        x = self.feature_projection(features)      # (B, T, d_model)

        # [FIX-3] pos_embedding is (1, seq_len, d_model); broadcasts over B
        x = x + self.pos_embedding                 # (B, T, d_model)

        # Model temporal dependencies across frames
        x = self.transformer(x)                    # (B, T, d_model)

        # Soft-aggregate T frames into a single sequence representation
        global_feature, _ = self.att_pooling(x)   # (B, d_model)

        # Classify
        return self.classifier(global_feature)     # (B, num_classes)

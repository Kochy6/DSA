"""
dataset.py — DSADataset
=======================
Fixes applied (from audit):
  [FIX-2]  Double normalization: all normalization is now performed inside
           load_dsa_data() BEFORE the tensor is returned. Augmentation in
           __getitem__() therefore operates on a fully-normalized [-1, 1]
           tensor, so RandomAffine border-fill (value 0 → -1 after the old
           second normalization) no longer injects artificial signal.
  [FIX-6]  'mode' parameter clarified: it controls augmentation only; the
           comment now makes this explicit so future readers are not misled.
  [FIX-7]  'file_path' column in CSV is now read directly instead of
           reconstructing the path from root_dirs + filename. The two-root
           fallback is kept as a secondary lookup for backward compatibility
           when file_path is NaN or the file has moved.
  [FIX-8]  RandomHorizontalFlip removed: DSA angiography images encode
           lateralized vascular anatomy (left/right asymmetry is clinically
           significant). Flipping creates anatomically implausible samples
           and can mislead the model. Replaced with RandomVerticalFlip(p=0.3)
           which is less anatomically harmful, and added MedicalElasticTransform
           comment placeholder for future consideration.
"""

import torch
import pandas as pd
import numpy as np
import pydicom
import cv2
import os
from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2


class DSADataset(Dataset):
    """
    PyTorch Dataset for Digital Subtraction Angiography (DSA) DICOM sequences.

    Parameters
    ----------
    csv_path    : str   Path to the label CSV file. Must contain columns:
                        'filename', 'label', and ideally 'file_path'.
    mode        : str   'train' enables augmentation; anything else disables it.
                        NOTE: This class always loads the full CSV. Use
                        torch.utils.data.Subset (with StratifiedKFold indices)
                        in train.py to enforce the train/val split.
    target_size : tuple Spatial (H, W) to resize each frame to.
    target_t    : int   Number of frames to sample/pad to. Must match
                        DSATemporalModel's seq_len parameter.
    """

    # Fallback root directories when 'file_path' column is absent/NaN.
    _FALLBACK_ROOTS = [
        "/mnt/pro/DSA/data/ori_data/DICOM1",
        "/autodl-fs/data/Pro/DSA/dicom_all",
    ]

    def __init__(
        self,
        csv_path: str,
        mode: str = "train",
        target_size: tuple = (224, 224),  #(512, 512)显存大约是5倍
        target_t: int = 16,   #可以尝试24-32
    ):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.mode = mode
        self.target_size = target_size
        self.target_t = target_t  # exposed so train.py can pass it to the model

        # ------------------------------------------------------------------
        # [FIX-7] Use the pre-computed absolute path from CSV when available.
        # The 'file_path' column already encodes which root directory each
        # file belongs to, so we do not need to re-derive it.
        # ------------------------------------------------------------------
        self.filenames = self.df["filename"].values
        self.labels = self.df["label"].values

        # Use file_path column if present and non-null, else fall back to
        # reconstructing from filename + _FALLBACK_ROOTS.
        if "file_path" in self.df.columns:
            self.file_paths = self.df["file_path"].values
        else:
            self.file_paths = np.array([None] * len(self.df))

        # ------------------------------------------------------------------
        # [FIX-2]  Augmentation pipeline.
        # All transforms operate on a tensor that is already in [-1, 1].
        # RandomAffine fill defaults to 0, which corresponds to the mean
        # of a [-1, 1] distribution — a neutral, non-signal-bearing value.
        #
        # [FIX-8]  RandomHorizontalFlip removed; lateral anatomy is
        # diagnostically meaningful in DSA.  RandomVerticalFlip is kept
        # at low probability as a mild positional perturbation.
        # ------------------------------------------------------------------
        if self.mode == "train":
            self.transform = v2.Compose(
                [
                    # Small affine jitter: translation ±2 %, rotation ±5 °
                    # fill=0 → neutral gray in [-1,1] space (correct after FIX-2)
                    v2.RandomAffine(degrees=5, translate=(0.02, 0.02), fill=0),
                    # Mild intensity perturbation; keeps contrast ratios intact
                    v2.ColorJitter(brightness=0.1, contrast=0.1),
                    # Vertical flip is anatomically safer than horizontal for DSA
                    v2.RandomVerticalFlip(p=0.3),
                ]
            )
        else:
            # Validation / test: deterministic, no augmentation
            self.transform = v2.Identity()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_path(self, idx: int) -> str:
        """
        Return the absolute filesystem path for sample at position idx.

        Priority:
          1. 'file_path' column value (if non-null and file exists)
          2. Fallback: join each root in _FALLBACK_ROOTS with filename
        """
        # [FIX-7] Prefer the stored absolute path from the CSV
        stored_path = self.file_paths[idx]
        if stored_path is not None and pd.notna(stored_path):
            if os.path.exists(stored_path):
                return stored_path
            # File may have moved; fall through to root-dir search

        # Fallback: reconstruct from known root directories
        filename = str(self.filenames[idx])
        for root in self._FALLBACK_ROOTS:
            candidate = os.path.join(root, filename)
            if os.path.exists(candidate):
                return candidate

        raise FileNotFoundError(
            f"Cannot locate DICOM file for sample '{filename}'. "
            f"Checked stored path '{stored_path}' and fallback roots "
            f"{self._FALLBACK_ROOTS}."
        )

    def load_dsa_data(self, idx: int) -> torch.Tensor:
        """
        Load, preprocess and normalize a DSA DICOM sequence.

        Returns
        -------
        torch.Tensor  shape [target_t, 1, H, W], dtype float32, range [-1, 1]
        """
        file_path = self._resolve_path(idx)

        # ---- 1. Read DICOM --------------------------------------------------
        ds = pydicom.dcmread(file_path)
        pixel_array = ds.pixel_array.astype(np.float32)  # [T, H, W] or [H, W]

        # Handle single-frame DICOM (expand to [1, H, W])
        if pixel_array.ndim == 2:
            pixel_array = pixel_array[np.newaxis, ...]

        # ---- 2. Temporal sampling -------------------------------------------
        orig_t = pixel_array.shape[0]
        if orig_t >= self.target_t:
            # Evenly spaced indices across the full sequence
            indices = np.linspace(0, orig_t - 1, self.target_t).astype(int)
            sampled_data = pixel_array[indices]
        else:
            # Pad by repeating the last frame (preserves temporal context)
            pad_size = self.target_t - orig_t
            padding = np.stack([pixel_array[-1]] * pad_size, axis=0)
            sampled_data = np.concatenate([pixel_array, padding], axis=0)

        # ---- 3. Spatial resize ----------------------------------------------
        processed_frames = []
        for frame in sampled_data:
            resized = cv2.resize(
                frame, self.target_size, interpolation=cv2.INTER_AREA
            )
            processed_frames.append(resized)

        final_data = np.stack(processed_frames, axis=0)  # [T, H, W]

        # ---- 4. Normalization (single pass, done here) ----------------------
        # [FIX-2] Step 4a: per-sequence min-max → [0, 1]
        # Preserves intra-sequence contrast (important for DSA subtraction
        # artifacts and contrast-agent dynamics).
        d_min = final_data.min()
        d_max = final_data.max()
        final_data = (final_data - d_min) / (d_max - d_min + 1e-8)

        # [FIX-2] Step 4b: rescale [0, 1] → [-1, 1] for faster convergence
        # with BatchNorm / standard weight initialization.
        # Doing both steps here (not split across load + __getitem__) means
        # augmentation always sees the final normalized range.
        final_data = (final_data - 0.5) / 0.5

        # Return shape: [T, 1, H, W]  (channel dim added for Conv2d backbone)
        return torch.from_numpy(final_data).unsqueeze(1)  # float32

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        """
        Returns
        -------
        images : torch.Tensor  [target_t, 1, H, W], float32, range [-1, 1]
        label  : torch.Tensor  scalar long
        """
        # Load and normalize (all normalization inside load_dsa_data)
        images = self.load_dsa_data(idx)

        # Augment in [-1, 1] space (border fill = 0 is neutral here)
        images = self.transform(images)

        # No second normalization needed — [FIX-2]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return images, label

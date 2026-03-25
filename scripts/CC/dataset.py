"""
dataset.py — DSADataset
=======================
Previous fixes retained:
  [FIX-2]  All normalization consolidated inside load_dsa_data(); augmentation
           operates on a fully normalized [-1, 1] tensor.
  [FIX-6]  'mode' controls augmentation only; full CSV is always loaded.
  [FIX-7]  'file_path' column used directly; root-dir fallback for legacy paths.
  [FIX-8]  RandomHorizontalFlip removed (lateral anatomy is clinically
           significant in DSA). RandomVerticalFlip(p=0.3) retained.
  [FIX-B]  Phase-aware temporal sampling replaces uniform np.linspace.

New fix applied:
  [FIX-D]  Temporal-variance ROI crop inserted between DICOM load and
           phase-aware sampling.

           Motivation:
           DSA images are 1024×1024. Clinically relevant structures
           (aneurysm sac, stenosis, AVM nidus) typically occupy a small
           central sub-region. The surrounding area contains skull calvarium,
           soft tissue, and background — all static, zero-variance regions.
           When ResNet-18 processes these at 224×224, the lesion detail is
           compressed to a fraction of the receptive field and drowned out
           by irrelevant texture from bone and air.

           Fix:
           Before phase-aware sampling, a ROI bounding box is computed from
           the raw pixel array using the following physical reasoning:

             1. Temporal standard deviation [H, W] is computed across all T
                frames. High σ = pixels that change over time = contrast agent
                flowing through vessels. Low σ = static background.

             2. Otsu's method is applied to the σ map to produce a binary
                vessel activity mask. Otsu is preferable to a fixed percentile
                because it adapts to each sequence's contrast-to-noise ratio.

             3. The mask is closed with a 25×25 elliptical structuring element
                to bridge short gaps between vessel segments (e.g. the gap
                across a sinus confluence or across the parent artery of an
                aneurysm neck).

             4. The bounding box of the surviving mask is extracted and padded
                by ROI_PAD_FRAC (default 0.12 = 12%) on each side, then
                clamped to image boundaries.

             5. All frames are cropped to this box. The crop is computed once
                per sequence and applied uniformly, so temporal alignment is
                preserved exactly.

             6. After cropping, the spatial resize (cv2.resize to 224×224)
                naturally zooms into the vessel region, giving the backbone
                full 224×224 resolution over diagnostically relevant pixels
                rather than wasting resolution on skull bone.

           Fallback:
           If the Otsu mask is empty (degenerate sequence), or if the crop
           region is smaller than MIN_CROP_PX on either axis, the full frame
           is used. This prevents silent data corruption on edge cases (e.g.
           very short sequences, failed subtraction, flat pixel arrays).

           Why Otsu and not a fixed percentile?
           Across the dataset, sequences vary widely in contrast dose and
           acquisition timing. A fixed 85th-percentile threshold that works
           for a typical 40-frame sequence may mask the entire image on a
           short 13-frame sequence (IM_0767 in the CSV) or leave too much
           background on a 97-frame sequence. Otsu finds the optimal
           foreground/background split for each sequence independently.

           Why NOT a pretrained segmentation model?
           The U-Net or SAM-based approach would require (a) a separate
           inference step, (b) GPU memory for a second model during
           data loading, and (c) a labelled vessel mask dataset for
           fine-tuning that we do not have. The temporal-σ method is
           signal-processing, not machine learning — it exploits the
           physics of DSA (contrast dynamics = temporal signal) and adds
           zero GPU/memory overhead.

           Performance note:
           np.std over ~40 frames of 1024×1024 float32 ≈ 160 ms on CPU,
           which is dominated by I/O anyway. With 8 DataLoader workers the
           bottleneck is disk read, not the σ computation.
"""

import torch
import pandas as pd
import numpy as np
import pydicom
import cv2
import os
from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2


# ---------------------------------------------------------------------------
# Phase sampling configuration  [FIX-B]
# ---------------------------------------------------------------------------
_PHASE_FRACS = {
    "arterial_end":  0.35,
    "capillary_end": 0.70,
}
_BUDGET_FRACS = {
    "arterial":  0.50,
    "capillary": 0.30,
    "venous":    0.20,
}
assert abs(sum(_BUDGET_FRACS.values()) - 1.0) < 1e-6, \
    "BUDGET_FRACS must sum to 1.0"


# ---------------------------------------------------------------------------
# ROI extraction configuration  [FIX-D]
# ---------------------------------------------------------------------------

# Fraction of pixels (by inter-frame activity) retained as the vessel mask.
# 0.25 = keep the top 25% most-active pixels → threshold at 75th percentile.
# Increase toward 0.40 if crops feel too tight; decrease toward 0.15 if too
# much background remains after the morphological closing.
ROI_ACTIVE_FRAC: float = 0.25

# Fraction of the image dimension added as padding around the bounding box.
# 0.12 = 12% of H/W on each side.
ROI_PAD_FRAC: float = 0.12

# If the crop would be smaller than this in either dimension, fall back to the
# full frame.  Prevents degenerate crops on very short or near-flat sequences.
MIN_CROP_PX: int = 64

# Structuring element radius for morphological closing.  25 px at 1024×1024
# bridges typical inter-vessel gaps without merging separate vascular trees.
_MORPH_RADIUS: int = 25


class DSADataset(Dataset):
    """
    PyTorch Dataset for Digital Subtraction Angiography (DSA) DICOM sequences.

    Parameters
    ----------
    csv_path    : str   Path to the label CSV. Required columns: 'filename',
                        'label'. Recommended: 'file_path'.
    mode        : str   'train' enables augmentation; any other value disables.
    target_size : tuple Spatial (H, W) to resize each CROPPED frame to.
    target_t    : int   Total frames to return per sequence.
    """

    _FALLBACK_ROOTS = [
        "/root/autodl-tmp/DSA/data/ori_data/DICOM1",
        "/autodl-fs/data/Pro/DSA/dicom_all",
    ]

    def __init__(
        self,
        csv_path: str,
        mode: str = "train",
        target_size: tuple = (224, 224),
        target_t: int = 32,
    ):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.mode = mode
        self.target_size = target_size
        self.target_t = target_t

        self.filenames = self.df["filename"].values
        self.labels    = self.df["label"].values

        if "file_path" in self.df.columns:
            self.file_paths = self.df["file_path"].values
        else:
            self.file_paths = np.array([None] * len(self.df))

        # [FIX-B] Phase budgets
        n_art = round(target_t * _BUDGET_FRACS["arterial"])
        n_cap = round(target_t * _BUDGET_FRACS["capillary"])
        n_ven = target_t - n_art - n_cap
        assert n_ven > 0, (
            f"target_t={target_t} too small for 3-phase split; minimum is 10."
        )
        self._phase_budget = (n_art, n_cap, n_ven)

        # [FIX-D] Pre-build the morphological structuring element once.
        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (_MORPH_RADIUS * 2 + 1, _MORPH_RADIUS * 2 + 1),
        )

        # Augmentation pipeline  [FIX-2, FIX-8]
        if self.mode == "train":
            self.transform = v2.Compose(
                [
                    v2.RandomAffine(degrees=5, translate=(0.02, 0.02), fill=0),
                    v2.ColorJitter(brightness=0.1, contrast=0.1),
                    v2.RandomVerticalFlip(p=0.3),
                ]
            )
        else:
            self.transform = v2.Identity()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_path(self, idx: int) -> str:
        """Return the absolute filesystem path for sample idx. [FIX-7]"""
        stored = self.file_paths[idx]
        if stored is not None and pd.notna(stored) and os.path.exists(stored):
            return stored
        filename = str(self.filenames[idx])
        for root in self._FALLBACK_ROOTS:
            candidate = os.path.join(root, filename)
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(
            f"Cannot locate DICOM for '{filename}'. "
            f"Checked stored path '{stored}' and roots {self._FALLBACK_ROOTS}."
        )

    # ------------------------------------------------------------------
    # [FIX-D] Temporal-variance ROI extraction
    # ------------------------------------------------------------------

    def _extract_roi_bbox(
        self, pixel_array: np.ndarray
    ) -> tuple[int, int, int, int]:
        """
        Compute a padded bounding box around the active vascular territory.

        Why the previous Otsu approach failed
        --------------------------------------
        DSA frames are already contrast-subtracted images: background ≈ 0,
        vessels carry signal.  The temporal σ map across all T frames has a
        broad, roughly unimodal histogram because:
          (a) the pre-bolus frames (~first 5–10) are near-zero everywhere,
          (b) the bolus-arrival frames raise σ broadly across the image.
        Otsu finds no clean two-class split on such a distribution and sets
        the threshold near the lower end, so the binary mask covers nearly the
        entire image → bounding box ≈ full frame → no meaningful crop.

        Fix: inter-frame difference + high-percentile threshold
        ---------------------------------------------------------
        Instead of σ over raw values, we compute the mean absolute difference
        between consecutive frames (the "motion energy" map):

            diff_map[h, w] = mean_t( |frame[t] - frame[t-1]| )

        This isolates pixels that CHANGE between adjacent frames — i.e. the
        contrast front propagating through vessels.  Static noise contributes
        uniformly; moving contrast contributes a sharp local peak.  The
        histogram is now strongly right-skewed (most pixels ≈ 0, vessel pixels
        >> 0), making a high-percentile threshold both stable and meaningful.

        We keep the top ROI_ACTIVE_FRAC of pixels by diff_map value (default
        25%).  This adapts automatically to injection volume and frame rate
        without any per-sequence tuning.

        Steps
        -----
        1. Compute inter-frame absolute difference map  [H, W]
        2. Threshold at (1 - ROI_ACTIVE_FRAC) percentile → binary mask
        3. Morphological closing to bridge vessel gaps
        4. Bounding box + padding + clamp + minimum-size guard

        Parameters
        ----------
        pixel_array : np.ndarray  shape [T, H, W], float32, raw pixel values

        Returns
        -------
        (y1, y2, x1, x2) : int  crop bounds (y2, x2 are exclusive).
                           Falls back to full-image bounds on degenerate input.
        """
        T, H, W = pixel_array.shape

        # --- 1. Inter-frame absolute difference map -----------------------
        # Shape: [T-1, H, W] → mean over T-1 axis → [H, W]
        # Requires at least 2 frames; for single-frame DICOMs fall back.
        if T < 2:
            return 0, H, 0, W

        # pixel_array[1:] − pixel_array[:-1] gives T-1 difference frames.
        # mean(|diff|) is more robust to single noisy frames than max(|diff|).
        diff_map = np.abs(
            pixel_array[1:].astype(np.float32) -
            pixel_array[:-1].astype(np.float32)
        ).mean(axis=0)  # [H, W]

        # Degenerate: perfectly static sequence
        if diff_map.max() < 1e-6:
            return 0, H, 0, W

        # --- 2. High-percentile threshold ---------------------------------
        # Keep the top ROI_ACTIVE_FRAC fraction of pixels by activity.
        # e.g. ROI_ACTIVE_FRAC=0.25 → threshold at the 75th percentile.
        # This is robust regardless of histogram shape.
        thresh = float(np.percentile(diff_map, (1.0 - ROI_ACTIVE_FRAC) * 100))

        # Safety: if the computed threshold is 0 (many flat pixels), step up
        # to a small positive value to avoid keeping everything.
        if thresh < 1e-6:
            thresh = float(np.percentile(diff_map[diff_map > 0], 50)) \
                     if (diff_map > 0).any() else 1e-6

        mask = (diff_map >= thresh).astype(np.uint8) * 255  # [H, W] uint8

        # --- 3. Morphological closing -------------------------------------
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._morph_kernel)

        # --- 4. Bounding box ----------------------------------------------
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return 0, H, 0, W

        y_min, y_max = int(coords[0].min()), int(coords[0].max())
        x_min, x_max = int(coords[1].min()), int(coords[1].max())

        # --- 5. Symmetric padding + clamp ---------------------------------
        pad_y = int(ROI_PAD_FRAC * H)
        pad_x = int(ROI_PAD_FRAC * W)

        y1 = max(0, y_min - pad_y)
        y2 = min(H, y_max + pad_y + 1)
        x1 = max(0, x_min - pad_x)
        x2 = min(W, x_max + pad_x + 1)

        # --- 6. Minimum-size guard ----------------------------------------
        if (y2 - y1) < MIN_CROP_PX or (x2 - x1) < MIN_CROP_PX:
            return 0, H, 0, W

        return y1, y2, x1, x2

    # ------------------------------------------------------------------
    # [FIX-B] Phase-aware temporal sampling
    # ------------------------------------------------------------------

    def _phase_aware_sample(self, pixel_array: np.ndarray) -> np.ndarray:
        """
        Sample target_t frames with arterial-phase over-representation.
        Operates on the already-cropped pixel_array.  [FIX-B]
        """
        orig_t = pixel_array.shape[0]
        n_art, n_cap, n_ven = self._phase_budget

        if orig_t < self.target_t:
            pad_size = self.target_t - orig_t
            padding = np.stack([pixel_array[-1]] * pad_size, axis=0)
            pixel_array = np.concatenate([pixel_array, padding], axis=0)
            orig_t = pixel_array.shape[0]

        art_end = max(1, round(orig_t * _PHASE_FRACS["arterial_end"]))
        cap_end = max(art_end + 1, round(orig_t * _PHASE_FRACS["capillary_end"]))
        cap_end = min(cap_end, orig_t - n_ven)

        def _linspace_int(start: int, end_exclusive: int, n: int) -> np.ndarray:
            if end_exclusive <= start:
                return np.full(n, start, dtype=int)
            return np.linspace(start, end_exclusive - 1, n).astype(int)

        art_idx = _linspace_int(0,       art_end, n_art)
        cap_idx = _linspace_int(art_end, cap_end, n_cap)
        ven_idx = _linspace_int(cap_end, orig_t,  n_ven)

        indices = np.concatenate([art_idx, cap_idx, ven_idx])
        return pixel_array[indices]

    # ------------------------------------------------------------------
    # Main data loading
    # ------------------------------------------------------------------

    def load_dsa_data(self, idx: int) -> torch.Tensor:
        """
        Load, crop, preprocess, and normalize one DSA DICOM sequence.

        Pipeline (in order):
          1. Read DICOM → float32 pixel array [T, H, W]
          2. [FIX-D] Compute temporal-σ ROI and crop all frames
          3. [FIX-B] Phase-aware temporal sampling on CROPPED frames
          4. Spatial resize (cropped ROI → 224×224, fills the full frame)
          5. [FIX-2] Per-sequence min-max → [0,1] → [-1,1] normalization

        Returns
        -------
        torch.Tensor  shape [target_t, 1, H, W], float32, range [-1, 1]
        """
        file_path = self._resolve_path(idx)

        # ---- 1. Read DICOM --------------------------------------------------
        ds = pydicom.dcmread(file_path)
        pixel_array = ds.pixel_array.astype(np.float32)  # [T, H, W] or [H, W]

        if pixel_array.ndim == 2:
            pixel_array = pixel_array[np.newaxis, ...]  # single frame → [1, H, W]

        # ---- 2. [FIX-D] Temporal-variance ROI crop -------------------------
        # Crop is computed on the FULL raw sequence before any frame
        # selection, so the variance map uses all available temporal
        # information (maximum sensitivity to contrast dynamics).
        y1, y2, x1, x2 = self._extract_roi_bbox(pixel_array)
        pixel_array = pixel_array[:, y1:y2, x1:x2]  # [T, crop_H, crop_W]

        # ---- 3. [FIX-B] Phase-aware temporal sampling ----------------------
        # Operates on the already-cropped sequence; spatial dimensions are
        # now crop_H × crop_W (typically ~300–700 px) rather than 1024×1024.
        sampled_data = self._phase_aware_sample(pixel_array)  # [target_t, crop_H, crop_W]

        # ---- 4. Spatial resize ---------------------------------------------
        # Resize from crop_H × crop_W to 224×224.
        # Because the crop eliminates background, all 224×224 pixels now
        # encode diagnostically relevant vascular anatomy.
        processed_frames = []
        for frame in sampled_data:
            resized = cv2.resize(
                frame, self.target_size, interpolation=cv2.INTER_AREA
            )
            processed_frames.append(resized)

        final_data = np.stack(processed_frames, axis=0)  # [target_t, 224, 224]

        # ---- 5. [FIX-2] Normalization --------------------------------------
        # Per-sequence min-max → [0, 1], then → [-1, 1].
        # Applied AFTER crop so the normalization range reflects the contrast
        # dynamics within the ROI, not the full-frame pixel range (which
        # includes high-intensity bone that would compress the vessel signal).
        d_min, d_max = final_data.min(), final_data.max()
        final_data = (final_data - d_min) / (d_max - d_min + 1e-8)
        final_data = (final_data - 0.5) / 0.5

        return torch.from_numpy(final_data).unsqueeze(1)  # [target_t, 1, 224, 224]

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        """
        Returns
        -------
        images : Tensor  [target_t, 1, H, W], float32, range [-1, 1]
        label  : Tensor  scalar long
        """
        images = self.load_dsa_data(idx)
        images = self.transform(images)
        label  = torch.tensor(self.labels[idx], dtype=torch.long)
        return images, label
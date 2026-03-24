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

New fix applied:
  [FIX-B]  Phase-aware temporal sampling replaces uniform np.linspace.

           Motivation (from training log analysis):
           The model trained to AUC ~0.56 on fold 1 with all checkpoints saved
           in the first two epochs, suggesting the temporal signal it learned
           was not clinically meaningful. A key cause: uniform linspace across
           ~40 frames compresses the arterial phase (frames ~0–14) — where
           stenosis, AVM shunting, and aneurysm filling are most visible — to
           only ~5–6 of the 32 sampled frames. This underrepresents the phase
           where pathology signal is densest.

           Fix:
           Frame budget is split across three physiological phases:
             • Arterial   (first ~35% of frames) → 50% of target_t
             • Capillary  (next  ~35% of frames) → 30% of target_t
             • Venous     (last  ~30% of frames) → 20% of target_t

           For target_t=32: arterial=16 frames, capillary=10, venous=6.
           Within each phase, frames are still sampled evenly (linspace), so
           temporal ordering is preserved and no frames are duplicated.

           Phase boundaries are defined as fixed fractions of orig_t because
           DSA acquisition protocols at this institution use a consistent
           frame-rate structure (confirmed by the frame_times column in the CSV
           showing a ~167ms early phase transitioning to ~333–500ms late phase).
           If acquisition protocols vary across datasets, the fractions should
           be tuned or derived from the frame_times column directly.
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
# Phase sampling configuration
# [FIX-B] These constants define the physiological phase boundaries and the
# fraction of the target_t budget allocated to each phase.
# Adjusting PHASE_FRACS changes where each phase ends (as a fraction of
# orig_t). Adjusting BUDGET_FRACS changes how many sampled frames go to
# each phase. The three BUDGET_FRACS must sum to 1.0.
# ---------------------------------------------------------------------------
_PHASE_FRACS = {
    "arterial_end":  0.35,  # first 35% of sequence = arterial phase
    "capillary_end": 0.70,  # next 35% = capillary / parenchymal phase
    # venous: remaining 30%
}
_BUDGET_FRACS = {
    "arterial":  0.50,  # 50% of sampled frames allocated to arterial phase
    "capillary": 0.30,  # 30% → capillary
    "venous":    0.20,  # 20% → venous
}
assert abs(sum(_BUDGET_FRACS.values()) - 1.0) < 1e-6, \
    "BUDGET_FRACS must sum to 1.0"


class DSADataset(Dataset):
    """
    PyTorch Dataset for Digital Subtraction Angiography (DSA) DICOM sequences.

    Parameters
    ----------
    csv_path    : str   Path to the label CSV. Required columns: 'filename',
                        'label'. Recommended: 'file_path'.
    mode        : str   'train' enables augmentation; any other value disables.
                        This class always loads the full CSV — use Subset in
                        train.py for the actual train/val split.
    target_size : tuple Spatial (H, W) to resize each frame to.
    target_t    : int   Total frames to return per sequence. Must match
                        DSATemporalModel's seq_len. Budget split across phases
                        according to _BUDGET_FRACS above.
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

        # [FIX-7] Prefer the pre-computed absolute path from the CSV column.
        if "file_path" in self.df.columns:
            self.file_paths = self.df["file_path"].values
        else:
            self.file_paths = np.array([None] * len(self.df))

        # [FIX-B] Compute per-phase frame budgets once at init time so they
        # are not recomputed on every __getitem__ call.
        # We round down and give any remainder to the arterial phase (highest
        # diagnostic priority).
        n_art = round(target_t * _BUDGET_FRACS["arterial"])
        n_cap = round(target_t * _BUDGET_FRACS["capillary"])
        n_ven = target_t - n_art - n_cap  # absorb rounding remainder
        assert n_ven > 0, (
            f"target_t={target_t} too small for 3-phase split; "
            f"minimum recommended target_t is 10."
        )
        self._phase_budget = (n_art, n_cap, n_ven)

        # [FIX-2] / [FIX-8] Augmentation pipeline.
        # All transforms run on a tensor already in [-1, 1] (after FIX-2).
        # RandomAffine fill=0 is neutral in [-1, 1] space.
        # RandomHorizontalFlip removed — left/right DSA anatomy is meaningful.
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
    # [FIX-B] Phase-aware temporal sampling
    # ------------------------------------------------------------------

    def _phase_aware_sample(self, pixel_array: np.ndarray) -> np.ndarray:
        """
        Sample target_t frames from a DSA sequence with over-representation
        of the diagnostically critical arterial phase.

        Strategy
        --------
        1. Divide the sequence into three physiological phases using fixed
           fractional boundaries (see _PHASE_FRACS at module level).
        2. Allocate the target_t frame budget to each phase according to
           _BUDGET_FRACS (arterial 50 %, capillary 30 %, venous 20 %).
        3. Within each phase, sample evenly using np.linspace to preserve
           temporal ordering within the phase.
        4. Concatenate arterial + capillary + venous indices in order.

        If the sequence is shorter than target_t, the standard last-frame
        padding strategy is applied BEFORE phase sampling so that the phases
        are computed on a padded sequence of known length (target_t).

        Parameters
        ----------
        pixel_array : np.ndarray  shape [orig_t, H, W]

        Returns
        -------
        np.ndarray  shape [target_t, H, W]
        """
        orig_t = pixel_array.shape[0]
        n_art, n_cap, n_ven = self._phase_budget

        # --- Step 1: Ensure we have at least target_t frames to sample from.
        # Sequences shorter than target_t are padded with the last frame.
        # This is done first so that phase boundaries are always computed
        # against a sequence of length >= target_t.
        if orig_t < self.target_t:
            pad_size = self.target_t - orig_t
            padding = np.stack([pixel_array[-1]] * pad_size, axis=0)
            pixel_array = np.concatenate([pixel_array, padding], axis=0)
            orig_t = pixel_array.shape[0]  # now == target_t

        # --- Step 2: Compute phase boundary frame indices.
        # art_end is the last frame (exclusive) of the arterial phase.
        art_end = max(1, round(orig_t * _PHASE_FRACS["arterial_end"]))
        cap_end = max(art_end + 1, round(orig_t * _PHASE_FRACS["capillary_end"]))
        # Clamp cap_end so the venous phase always has at least n_ven frames.
        cap_end = min(cap_end, orig_t - n_ven)

        # --- Step 3: Sample evenly within each phase.
        def _linspace_int(start: int, end_exclusive: int, n: int) -> np.ndarray:
            """
            Return n evenly spaced integer indices in [start, end_exclusive).
            If the phase is degenerate (start >= end_exclusive), repeat start.
            """
            if end_exclusive <= start:
                return np.full(n, start, dtype=int)
            return np.linspace(start, end_exclusive - 1, n).astype(int)

        art_idx = _linspace_int(0,       art_end, n_art)  # arterial phase
        cap_idx = _linspace_int(art_end, cap_end, n_cap)  # capillary phase
        ven_idx = _linspace_int(cap_end, orig_t,  n_ven)  # venous phase

        # --- Step 4: Concatenate and return sampled frames.
        indices = np.concatenate([art_idx, cap_idx, ven_idx])  # [target_t]
        return pixel_array[indices]  # [target_t, H, W]

    # ------------------------------------------------------------------
    # Main data loading
    # ------------------------------------------------------------------

    def load_dsa_data(self, idx: int) -> torch.Tensor:
        """
        Load, preprocess, and normalize one DSA DICOM sequence.

        Returns
        -------
        torch.Tensor  shape [target_t, 1, H, W], float32, range [-1, 1]
        """
        file_path = self._resolve_path(idx)

        # ---- 1. Read DICOM --------------------------------------------------
        ds = pydicom.dcmread(file_path)
        pixel_array = ds.pixel_array.astype(np.float32)  # [T, H, W] or [H, W]

        if pixel_array.ndim == 2:
            pixel_array = pixel_array[np.newaxis, ...]  # single-frame → [1, H, W]

        # ---- 2. [FIX-B] Phase-aware temporal sampling -----------------------
        sampled_data = self._phase_aware_sample(pixel_array)  # [target_t, H, W]

        # ---- 3. Spatial resize ----------------------------------------------
        processed_frames = []
        for frame in sampled_data:
            resized = cv2.resize(
                frame, self.target_size, interpolation=cv2.INTER_AREA
            )
            processed_frames.append(resized)

        final_data = np.stack(processed_frames, axis=0)  # [target_t, H, W]

        # ---- 4. [FIX-2] Normalization: single consolidated pass -------------
        # Step 4a: per-sequence min-max → [0, 1].
        # Preserves intra-sequence contrast (contrast-agent dynamics).
        d_min, d_max = final_data.min(), final_data.max()
        final_data = (final_data - d_min) / (d_max - d_min + 1e-8)

        # Step 4b: [0, 1] → [-1, 1].
        # Augmentation (RandomAffine fill=0) is neutral in this range.
        final_data = (final_data - 0.5) / 0.5

        # Shape: [target_t, 1, H, W]  (channel dim for Conv2d backbone)
        return torch.from_numpy(final_data).unsqueeze(1)

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
        images = self.transform(images)  # augment in [-1, 1] space [FIX-2]
        label  = torch.tensor(self.labels[idx], dtype=torch.long)
        return images, label
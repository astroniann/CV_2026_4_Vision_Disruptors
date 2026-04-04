# PyTorch Dataset for BraTS brain tumour data, aligned with cwdm's BRATSVolumes.
#
# Key design decisions matching cwdm (guided_diffusion/bratsloader.py):
#   - __getitem__ returns {'t1n', 't1c', 't2w', 't2f'} as separate (1, H, W, D) tensors
#   - Normalisation is clip-and-normalize to [0, 1] (not Z-score)
#   - No DWT is done here — cwdm does DWT inside gaussian_diffusion.training_losses()
#   - No target/condition split here — cwdm does that inside training_losses() via `contr` arg
#   - Volumes are zero-padded to (1, 240, 240, 160) then centre-cropped to (1, 224, 224, 160)
#     matching cwdm's exact spatial handling
#
# Modality dropout (for training robustness / sample_auto.py compatibility):
#   - When dropout_modality=True, one modality is randomly zeroed out per sample.
#   - The dropped modality name is stored in sample['missing'], exactly matching the
#     convention in cwdm's BRATSVolumes and consumed by sample_auto.py.
#   - The zeroed tensor is torch.zeros(1) to match cwdm's convention (not full-size zeros),
#     so downstream code that checks `if missing != 'none'` works without changes.
#   - During validation/evaluation, set dropout_modality=False (default) to keep all modalities.

import random as _random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .patient import load_patient


# Center crop to remove background (240 -> 192)
CROP_H, CROP_W = 192, 192
# Target spatial resolution
OUT_SIZE = (96, 96, 96)

# All four modality keys in a fixed order so random choice is reproducible
_MODALITIES = ['t1n', 't1c', 't2w', 't2f']


class BraTS20Dataset(Dataset):
    """
    Parameters
    ----------
    data_root         : root folder; patient subfolders sit directly inside
                        (e.g. data_root/BraTS-GLI-00000-000/)
    csv_path          : optional path to an Excel/CSV file with a 'BraTS Subject ID'
                        column; if given, only listed patients are loaded
    split             : 'train' | 'validation' | 'additional'
                        Controls whether segmentation masks are loaded.
    dropout_modality  : if True, randomly zero out one modality per sample and
                        set sample['missing'] to its name. Designed for training
                        to simulate missing-modality scenarios and for compatibility
                        with sample_auto.py which routes on batch['missing'].
                        Default False — all four modalities are always returned.

    Returns (per __getitem__)
    -------
    dict with keys:
        't1n'        : torch.FloatTensor (1, 224, 224, 160)  range [0, 1]
                       or torch.zeros(1) if this modality was dropped
        't1c'        : torch.FloatTensor (1, 224, 224, 160)  range [0, 1]
                       or torch.zeros(1) if this modality was dropped
        't2w'        : torch.FloatTensor (1, 224, 224, 160)  range [0, 1]
                       or torch.zeros(1) if this modality was dropped
        't2f'        : torch.FloatTensor (1, 224, 224, 160)  range [0, 1]
                       or torch.zeros(1) if this modality was dropped
        'missing'    : str  name of dropped modality, or 'none'
        'patient_id' : str
        'subj'       : str  same as patient_id — kept for sample.py / sample_auto.py
                       compatibility (those scripts do batch['subj'][0].split(...))
        'seg'        : torch.LongTensor (H, W, D)  only when split in {'train','validation'}
    """

    HAS_SEG = {"train", "validation"}

    SPLIT_DIRS = {
        "train":      "BraTS2024-BraTS-GLI-TrainingData/training_data1_v2",
        "validation": "BraTS2024-BraTS-GLI-ValidationData/validation_data",
        "additional": "BraTS2024-BraTS-GLI-AdditionalTrainingData/training_data_additional",
    }

    def __init__(
        self,
        data_root: str,
        csv_path: Optional[str] = None,
        split: str = "train",
        dropout_modality: bool = False,
    ):
        assert split in self.SPLIT_DIRS, \
            f"split must be one of {list(self.SPLIT_DIRS)}, got '{split}'"

        self.split            = split
        self.has_seg          = split in self.HAS_SEG
        self.dropout_modality = dropout_modality

        split_dir = Path(data_root) / self.SPLIT_DIRS[split]
        assert split_dir.exists(), (
            f"Split directory not found: {split_dir}\n"
            f"Check that data_root='{data_root}' points to the BraTS2024 root."
        )
        self.patient_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])

        if csv_path and Path(csv_path).exists():
            df        = pd.read_excel(csv_path)
            valid_ids = set(df["BraTS Subject ID"].astype(str))
            self.patient_dirs = [
                d for d in self.patient_dirs if d.name in valid_ids
            ]

        assert len(self.patient_dirs) > 0, (
            f"No patient directories found under {split_dir}. "
            "Check your folder structure."
        )
        print(
            f"[BraTS20Dataset | {split}] {len(self.patient_dirs)} patients loaded."
            + (" (modality dropout ON)" if dropout_modality else "")
        )

    # ------------------------------------------------------------------
    # Internal helper: pad + crop to match cwdm spatial handling
    # ------------------------------------------------------------------

    @staticmethod
    def _pad_and_crop(vol_np: np.ndarray, is_seg: bool = False) -> torch.Tensor:
        """
        vol_np : (1, H, W, D) or (H, W, D) raw loaded volume.

        Returns torch.Tensor (1, 96, 96, 96) for modalities
        or (96, 96, 96) for segmentation.
        """
        if is_seg:
            H, W, D = vol_np.shape
            h_start = max(0, (H - CROP_H) // 2)
            w_start = max(0, (W - CROP_W) // 2)
            
            t = torch.from_numpy(vol_np).float()
            cropped = t[h_start:h_start+CROP_H, w_start:w_start+CROP_W, :]
            
            # Interpolate expects (1, 1, H, W, D) for 3D interpolation
            cropped = cropped.unsqueeze(0).unsqueeze(0)
            resized = F.interpolate(cropped, size=OUT_SIZE, mode='nearest')
            return resized.squeeze(0).squeeze(0).long()
        else:
            _, H, W, D = vol_np.shape
            h_start = max(0, (H - CROP_H) // 2)
            w_start = max(0, (W - CROP_W) // 2)
            
            t = torch.from_numpy(vol_np).float()
            cropped = t[:, h_start:h_start+CROP_H, w_start:w_start+CROP_W, :]
            
            # Interpolate expects (1, 1, H, W, D) 
            cropped = cropped.unsqueeze(0)
            resized = F.interpolate(cropped, size=OUT_SIZE, mode='trilinear', align_corners=False)
            return resized.squeeze(0)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.patient_dirs)

    def __getitem__(self, idx: int) -> dict:
        patient_dir = self.patient_dirs[idx]

        # load_patient returns a dict of (1, H, W, D) np arrays + optional seg
        modality_vols, seg = load_patient(str(patient_dir), load_seg=self.has_seg)

        # Build base sample with all four modalities
        sample = {
            't1n':        self._pad_and_crop(modality_vols['t1n']),
            't1c':        self._pad_and_crop(modality_vols['t1c']),
            't2w':        self._pad_and_crop(modality_vols['t2w']),
            't2f':        self._pad_and_crop(modality_vols['t2f']),
            'missing':    'none',
            'patient_id': patient_dir.name,
            # 'subj' mirrors cwdm's BRATSVolumes key; sample.py and sample_auto.py
            # do batch['subj'][0].split(...) so we must provide it.
            'subj':       str(patient_dir),
        }

        # ------------------------------------------------------------------
        # Modality dropout — randomly zero one modality and record which one.
        # Mirrors the offline dropout_modality.py but done on-the-fly so the
        # model sees every modality missing roughly equally over training.
        # The zeroed tensor is torch.zeros(1) to match cwdm's BRATSVolumes
        # convention (not a full-size zero volume), so downstream code that
        # checks `if missing != 'none'` or `batch[missing]` still works.
        # ------------------------------------------------------------------
        if self.dropout_modality:
            drop = _random.choice(_MODALITIES)
            sample[drop]      = torch.zeros(1)
            sample['missing'] = drop

        if seg is not None:
            sample['seg'] = self._pad_and_crop(seg, is_seg=True)

        return sample


# PyTorch Dataset for BraTS20 brain tumour segmentation.


from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .patient import load_patient


class BraTS20Dataset(Dataset):
    """
    Parameters
    ----------
    data_root : root folder containing train/ validation/ additional/
    csv_path  : path to brats20_data.csv (or None to load all patients found)
    split     : 'train' | 'validation' | 'additional'
    """

    HAS_SEG = {"train", "validation"}   # splits that carry ground-truth masks

    def __init__(
        self,
        data_root: str,
        csv_path: Optional[str],
        split: str = "train",
    ):
        assert split in {"train", "validation", "additional"}, \
            f"split must be 'train', 'validation', or 'additional', got '{split}'"

        self.split   = split
        self.has_seg = split in self.HAS_SEG

        SPLIT_DIRS = {
            "train":      "BraTS2024-BraTS-GLI-TrainingData/training_data1_v2",
            "validation": "BraTS2024-BraTS-GLI-ValidationData/validation_data",
            "additional": "BraTS2024-BraTS-GLI-AdditionalTrainingData/training_data_additional",
        }
        split_dir = Path(data_root) / SPLIT_DIRS[split]
        self.patient_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])

        if csv_path and Path(csv_path).exists():
            df        = pd.read_excel(csv_path)
            valid_ids = set(df["BraTS Subject ID"].astype(str))
            self.patient_dirs = [
                d for d in self.patient_dirs if d.name in valid_ids
            ]

        print(f"[{split}] {len(self.patient_dirs)} patients loaded.")

    def __len__(self) -> int:
        return len(self.patient_dirs)

    def __getitem__(self, idx: int) -> dict:
        patient_dir = self.patient_dirs[idx]

        image, seg = load_patient(str(patient_dir), load_seg=self.has_seg)

        sample = {
            "image":      torch.from_numpy(image).float(),   # (4, H, W, D)
            "patient_id": patient_dir.name,
        }

        if seg is not None:
            sample["seg"] = torch.from_numpy(seg).long()     # (H, W, D)

        return sample
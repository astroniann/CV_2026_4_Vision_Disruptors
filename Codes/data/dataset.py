"""
BraTS 2024 Synthesis Dataset Loader
"""

import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BraTSDataset(Dataset):
    """
    BraTS 2024 Synthesis Dataset
    
    Expected folder structure:
    root_dir/
    ├── BraTS-GLI-00000-000/
    │   ├── BraTS-GLI-00000-000-t1n.nii.gz
    │   ├── BraTS-GLI-00000-000-t1c.nii.gz
    │   ├── BraTS-GLI-00000-000-t2w.nii.gz
    │   ├── BraTS-GLI-00000-000-t2f.nii.gz
    │   └── BraTS-GLI-00000-000-seg.nii.gz
    ├── BraTS-GLI-00001-000/
    │   └── ...
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        train_ratio: float = 0.8,
        seed: int = 42,
        transforms=None,
        missing_modality: Optional[str] = None,
        load_seg: bool = True
    ):
        """
        Args:
            root_dir: Path to BraTS dataset
            split: 'train' or 'val'
            train_ratio: Ratio of training data
            seed: Random seed for splitting
            transforms: Data augmentation transforms
            missing_modality: Which modality to synthesize ('t1', 't2', 'flair', 't1ce')
            load_seg: Whether to load segmentation masks
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transforms = transforms
        self.missing_modality = missing_modality
        self.load_seg = load_seg
        
        # Modality mapping (BraTS naming -> our naming)
        self.modality_map = {
            't1': 't1n',
            't1ce': 't1c',
            't2': 't2w',
            'flair': 't2f'
        }
        
        # Get all case directories
        self.case_dirs = self._get_case_dirs()
        
        # Split into train/val
        np.random.seed(seed)
        indices = np.random.permutation(len(self.case_dirs))
        split_idx = int(len(self.case_dirs) * train_ratio)
        
        if split == 'train':
            self.case_dirs = [self.case_dirs[i] for i in indices[:split_idx]]
        else:
            self.case_dirs = [self.case_dirs[i] for i in indices[split_idx:]]
        
        logger.info(f"Loaded {len(self.case_dirs)} cases for {split} split")
    
    def _get_case_dirs(self) -> List[Path]:
        """Get all valid case directories"""
        case_dirs = []
        
        for case_dir in sorted(self.root_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            
            # Check if all required files exist
            required_files = [
                f"{case_dir.name}-{self.modality_map['t1']}.nii.gz",
                f"{case_dir.name}-{self.modality_map['t1ce']}.nii.gz",
                f"{case_dir.name}-{self.modality_map['t2']}.nii.gz",
                f"{case_dir.name}-{self.modality_map['flair']}.nii.gz",
            ]
            
            if self.load_seg:
                required_files.append(f"{case_dir.name}-seg.nii.gz")
            
            if all((case_dir / f).exists() for f in required_files):
                case_dirs.append(case_dir)
        
        return case_dirs
    
    def _load_nifti(self, filepath: Path) -> np.ndarray:
        """Load NIfTI file and return numpy array"""
        nii = nib.load(str(filepath))
        data = nii.get_fdata().astype(np.float32)
        return data
    
    def _preprocess(self, volume: np.ndarray) -> np.ndarray:
        """
        Preprocessing pipeline:
        1. Clip outliers
        2. Z-score normalization
        """
        # Clip outliers (brain tissue only, ignore background)
        brain_mask = volume > 0
        
        if brain_mask.sum() > 0:
            p_low = np.percentile(volume[brain_mask], 0.5)
            p_high = np.percentile(volume[brain_mask], 99.5)
            volume = np.clip(volume, p_low, p_high)
            
            # Z-score normalization
            mean = volume[brain_mask].mean()
            std = volume[brain_mask].std()
            
            if std > 0:
                volume[brain_mask] = (volume[brain_mask] - mean) / std
        
        return volume
    
    def __len__(self) -> int:
        return len(self.case_dirs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary with keys:
            - 't1': T1 volume
            - 't2': T2 volume
            - 'flair': FLAIR volume
            - 't1ce': T1ce volume
            - 'seg': Segmentation mask (if load_seg=True)
            - 'case_id': Case identifier
        """
        case_dir = self.case_dirs[idx]
        case_name = case_dir.name
        
        # Load all modalities
        modalities = {}
        for mod in ['t1', 't2', 'flair', 't1ce']:
            filepath = case_dir / f"{case_name}-{self.modality_map[mod]}.nii.gz"
            volume = self._load_nifti(filepath)
            volume = self._preprocess(volume)
            modalities[mod] = volume
        
        # Load segmentation if needed
        if self.load_seg:
            seg_path = case_dir / f"{case_name}-seg.nii.gz"
            seg = self._load_nifti(seg_path)
            modalities['seg'] = seg
        
        # Apply transforms
        if self.transforms:
            modalities = self.transforms(modalities)
        
        # Convert to torch tensors and add channel dimension
        sample = {}
        for key, value in modalities.items():
            if key != 'seg':
                # Add channel dimension: (D, H, W) -> (1, D, H, W)
                sample[key] = torch.from_numpy(value).unsqueeze(0).float()
            else:
                sample[key] = torch.from_numpy(value).long()
        
        sample['case_id'] = case_name
        
        return sample


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for DataLoader"""
    keys = batch[0].keys()
    collated = {}
    
    for key in keys:
        if key == 'case_id':
            collated[key] = [item[key] for item in batch]
        else:
            collated[key] = torch.stack([item[key] for item in batch], dim=0)
    
    return collated
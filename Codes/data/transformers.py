"""
Data augmentation transforms for BraTS dataset
"""

import numpy as np
import torch
from typing import Dict
import random


class RandomFlip3D:
    """Randomly flip 3D volume along specified axes"""
    
    def __init__(self, axes=(0, 1, 2), p=0.5):
        """
        Args:
            axes: Tuple of axes to flip
            p: Probability of applying flip
        """
        self.axes = axes
        self.p = p
    
    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if random.random() < self.p:
            axis = random.choice(self.axes)
            
            for key in sample.keys():
                if key != 'case_id':
                    sample[key] = np.flip(sample[key], axis=axis).copy()
        
        return sample


class RandomRotation90:
    """Random 90-degree rotation in axial plane"""
    
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if random.random() < self.p:
            k = random.randint(1, 3)  # Number of 90-degree rotations
            
            for key in sample.keys():
                if key != 'case_id':
                    # Rotate in (H, W) plane (axes 1, 2)
                    sample[key] = np.rot90(sample[key], k=k, axes=(1, 2)).copy()
        
        return sample


class RandomIntensityShift:
    """Random intensity shift for data augmentation"""
    
    def __init__(self, shift_range=0.1, p=0.5):
        """
        Args:
            shift_range: Maximum shift in standard deviations
            p: Probability of applying shift
        """
        self.shift_range = shift_range
        self.p = p
    
    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if random.random() < self.p:
            shift = np.random.uniform(-self.shift_range, self.shift_range)
            
            for key in ['t1', 't2', 'flair', 't1ce']:
                if key in sample:
                    brain_mask = sample[key] > 0
                    sample[key][brain_mask] += shift
        
        return sample


class Compose:
    """Compose multiple transforms"""
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample


def get_transforms(split='train'):
    """Get transforms for train/val split"""
    
    if split == 'train':
        return Compose([
            RandomFlip3D(axes=(0, 1, 2), p=0.5),
            RandomRotation90(p=0.5),
            RandomIntensityShift(shift_range=0.1, p=0.3),
        ])
    else:
        # No augmentation for validation
        return None
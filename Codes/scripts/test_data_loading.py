"""
Test script to verify data loading works correctly
"""

import sys
sys.path.append('..')

import torch
from torch.utils.data import DataLoader
from data.dataset import BraTSDataset, collate_fn
from data.transforms import get_transforms
import matplotlib.pyplot as plt
import numpy as np


def test_data_loading():
    """Test data loading and visualization"""
    
    # Create dataset
    dataset = BraTSDataset(
        root_dir="/path/to/BraTS2024_Synthesis",  # UPDATE THIS
        split='train',
        train_ratio=0.8,
        transforms=get_transforms('train'),
        load_seg=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # Get one batch
    batch = next(iter(dataloader))
    
    print("\nBatch contents:")
    for key, value in batch.items():
        if key != 'case_id':
            print(f"{key}: {value.shape}, min={value.min():.3f}, max={value.max():.3f}")
        else:
            print(f"{key}: {value}")
    
    # Visualize middle slice
    visualize_batch(batch, slice_idx=77)


def visualize_batch(batch, slice_idx=77):
    """Visualize all modalities from a batch"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Case: {batch['case_id'][0]}", fontsize=16)
    
    modalities = ['t1', 't2', 'flair', 't1ce', 'seg']
    titles = ['T1', 'T2', 'FLAIR', 'T1ce', 'Segmentation']
    
    for idx, (mod, title) in enumerate(zip(modalities, titles)):
        row = idx // 3
        col = idx % 3
        
        # Get slice (batch, channel, D, H, W)
        if mod == 'seg':
            slice_data = batch[mod][0, slice_idx, :, :].cpu().numpy()
            axes[row, col].imshow(slice_data, cmap='jet', vmin=0, vmax=3)
        else:
            slice_data = batch[mod][0, 0, slice_idx, :, :].cpu().numpy()
            axes[row, col].imshow(slice_data, cmap='gray')
        
        axes[row, col].set_title(title)
        axes[row, col].axis('off')
    
    # Hide last subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('data_sample.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'data_sample.png'")
    plt.show()


if __name__ == '__main__':
    test_data_loading()
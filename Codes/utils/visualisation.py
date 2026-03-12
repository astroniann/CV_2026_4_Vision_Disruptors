"""
Visualization utilities
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def save_comparison_plot(
    t1: np.ndarray,
    t2: np.ndarray,
    flair: np.ndarray,
    real: np.ndarray,
    synthetic: np.ndarray,
    save_path: Path,
    slice_idx: int = 77
):
    """
    Save comparison plot of inputs, ground truth, and prediction
    
    Args:
        t1, t2, flair: Input modalities (D, H, W)
        real: Ground truth T1ce (D, H, W)
        synthetic: Predicted T1ce (D, H, W)
        save_path: Where to save the plot
        slice_idx: Which slice to visualize
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Inputs
    axes[0, 0].imshow(t1[slice_idx], cmap='gray')
    axes[0, 0].set_title('T1', fontsize=14)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(t2[slice_idx], cmap='gray')
    axes[0, 1].set_title('T2', fontsize=14)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(flair[slice_idx], cmap='gray')
    axes[0, 2].set_title('FLAIR', fontsize=14)
    axes[0, 2].axis('off')
    
    # Row 2: Ground truth, Synthetic, Difference
    axes[1, 0].imshow(real[slice_idx], cmap='gray')
    axes[1, 0].set_title('Ground Truth T1ce', fontsize=14)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(synthetic[slice_idx], cmap='gray')
    axes[1, 1].set_title('Synthetic T1ce', fontsize=14)
    axes[1, 1].axis('off')
    
    # Difference map
    diff = np.abs(real[slice_idx] - synthetic[slice_idx])
    im = axes[1, 2].imshow(diff, cmap='hot')
    axes[1, 2].set_title('Absolute Difference', fontsize=14)
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
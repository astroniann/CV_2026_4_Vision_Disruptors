"""
Loss functions for MRI synthesis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim3d


class CombinedLoss(nn.Module):
    """Combined loss: MSE + SSIM"""
    
    def __init__(self, mse_weight=1.0, ssim_weight=0.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted image (B, 1, D, H, W)
            target: Ground truth image (B, 1, D, H, W)
        
        Returns:
            Combined loss
        """
        # MSE loss
        loss_mse = F.mse_loss(pred, target)
        
        # SSIM loss (1 - SSIM to minimize)
        if self.ssim_weight > 0:
            loss_ssim = 1 - ssim3d(pred, target, data_range=1.0)
        else:
            loss_ssim = torch.tensor(0.0).to(pred.device)
        
        # Combined
        total_loss = self.mse_weight * loss_mse + self.ssim_weight * loss_ssim
        
        return total_loss, {
            'mse': loss_mse.item(),
            'ssim': loss_ssim.item() if self.ssim_weight > 0 else 0.0
        }


def get_loss_function(loss_type='mse', **kwargs):
    """Factory function for loss functions"""
    
    if loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'l1':
        return nn.L1Loss()
    elif loss_type == 'combined':
        return CombinedLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
"""
Evaluation metrics for MRI synthesis
"""

import torch
import numpy as np
from pytorch_msssim import ssim3d


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute 3D SSIM
    
    Args:
        pred: Predicted image (B, 1, D, H, W)
        target: Ground truth image (B, 1, D, H, W)
    
    Returns:
        SSIM value
    """
    with torch.no_grad():
        ssim_val = ssim3d(pred, target, data_range=1.0)
    return ssim_val.item()


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute PSNR
    
    Args:
        pred: Predicted image (B, 1, D, H, W)
        target: Ground truth image (B, 1, D, H, W)
    
    Returns:
        PSNR in dB
    """
    with torch.no_grad():
        mse = torch.mean((pred - target) ** 2)
        if mse == 0:
            return 100.0
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()


def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute MSE"""
    with torch.no_grad():
        mse = torch.mean((pred - target) ** 2)
    return mse.item()


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute MAE"""
    with torch.no_grad():
        mae = torch.mean(torch.abs(pred - target))
    return mae.item()


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    Compute all metrics
    
    Args:
        pred: Predicted image (B, 1, D, H, W)
        target: Ground truth image (B, 1, D, H, W)
    
    Returns:
        Dictionary of metrics
    """
    return {
        'ssim': compute_ssim(pred, target),
        'psnr': compute_psnr(pred, target),
        'mse': compute_mse(pred, target),
        'mae': compute_mae(pred, target),
    }
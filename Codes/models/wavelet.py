"""
3D Discrete Wavelet Transform for medical images
Multi-scale wavelet decomposition
"""

import torch
import torch.nn as nn
import pywt
import numpy as np
from typing import Tuple, List


class WaveletTransform3D:
    """
    3D Discrete Wavelet Transform using PyWavelets
    
    Converts 3D volume to wavelet domain and back
    """
    
    def __init__(self, wavelet='haar'):
        """
        Args:
            wavelet: Wavelet type ('haar', 'db2', 'sym4', etc.)
        """
        self.wavelet = wavelet
    
    def dwt(self, volume: np.ndarray) -> np.ndarray:
        """
        Forward 3D Discrete Wavelet Transform
        
        Args:
            volume: Input volume (D, H, W)
        
        Returns:
            Wavelet coefficients (8, D/2, H/2, W/2)
        """
        # Apply 3D wavelet transform
        coeffs = pywt.dwtn(volume, self.wavelet, axes=(0, 1, 2))
        
        # Pack 8 subbands into tensor
        # LLL (approximation), LLH, LHL, LHH, HLL, HLH, HHL, HHH (details)
        wavelet_coeffs = np.stack([
            coeffs['aaa'],  # Low-Low-Low (smooth/approximation)
            coeffs['aad'],  # Low-Low-High (Z-direction details)
            coeffs['ada'],  # Low-High-Low (Y-direction details)
            coeffs['add'],  # Low-High-High
            coeffs['daa'],  # High-Low-Low (X-direction details)
            coeffs['dad'],  # High-Low-High
            coeffs['dda'],  # High-High-Low
            coeffs['ddd'],  # High-High-High (diagonal details)
        ], axis=0)
        
        return wavelet_coeffs
    
    def idwt(self, wavelet_coeffs: np.ndarray) -> np.ndarray:
        """
        Inverse 3D Discrete Wavelet Transform
        
        Args:
            wavelet_coeffs: Wavelet coefficients (8, D/2, H/2, W/2)
        
        Returns:
            Reconstructed volume (D, H, W)
        """
        # Unpack tensor into coefficient dictionary
        coeffs = {
            'aaa': wavelet_coeffs[0],
            'aad': wavelet_coeffs[1],
            'ada': wavelet_coeffs[2],
            'add': wavelet_coeffs[3],
            'daa': wavelet_coeffs[4],
            'dad': wavelet_coeffs[5],
            'dda': wavelet_coeffs[6],
            'ddd': wavelet_coeffs[7],
        }
        
        # Apply inverse transform
        volume = pywt.idwtn(coeffs, self.wavelet, axes=(0, 1, 2))
        
        return volume
    
    def forward_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Forward transform for batched tensors
        
        Args:
            batch: (B, C, D, H, W)
        
        Returns:
            Wavelet coefficients (B, C*8, D/2, H/2, W/2)
        """
        B, C, D, H, W = batch.shape
        device = batch.device
        
        # Process each sample and channel
        wavelet_batch = []
        
        for b in range(B):
            channel_wavelets = []
            for c in range(C):
                volume = batch[b, c].cpu().numpy()
                wavelet_coeffs = self.dwt(volume)
                channel_wavelets.append(wavelet_coeffs)
            
            # Stack channels: (C, 8, D/2, H/2, W/2) -> (C*8, D/2, H/2, W/2)
            channel_wavelets = np.stack(channel_wavelets, axis=0)
            channel_wavelets = channel_wavelets.reshape(C*8, D//2, H//2, W//2)
            wavelet_batch.append(channel_wavelets)
        
        wavelet_batch = np.stack(wavelet_batch, axis=0)
        
        return torch.from_numpy(wavelet_batch).to(device)
    
    def inverse_batch(self, wavelet_batch: torch.Tensor, original_channels: int = 1) -> torch.Tensor:
        """
        Inverse transform for batched tensors
        
        Args:
            wavelet_batch: (B, C*8, D/2, H/2, W/2)
            original_channels: Number of original channels
        
        Returns:
            Reconstructed batch (B, C, D, H, W)
        """
        B, _, Dw, Hw, Ww = wavelet_batch.shape
        device = wavelet_batch.device
        
        reconstructed_batch = []
        
        for b in range(B):
            channel_reconstructed = []
            
            for c in range(original_channels):
                # Extract 8 wavelet coefficients for this channel
                start_idx = c * 8
                end_idx = start_idx + 8
                wavelet_coeffs = wavelet_batch[b, start_idx:end_idx].cpu().numpy()
                
                # Reconstruct
                volume = self.idwt(wavelet_coeffs)
                channel_reconstructed.append(volume)
            
            channel_reconstructed = np.stack(channel_reconstructed, axis=0)
            reconstructed_batch.append(channel_reconstructed)
        
        reconstructed_batch = np.stack(reconstructed_batch, axis=0)
        
        return torch.from_numpy(reconstructed_batch).to(device).float()


class MultiScaleWaveletTransform:
    """
    Multi-scale 3D wavelet decomposition
    
    Decomposes into 3 scales:
    - Scale 1 (fine): (8, D/2, H/2, W/2)
    - Scale 2 (medium): (8, D/4, H/4, W/4)
    - Scale 3 (coarse): (8, D/8, H/8, W/8)
    """
    
    def __init__(self, wavelet='haar'):
        self.wavelet = wavelet
        self.wt = WaveletTransform3D(wavelet)
    
    def decompose(self, volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Multi-scale decomposition
        
        Args:
            volume: (D, H, W)
        
        Returns:
            scale1: (8, D/2, H/2, W/2)
            scale2: (8, D/4, H/4, W/4)
            scale3: (8, D/8, H/8, W/8)
        """
        # Scale 1: First decomposition
        scale1 = self.wt.dwt(volume)
        
        # Scale 2: Decompose LLL component from scale 1
        lll_1 = scale1[0]  # Low-Low-Low component
        scale2 = self.wt.dwt(lll_1)
        
        # Scale 3: Decompose LLL component from scale 2
        lll_2 = scale2[0]
        scale3 = self.wt.dwt(lll_2)
        
        return scale1, scale2, scale3
    
    def reconstruct(
        self,
        scale1: np.ndarray,
        scale2: np.ndarray,
        scale3: np.ndarray
    ) -> np.ndarray:
        """
        Multi-scale reconstruction
        
        Args:
            scale1, scale2, scale3: Wavelet coefficients at 3 scales
        
        Returns:
            Reconstructed volume (D, H, W)
        """
        # Reconstruct from scale 3
        lll_2 = self.wt.idwt(scale3)
        
        # Replace LLL component in scale 2 and reconstruct
        scale2_modified = scale2.copy()
        scale2_modified[0] = lll_2
        lll_1 = self.wt.idwt(scale2_modified)
        
        # Replace LLL component in scale 1 and reconstruct
        scale1_modified = scale1.copy()
        scale1_modified[0] = lll_1
        volume = self.wt.idwt(scale1_modified)
        
        return volume


def test_wavelet_transform():
    """Test wavelet transform and verify perfect reconstruction"""
    
    print("Testing 3D Wavelet Transform...")
    
    # Create random volume
    np.random.seed(42)
    volume = np.random.randn(240, 240, 155).astype(np.float32)
    
    # Single scale test
    wt = WaveletTransform3D('haar')
    
    # Forward
    wavelet_coeffs = wt.dwt(volume)
    print(f"Original shape: {volume.shape}")
    print(f"Wavelet shape: {wavelet_coeffs.shape}")
    
    # Inverse
    reconstructed = wt.idwt(wavelet_coeffs)
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Check reconstruction error
    error = np.abs(volume - reconstructed).max()
    print(f"Max reconstruction error: {error:.10f}")
    assert error < 1e-6, "Reconstruction error too large!"
    
    print("✓ Single-scale transform test passed!\n")
    
    # Multi-scale test
    print("Testing Multi-Scale Wavelet Transform...")
    
    mwt = MultiScaleWaveletTransform('haar')
    
    scale1, scale2, scale3 = mwt.decompose(volume)
    print(f"Scale 1: {scale1.shape}")
    print(f"Scale 2: {scale2.shape}")
    print(f"Scale 3: {scale3.shape}")
    
    reconstructed = mwt.reconstruct(scale1, scale2, scale3)
    error = np.abs(volume - reconstructed).max()
    print(f"Max reconstruction error: {error:.10f}")
    assert error < 1e-6, "Multi-scale reconstruction error too large!"
    
    print("✓ Multi-scale transform test passed!\n")
    
    # Batch test
    print("Testing batched transform...")
    
    batch = torch.randn(2, 1, 240, 240, 155)  # (B, C, D, H, W)
    wavelet_batch = wt.forward_batch(batch)
    print(f"Batch wavelet shape: {wavelet_batch.shape}")
    
    reconstructed_batch = wt.inverse_batch(wavelet_batch, original_channels=1)
    print(f"Reconstructed batch shape: {reconstructed_batch.shape}")
    
    error = (batch - reconstructed_batch).abs().max()
    print(f"Max reconstruction error: {error:.10f}")
    assert error < 1e-5, "Batch reconstruction error too large!"
    
    print("✓ Batched transform test passed!")
    
    print("\n✅ All wavelet transform tests passed!")


if __name__ == '__main__':
    test_wavelet_transform()
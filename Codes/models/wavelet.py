"""
Stage 1: Multi-Scale Wavelet Decomposition for TAMS-WDM
========================================================

Applies 3-level 3D Discrete Wavelet Transform (DWT) with Haar wavelets
to decompose MRI volumes into hierarchical subband representations.

For an input volume X ∈ R^{240×240×155}, each DWT level produces 8 subband
coefficients (1 approximation + 7 detail subbands), yielding:

  Level 1 (fine):   (8, 120, 120, 78)   — tumor boundaries & fine texture
  Level 2 (medium): (8,  60,  60, 39)   — intermediate structure
  Level 3 (coarse): (8,  30,  30, 20)   — overall brain anatomy

Subband ordering (consistent with pywt 3D notation):
  Index 0: 'aaa' — approximation (LLL) — smooth/low-frequency
  Index 1: 'aad' — LLH — Z-axis edges
  Index 2: 'ada' — LHL — Y-axis edges
  Index 3: 'add' — LHH — YZ-diagonal
  Index 4: 'daa' — HLL — X-axis edges
  Index 5: 'dad' — HLH — XZ-diagonal
  Index 6: 'dda' — HHL — XY-diagonal
  Index 7: 'ddd' — HHH — trilinear diagonal (high-frequency noise/detail)

DWT guarantees perfect reconstruction via iDWT, unlike learned latent spaces.
"""

import torch
import torch.nn as nn
import numpy as np
import pywt
from typing import Dict, List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
#  Low-level numpy utilities (preserved from original implementation)
# ─────────────────────────────────────────────────────────────────────────────

# Fixed subband key ordering — must be consistent across DWT and iDWT
SUBBAND_KEYS = ['aaa', 'aad', 'ada', 'add', 'daa', 'dad', 'dda', 'ddd']


def _dwt3d(volume: np.ndarray, wavelet: str = 'haar') -> np.ndarray:
    """
    Single-level 3D DWT on a volume of shape (D, H, W).

    Returns:
        np.ndarray of shape (8, D//2, H//2, W//2-ish)
        e.g. for (240,240,155) → (8, 120, 120, 78)
    """
    coeffs = pywt.dwtn(volume, wavelet, axes=(0, 1, 2))
    # Stack 8 subbands along new leading axis
    wavelet_coeffs = np.stack([coeffs[k] for k in SUBBAND_KEYS], axis=0)
    return wavelet_coeffs  # (8, D', H', W')


def _idwt3d(wavelet_coeffs: np.ndarray, wavelet: str = 'haar') -> np.ndarray:
    """
    Single-level 3D iDWT.

    Args:
        wavelet_coeffs: (8, D', H', W')

    Returns:
        np.ndarray of shape (D, H, W) — perfectly reconstructed volume
    """
    coeffs = {k: wavelet_coeffs[i] for i, k in enumerate(SUBBAND_KEYS)}
    volume = pywt.idwtn(coeffs, wavelet, axes=(0, 1, 2))
    return volume


def _multiscale_dwt3d(
    volume: np.ndarray,
    num_levels: int = 3,
    wavelet: str = 'haar'
) -> List[np.ndarray]:
    """
    Multi-level 3D DWT.  Iteratively applies DWT to the approximation (aaa)
    subband from the previous level.

    Args:
        volume: (D, H, W)
        num_levels: Number of decomposition levels (default 3)
        wavelet: Wavelet family

    Returns:
        List of length `num_levels`, each element of shape (8, D', H', W')
        where D', H', W' halve at each level (with ceiling for odd dims).
        Index 0 = finest (level 1), index 2 = coarsest (level 3).
    """
    scales = []
    current = volume

    for _ in range(num_levels):
        coeffs = _dwt3d(current, wavelet)    # (8, ...)
        scales.append(coeffs)
        current = coeffs[0]                  # approximate (aaa) → next level input

    return scales  # [level1, level2, level3]


def _multiscale_idwt3d(
    scales: List[np.ndarray],
    wavelet: str = 'haar'
) -> np.ndarray:
    """
    Inverse of `_multiscale_dwt3d`. Reconstructs original volume from
    multi-scale wavelet coefficients.

    Args:
        scales: List [level1, level2, level3] — same format as output of
                _multiscale_dwt3d. Coarsest level is scales[-1].

    Returns:
        Reconstructed volume (D, H, W)
    """
    num_levels = len(scales)

    # Start from coarsest level
    # Replace aaa subband iteratively going from coarse → fine
    approx = None

    for level_coeffs in reversed(scales):  # level3 -> level2 -> level1
        if approx is not None:
            # Substitute the reconstructed approximation back.
            # pywt idwt may produce one extra sample along odd dimensions
            # (e.g. idwt of size-39 produces size-40). Trim to match c[0].
            level_coeffs = level_coeffs.copy()
            target_shape = level_coeffs[0].shape
            slices = tuple(slice(0, s) for s in target_shape)
            level_coeffs[0] = approx[slices]
        approx = _idwt3d(level_coeffs, wavelet)

    return approx


# ─────────────────────────────────────────────────────────────────────────────
#  PyTorch nn.Module: WaveletEncoder — Stage 1 of TAMS-WDM
# ─────────────────────────────────────────────────────────────────────────────

class WaveletEncoder(nn.Module):
    """
    Stage 1 of TAMS-WDM: Multi-Scale Wavelet Decomposition.

    Takes a batch of multi-modal MRI volumes and decomposes each modality
    into 3 hierarchical wavelet scales independently.

    Input:
        Separate tensors per modality, each (B, 1, D, H, W)
        e.g. t1, t2, flair each of shape (B, 1, 240, 240, 155)

    Output:
        Dict mapping modality name → list of 3 tensors (one per level):
        {
            't1':    [level1 (B,8,120,120,78), level2 (B,8,60,60,39), level3 (B,8,30,30,20)],
            't2':    [...],
            'flair': [...]
        }

    Note:
        pywt operates on numpy arrays (CPU). Tensors on CUDA are temporarily
        moved to CPU for the DWT, then returned to the original device.
        For training on GPU this adds a small host↔device transfer — acceptable
        because the DWT itself is not differentiable w.r.t. the filter bank
        (Haar filters are fixed), and gradients flow through the U-Net.
    """

    def __init__(self, wavelet: str = 'haar', num_levels: int = 3):
        """
        Args:
            wavelet: Wavelet type — 'haar' (default), 'db2', 'sym4', etc.
            num_levels: Number of decomposition levels (default 3)
        """
        super().__init__()
        self.wavelet = wavelet
        self.num_levels = num_levels

        # Register (non-trainable) metadata so it shows up in model.named_modules()
        self.register_buffer('_dummy', torch.zeros(1))

    def _encode_volume(self, volume_tensor: torch.Tensor) -> List[torch.Tensor]:
        """
        Decompose a single (B, 1, D, H, W) tensor into num_levels wavelet scales.

        Returns list of (B, 8, D', H', W') tensors, one per level.
        """
        B = volume_tensor.shape[0]
        device = volume_tensor.device
        dtype = volume_tensor.dtype

        batch_scales = [[] for _ in range(self.num_levels)]

        # Process each sample in the batch
        for b in range(B):
            # volume: (D, H, W) — squeeze out batch + channel dims
            volume_np = volume_tensor[b, 0].cpu().numpy().astype(np.float64)

            # Multi-scale decomposition
            scales = _multiscale_dwt3d(volume_np, self.num_levels, self.wavelet)

            for lvl, coeffs in enumerate(scales):
                batch_scales[lvl].append(coeffs)   # each coeffs: (8, D', H', W')

        # Stack batch along dim-0, convert back to torch on original device
        level_tensors = []
        for lvl in range(self.num_levels):
            stacked = np.stack(batch_scales[lvl], axis=0)   # (B, 8, D', H', W')
            level_tensors.append(
                torch.from_numpy(stacked).to(device=device, dtype=dtype)
            )

        return level_tensors   # [level1_tensor, level2_tensor, level3_tensor]

    def forward(self, **modalities: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        """
        Args:
            **modalities: keyword arguments mapping modality name → tensor (B,1,D,H,W)
                          e.g. t1=..., t2=..., flair=...

        Returns:
            Dict[modality_name, List[level_tensor]]
            where level_tensor has shape (B, 8, D', H', W')
        """
        result = {}
        for name, volume in modalities.items():
            result[name] = self._encode_volume(volume)
        return result

    def extra_repr(self) -> str:
        return f"wavelet='{self.wavelet}', num_levels={self.num_levels}"


class WaveletDecoder(nn.Module):
    """
    Inverse of WaveletEncoder — used in Stage 4 (Multi-Scale Fusion).

    Takes multi-scale wavelet coefficients and reconstructs the full-resolution volume.

    Input:
        scales: List of 3 tensors [(B,8,D1,H1,W1), (B,8,D2,H2,W2), (B,8,D3,H3,W3)]
                Level 0 = finest, Level 2 = coarsest.

    Output:
        Reconstructed volume (B, 1, D, H, W)
    """

    def __init__(self, wavelet: str = 'haar', num_levels: int = 3):
        super().__init__()
        self.wavelet = wavelet
        self.num_levels = num_levels
        self.register_buffer('_dummy', torch.zeros(1))

    def forward(self, scales: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            scales: list of `num_levels` tensors, each (B, 8, D', H', W')

        Returns:
            Reconstructed volume (B, 1, D, H, W)
        """
        B = scales[0].shape[0]
        device = scales[0].device
        dtype = scales[0].dtype

        reconstructed_batch = []

        for b in range(B):
            # Convert each level to numpy
            np_scales = [s[b].cpu().numpy().astype(np.float64) for s in scales]

            # Inverse multi-scale transform
            volume_np = _multiscale_idwt3d(np_scales, self.wavelet)
            reconstructed_batch.append(volume_np)

        stacked = np.stack(reconstructed_batch, axis=0)          # (B, D, H, W)
        volume_tensor = torch.from_numpy(stacked).to(device=device, dtype=dtype)
        return volume_tensor.unsqueeze(1)                         # (B, 1, D, H, W)

    def extra_repr(self) -> str:
        return f"wavelet='{self.wavelet}', num_levels={self.num_levels}"


# ─────────────────────────────────────────────────────────────────────────────
#  Backward-compat exports (used by existing code / tests)
# ─────────────────────────────────────────────────────────────────────────────

class WaveletTransform3D:
    """
    Legacy class — single-level numpy wavelet transform.
    Kept for backward compatibility with existing test scripts.
    """

    def __init__(self, wavelet='haar'):
        self.wavelet = wavelet

    def dwt(self, volume: np.ndarray) -> np.ndarray:
        """Forward single-level DWT. Input (D,H,W) → Output (8, D', H', W')"""
        return _dwt3d(volume, self.wavelet)

    def idwt(self, wavelet_coeffs: np.ndarray) -> np.ndarray:
        """Inverse single-level DWT. Input (8, D', H', W') → Output (D,H,W)"""
        return _idwt3d(wavelet_coeffs, self.wavelet)

    def forward_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Batched forward DWT.
        Input:  (B, C, D, H, W)
        Output: (B, C*8, D', H', W')
        """
        B, C, D, H, W = batch.shape
        device = batch.device
        wavelet_batch = []

        for b in range(B):
            channel_wavelets = []
            for c in range(C):
                volume = batch[b, c].cpu().numpy()
                coeffs = self.dwt(volume)      # (8, D', H', W')
                channel_wavelets.append(coeffs)
            stacked = np.stack(channel_wavelets, axis=0)            # (C, 8, D', H', W')
            Dp, Hp, Wp = stacked.shape[2], stacked.shape[3], stacked.shape[4]
            wavelet_batch.append(stacked.reshape(C * 8, Dp, Hp, Wp))

        return torch.from_numpy(np.stack(wavelet_batch, axis=0)).to(device)

    def inverse_batch(self, wavelet_batch: torch.Tensor, original_channels: int = 1) -> torch.Tensor:
        """
        Batched inverse DWT.
        Input:  (B, C*8, D', H', W')
        Output: (B, C, D, H, W)
        """
        B = wavelet_batch.shape[0]
        device = wavelet_batch.device
        reconstructed_batch = []

        for b in range(B):
            channel_reconstructed = []
            for c in range(original_channels):
                coeffs = wavelet_batch[b, c*8:(c+1)*8].cpu().numpy()
                volume = self.idwt(coeffs)
                channel_reconstructed.append(volume)
            reconstructed_batch.append(np.stack(channel_reconstructed, axis=0))

        result = np.stack(reconstructed_batch, axis=0)
        return torch.from_numpy(result).to(device).float()


class MultiScaleWaveletTransform:
    """
    Legacy class — multi-level numpy wavelet transform.
    Kept for backward compatibility with existing test scripts.
    """

    def __init__(self, wavelet='haar'):
        self.wavelet = wavelet
        self.wt = WaveletTransform3D(wavelet)

    def decompose(self, volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        3-level decomposition. Returns (scale1, scale2, scale3).
        scale1: (8, D/2, H/2, W') — fine
        scale2: (8, D/4, H/4, W'') — medium
        scale3: (8, D/8, H/8, W''') — coarse
        """
        scales = _multiscale_dwt3d(volume, num_levels=3, wavelet=self.wavelet)
        return scales[0], scales[1], scales[2]

    def reconstruct(
        self,
        scale1: np.ndarray,
        scale2: np.ndarray,
        scale3: np.ndarray
    ) -> np.ndarray:
        """Perfect reconstruction from 3 wavelet scales."""
        return _multiscale_idwt3d([scale1, scale2, scale3], self.wavelet)
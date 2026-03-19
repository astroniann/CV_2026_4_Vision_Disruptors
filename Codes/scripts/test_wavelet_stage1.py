"""
Stage 1 Verification Script: 3D Wavelet Decomposition & Perfect Reconstruction
===============================================================================

This script tests the WaveletEncoder (DWT) and WaveletDecoder (iDWT) on:
  1. Synthetic random volumes — shape matches BraTS: (240, 240, 155)
  2. All three modalities simultaneously (t1, t2, flair)
  3. Both numpy-level (low-level) and nn.Module-level correctness
  4. Exact output shapes at each hierarchical level

Expected output shapes per modality (from a 240×240×155 input):
  Level 1 (fine):   (B, 8, 120, 120, 78)
  Level 2 (medium): (B, 8,  60,  60, 39)
  Level 3 (coarse): (B, 8,  30,  30, 20)

A perfect reconstruction should have max absolute error < 1e-5.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from models.wavelet import (
    WaveletEncoder,
    WaveletDecoder,
    WaveletTransform3D,
    MultiScaleWaveletTransform,
    _dwt3d,
    _idwt3d,
    _multiscale_dwt3d,
    _multiscale_idwt3d,
    SUBBAND_KEYS,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Pretty printing helpers
# ─────────────────────────────────────────────────────────────────────────────

def section(title: str):
    print("\n" + "═" * 60)
    print(f"  {title}")
    print("═" * 60)

def ok(msg: str):
    print(f"  ✅  {msg}")

def info(msg: str):
    print(f"  ℹ️   {msg}")

def fail(msg: str):
    print(f"  ❌  {msg}")
    raise AssertionError(msg)

# ─────────────────────────────────────────────────────────────────────────────
#  Test 1: Low-level numpy DWT on BraTS-sized volume
# ─────────────────────────────────────────────────────────────────────────────

def test_single_level_dwt():
    section("TEST 1 — Single-Level numpy DWT on (240, 240, 155)")

    vol = np.random.randn(240, 240, 155).astype(np.float32)
    info(f"Input volume shape:  {vol.shape}")

    coeffs = _dwt3d(vol, wavelet='haar')
    info(f"Wavelet coeffs shape: {coeffs.shape}")

    # Shape assertions: ceiling division for odd dimensions
    assert coeffs.shape[0] == 8,   f"Expected 8 subbands, got {coeffs.shape[0]}"
    assert coeffs.shape[1] == 120, f"Expected D'=120, got {coeffs.shape[1]}"
    assert coeffs.shape[2] == 120, f"Expected H'=120, got {coeffs.shape[2]}"
    assert coeffs.shape[3] == 78,  f"Expected W'=78, got {coeffs.shape[3]}"

    ok(f"Output shape correct: {coeffs.shape}  → (8, 120, 120, 78)")

    # Subband labels
    info(f"Subbands: {SUBBAND_KEYS}")
    info(f"  [0] aaa (LLL approximation) — energy: {np.abs(coeffs[0]).mean():.4f}")
    info(f"  [7] ddd (HHH fine details)  — energy: {np.abs(coeffs[7]).mean():.4f}")

    # Perfect reconstruction
    reconstructed = _idwt3d(coeffs, wavelet='haar')
    # Trim to original size (pywt may pad to even)
    reconstructed = reconstructed[:240, :240, :155]
    error = np.abs(vol - reconstructed).max()
    info(f"Max reconstruction error: {error:.2e}")
    assert error < 1e-4, f"Reconstruction error too large: {error}"
    ok(f"Perfect reconstruction verified (max error = {error:.2e})")


# ─────────────────────────────────────────────────────────────────────────────
#  Test 2: Multi-scale (3-level) DWT — exact shapes
# ─────────────────────────────────────────────────────────────────────────────

def test_multiscale_dwt_shapes():
    section("TEST 2 — Multi-Scale (3-level) DWT — Shape Verification")

    vol = np.random.randn(240, 240, 155).astype(np.float32)
    info(f"Input volume shape:  {vol.shape}")

    scales = _multiscale_dwt3d(vol, num_levels=3, wavelet='haar')

    expected_shapes = [
        (8, 120, 120, 78),   # Level 1 — fine
        (8,  60,  60, 39),   # Level 2 — medium
        (8,  30,  30, 20),   # Level 3 — coarse
    ]

    for lvl, (scale, expected) in enumerate(zip(scales, expected_shapes), start=1):
        info(f"Level {lvl}: {scale.shape}")
        assert scale.shape == expected, \
            f"Level {lvl} shape mismatch! Expected {expected}, got {scale.shape}"
        ok(f"Level {lvl} shape correct: {scale.shape}")

    info("")
    info("Memory footprint comparison (float32):")
    original_bytes = vol.nbytes
    wavelet_bytes  = sum(s.nbytes for s in scales)
    info(f"  Original volume:   {original_bytes / 1e6:.2f} MB")
    info(f"  Wavelet subbands:  {wavelet_bytes / 1e6:.2f} MB")
    info(f"  Ratio:             {wavelet_bytes / original_bytes:.2f}x")


# ─────────────────────────────────────────────────────────────────────────────
#  Test 3: Multi-scale iDWT — perfect reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def test_multiscale_reconstruction():
    section("TEST 3 — Multi-Scale iDWT Perfect Reconstruction")

    np.random.seed(42)
    vol = np.random.randn(240, 240, 155).astype(np.float32)

    # Forward
    scales = _multiscale_dwt3d(vol, num_levels=3, wavelet='haar')

    # Inverse
    reconstructed = _multiscale_idwt3d(scales, wavelet='haar')
    reconstructed = reconstructed[:240, :240, :155].astype(np.float32)

    error_max  = np.abs(vol - reconstructed).max()
    error_mean = np.abs(vol - reconstructed).mean()

    info(f"Max  absolute error: {error_max:.2e}")
    info(f"Mean absolute error: {error_mean:.2e}")

    assert error_max < 1e-4, f"Multi-scale reconstruction error too large: {error_max}"
    ok(f"Multi-scale perfect reconstruction verified (max error = {error_max:.2e})")


# ─────────────────────────────────────────────────────────────────────────────
#  Test 4: WaveletEncoder nn.Module — batch of 3 modalities
# ─────────────────────────────────────────────────────────────────────────────

def test_wavelet_encoder_module(batch_size: int = 1):
    section(f"TEST 4 — WaveletEncoder nn.Module (batch_size={batch_size})")

    encoder = WaveletEncoder(wavelet='haar', num_levels=3)
    info(f"Model: {encoder}")

    # Simulate BraTS volumes — (B, 1, 240, 240, 155)
    D, H, W = 240, 240, 155
    t1    = torch.randn(batch_size, 1, D, H, W)
    t2    = torch.randn(batch_size, 1, D, H, W)
    flair = torch.randn(batch_size, 1, D, H, W)

    info(f"Input per modality: {t1.shape}")

    # Forward pass
    wavelet_features = encoder(t1=t1, t2=t2, flair=flair)

    expected_shapes = [
        (batch_size, 8, 120, 120, 78),
        (batch_size, 8,  60,  60, 39),
        (batch_size, 8,  30,  30, 20),
    ]

    for mod_name, scales in wavelet_features.items():
        info(f"\n  Modality: {mod_name.upper()}")
        assert len(scales) == 3, f"Expected 3 scales, got {len(scales)}"
        for lvl, (tensor, expected) in enumerate(zip(scales, expected_shapes), start=1):
            info(f"    Level {lvl}: {tuple(tensor.shape)}")
            assert tuple(tensor.shape) == expected, \
                f"[{mod_name}] Level {lvl}: expected {expected}, got {tuple(tensor.shape)}"
            ok(f"  {mod_name.upper()} Level {lvl} → {tuple(tensor.shape)}")

    info(f"\n  Total wavelet feature tensors: {len(wavelet_features)} modalities × 3 levels = {len(wavelet_features)*3}")


# ─────────────────────────────────────────────────────────────────────────────
#  Test 5: WaveletDecoder nn.Module — end-to-end round-trip
# ─────────────────────────────────────────────────────────────────────────────

def test_round_trip_module(batch_size: int = 1):
    section(f"TEST 5 — End-to-End Round-Trip: Encoder → Decoder (batch_size={batch_size})")

    encoder = WaveletEncoder(wavelet='haar', num_levels=3)
    decoder = WaveletDecoder(wavelet='haar', num_levels=3)

    D, H, W = 240, 240, 155
    t1 = torch.randn(batch_size, 1, D, H, W)

    info(f"Original shape:  {t1.shape}")

    # Encode
    wavelet_features = encoder(t1=t1)
    scales = wavelet_features['t1']  # List of 3 tensors

    # Decode
    reconstructed = decoder(scales)

    # Trim to original spatial size (iDWT can return slightly larger due to boundary)
    reconstructed = reconstructed[:, :, :D, :H, :W]

    info(f"Reconstructed shape: {reconstructed.shape}")
    assert reconstructed.shape == t1.shape, \
        f"Shape mismatch: {reconstructed.shape} vs {t1.shape}"

    error_max  = (t1 - reconstructed).abs().max().item()
    error_mean = (t1 - reconstructed).abs().mean().item()

    info(f"Max  absolute error: {error_max:.2e}")
    info(f"Mean absolute error: {error_mean:.2e}")

    assert error_max < 1e-4, f"Round-trip error too large: {error_max}"
    ok(f"End-to-end round-trip verified! (max error = {error_max:.2e})")


# ─────────────────────────────────────────────────────────────────────────────
#  Test 6: Subband energy analysis — semantic interpretation
# ─────────────────────────────────────────────────────────────────────────────

def test_subband_energy_analysis():
    section("TEST 6 — Subband Energy Analysis (Semantic Interpretation)")

    np.random.seed(0)
    vol = np.random.randn(240, 240, 155).astype(np.float32)
    scales = _multiscale_dwt3d(vol, num_levels=3, wavelet='haar')

    labels = [
        "LLL — approximation (📐 anatomy)",
        "LLH — Z-axis edges",
        "LHL — Y-axis edges",
        "LHH — YZ-diagonal",
        "HLL — X-axis edges",
        "HLH — XZ-diagonal",
        "HHL — XY-diagonal",
        "HHH — fine detail (🔬 noise/texture)",
    ]

    for lvl, scale in enumerate(scales, start=1):
        info(f"\n  Level {lvl} subbands:")
        total_energy = np.sum(scale ** 2)
        for i, label in enumerate(labels):
            subband_energy = np.sum(scale[i] ** 2)
            pct = 100.0 * subband_energy / total_energy
            bar = "█" * int(pct / 2)
            print(f"      [{i}] {label:35s}  {pct:5.1f}%  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
#  Test 7: Legacy backward-compat classes
# ─────────────────────────────────────────────────────────────────────────────

def test_legacy_classes():
    section("TEST 7 — Backward-Compatibility (WaveletTransform3D, MultiScaleWaveletTransform)")

    vol = np.random.randn(240, 240, 155).astype(np.float32)

    # WaveletTransform3D
    wt = WaveletTransform3D('haar')
    c = wt.dwt(vol)
    assert c.shape == (8, 120, 120, 78)
    r = wt.idwt(c)[:240, :240, :155]
    assert np.abs(vol - r).max() < 1e-4
    ok("WaveletTransform3D: DWT ↔ iDWT verified")

    # MultiScaleWaveletTransform
    mwt = MultiScaleWaveletTransform('haar')
    s1, s2, s3 = mwt.decompose(vol)
    rec = mwt.reconstruct(s1, s2, s3)[:240, :240, :155].astype(np.float32)
    assert np.abs(vol - rec).max() < 1e-4
    ok("MultiScaleWaveletTransform: decompose ↔ reconstruct verified")

    # Batched tensor interface
    batch = torch.randn(2, 1, 64, 64, 64)
    wb = wt.forward_batch(batch)
    assert wb.shape == (2, 8, 32, 32, 32)
    rb = wt.inverse_batch(wb, original_channels=1)
    assert rb.shape[0] == 2
    ok("WaveletTransform3D.forward_batch / inverse_batch verified")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "╔" + "═"*58 + "╗")
    print("║   TAMS-WDM Stage 1: Wavelet Decomposition Test Suite     ║")
    print("╚" + "═"*58 + "╝")
    print("\n  Testing 3D Haar DWT → iDWT on BraTS volumes (240×240×155)")

    test_single_level_dwt()
    test_multiscale_dwt_shapes()
    test_multiscale_reconstruction()
    test_wavelet_encoder_module(batch_size=1)
    test_round_trip_module(batch_size=1)
    test_subband_energy_analysis()
    test_legacy_classes()

    print("\n" + "╔" + "═"*58 + "╗")
    print("║         ✅  ALL STAGE 1 TESTS PASSED                     ║")
    print("╚" + "═"*58 + "╝\n")


if __name__ == '__main__':
    main()

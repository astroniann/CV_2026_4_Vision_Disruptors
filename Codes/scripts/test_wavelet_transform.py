# -*- coding: ascii -*-
"""
test_wavelet_transform.py
=========================
Standalone Stage-1 verification for TAMS-WDM:
  3D Discrete Wavelet Transform (DWT) + perfect reconstruction.

Dataset: BraTS 2024 Meningioma Radiotherapy (MEN-RT)
         D:\\user\\Brats-MenRet\\BraTS2024-MEN-RT-TrainingData\\BraTS-MEN-RT-Train-v2\\
         Case structure:
             BraTS-MEN-RT-XXXX-1/
                 BraTS-MEN-RT-XXXX-1_t1c.nii.gz   <- T1 contrast-enhanced
                 BraTS-MEN-RT-XXXX-1_gtv.nii.gz   <- GTV segmentation

Requirements (Python 3.9+):
    pip install numpy PyWavelets nibabel matplotlib

Usage:
    python test_wavelet_transform.py                   # synthetic only
    python test_wavelet_transform.py --nifti auto      # auto-pick first t1c
    python test_wavelet_transform.py --n-cases 50      # 50 cases, all files, CSV summary
    python test_wavelet_transform.py --nifti auto --n-cases 50
"""

import sys
import os
import argparse
import csv
import time
import traceback
from pathlib import Path

# --- Dataset config -----------------------------------------------------------

DATASET_ROOT = Path(r"D:\user\BraTS2024-GLI\BraTS2024-BraTS-GLI-AdditionalTrainingData\training_data_additional")
TRAIN_DIR    = DATASET_ROOT
VAL_DIR      = DATASET_ROOT  # Assuming split handling later

WAVELET    = "haar"
NUM_LEVELS = 3

OUTPUT_DIR = Path(__file__).resolve().parent / "wavelet_test_outputs"

# --- Dependency check ---------------------------------------------------------
print("=" * 65)
print("  TAMS-WDM  |  Stage 1: Wavelet Transform Verification")
print("  Dataset  : BraTS 2024 GLI (Additional)")
print("  Path     : %s" % DATASET_ROOT)
print("=" * 65)
print()
print("[INIT] Checking required libraries...")

REQUIRED = {
    "numpy":      "numpy",
    "PyWavelets": "pywt",
    "nibabel":    "nibabel",
    "matplotlib": "matplotlib",
}

missing = []
for pip_name, import_name in REQUIRED.items():
    try:
        __import__(import_name)
        mod = sys.modules[import_name]
        ver = getattr(mod, "__version__", "?")
        print("  [OK]  %s %s" % (pip_name, ver))
    except ImportError:
        print("  [FAIL]  %s  ->  pip install %s" % (pip_name, pip_name))
        missing.append(pip_name)

if missing:
    print()
    print("[ERROR] Missing: pip install %s" % " ".join(missing))
    sys.exit(1)

import numpy as np
import pywt
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print()
print("[INFO] Outputs -> %s" % OUTPUT_DIR)
print()


# --- Subband metadata ---------------------------------------------------------

SUBBAND_KEYS = ['aaa', 'aad', 'ada', 'add', 'daa', 'dad', 'dda', 'ddd']
SUBBAND_LABELS = [
    "aaa  LLL  approximation  (anatomy / smooth)",
    "aad  LLH  Z-edges",
    "ada  LHL  Y-edges",
    "add  LHH  YZ-diagonal",
    "daa  HLL  X-edges",
    "dad  HLH  XZ-diagonal",
    "dda  HHL  XY-diagonal",
    "ddd  HHH  fine detail   (texture / noise)",
]


# --- Core DWT helpers ---------------------------------------------------------

def dwt3d(volume):
    """Single-level 3D DWT (D,H,W) -> (8,D',H',W')."""
    c = pywt.dwtn(volume, WAVELET, axes=(0, 1, 2))
    return np.stack([c[k] for k in SUBBAND_KEYS], axis=0)


def idwt3d(coeffs):
    """Single-level 3D iDWT (8,D',H',W') -> (D,H,W)."""
    c = {k: coeffs[i] for i, k in enumerate(SUBBAND_KEYS)}
    return pywt.idwtn(c, WAVELET, axes=(0, 1, 2))


def multiscale_dwt(volume):
    """
    3-level DWT. Each level decomposes the previous approximation.
    Returns list [level1, level2, level3], finest first.
    For (240,240,155):
      level1 -> (8, 120, 120, 78)
      level2 -> (8,  60,  60, 39)
      level3 -> (8,  30,  30, 20)
    """
    scales = []
    cur = volume.astype(np.float64)
    for _ in range(NUM_LEVELS):
        c = dwt3d(cur)
        scales.append(c)
        cur = c[0]
    return scales


def multiscale_idwt(scales):
    """
    Reconstruct from 3-level DWT coefficients (coarse->fine).
    Trims reconstructed approx to match target shape before substituting
    (handles pywt's 1-sample padding on odd dimensions).
    """
    approx = None
    for c in reversed(scales):
        c = c.copy()
        if approx is not None:
            target_shape = c[0].shape
            slices = tuple(slice(0, s) for s in target_shape)
            c[0] = approx[slices]
        approx = idwt3d(c)
    return approx


def trim(vol, shape):
    """Trim iDWT output to original shape."""
    return vol[tuple(slice(0, s) for s in shape)]


def load_and_normalise(path, is_seg=False):
    """Load NIfTI, z-score normalise MRI (skip for segmentations)."""
    img = nib.load(str(path))
    vol = img.get_fdata().astype(np.float32)
    if not is_seg:
        mask = vol > 0
        if mask.sum() > 0:
            lo, hi = np.percentile(vol[mask], [0.5, 99.5])
            vol = np.clip(vol, lo, hi)
            mu, sig = vol[mask].mean(), vol[mask].std()
            if sig > 0:
                vol[mask] = (vol[mask] - mu) / sig
    return vol, img


def find_cases(root):
    """Return sorted list of (case_dir, t1c_path) from MEN-RT dataset."""
    cases = []
    if not root.exists():
        return cases
    for case_dir in sorted(root.iterdir()):
        if not case_dir.is_dir():
            continue
        t1c = list(case_dir.glob("*-t1c.nii.gz"))
        if t1c:
            cases.append((case_dir, t1c[0]))
    return cases


# --- Print helpers ------------------------------------------------------------

def sep(title=""):
    print()
    print("-" * 65)
    if title:
        print("  %s" % title)
        print("-" * 65)


# --- Core synthetic tests -----------------------------------------------------

def test1_synthetic_single_level():
    sep("TEST 1 - Single-Level DWT on synthetic (240x240x155)")
    np.random.seed(0)
    vol = np.random.randn(240, 240, 155).astype(np.float32)
    print("  Input : %s  dtype=%s" % (vol.shape, vol.dtype))

    t0     = time.time()
    coeffs = dwt3d(vol)
    t_fwd  = time.time() - t0
    print("  DWT   : %s  time=%.3fs" % (coeffs.shape, t_fwd))
    assert coeffs.shape == (8, 120, 120, 78), "Shape error: %s" % str(coeffs.shape)
    print("  [OK]  Shape (8, 120, 120, 78)")

    t0    = time.time()
    rec   = trim(idwt3d(coeffs), vol.shape).astype(np.float32)
    t_inv = time.time() - t0
    err   = float(np.abs(vol - rec).max())
    print("  iDWT  : time=%.3fs  max|err|=%.2e" % (t_inv, err))
    assert err < 1e-4, "Reconstruction error too large: %e" % err
    print("  [OK]  Perfect reconstruction verified")


def test2_multiscale_shapes():
    sep("TEST 2 - Multi-Scale (3-level) DWT shapes")
    np.random.seed(1)
    vol = np.random.randn(240, 240, 155).astype(np.float32)

    t0      = time.time()
    scales  = multiscale_dwt(vol)
    elapsed = time.time() - t0

    expected = [(8,120,120,78), (8,60,60,39), (8,30,30,20)]
    names    = ["Level 1 (fine)  ", "Level 2 (medium)", "Level 3 (coarse)"]

    for lvl, (s, exp, name) in enumerate(zip(scales, expected, names)):
        mark = "[OK]" if s.shape == exp else "[FAIL]"
        print("  %s  %s: %s  (expected %s)" % (mark, name, s.shape, exp))
        assert s.shape == exp, "Level %d: %s != %s" % (lvl+1, s.shape, exp)

    orig_mb = vol.nbytes / 1e6
    wav_mb  = sum(s.nbytes for s in scales) / 1e6
    print("\n  Memory  original=%.1f MB  wavelet=%.1f MB  ratio=%.2fx"
          % (orig_mb, wav_mb, wav_mb/orig_mb))
    print("  Time    %.3fs" % elapsed)
    return scales


def test3_reconstruction(scales, vol=None):
    sep("TEST 3 - Multi-Scale iDWT Reconstruction")
    if vol is None:
        np.random.seed(1)
        vol = np.random.randn(240, 240, 155).astype(np.float32)

    t0  = time.time()
    rec = trim(multiscale_idwt(scales), vol.shape).astype(np.float32)
    elapsed = time.time() - t0

    err_max  = float(np.abs(vol - rec).max())
    err_mean = float(np.abs(vol - rec).mean())
    print("  iDWT time    : %.3fs" % elapsed)
    print("  Max  |error| : %.2e" % err_max)
    print("  Mean |error| : %.2e" % err_mean)
    assert err_max < 1e-4, "Reconstruction error: %e" % err_max
    print("  [OK]  Perfect multi-scale reconstruction verified")


def test4_subband_energy(scales):
    sep("TEST 4 - Subband Energy Distribution")
    names = ["Level 1 (fine)", "Level 2 (medium)", "Level 3 (coarse)"]
    for scale, name in zip(scales, names):
        print("\n  %s  shape=%s" % (name, str(scale.shape)))
        total = np.sum(scale ** 2) + 1e-12
        for i, label in enumerate(SUBBAND_LABELS):
            pct = 100.0 * np.sum(scale[i] ** 2) / total
            bar = "#" * max(1, int(pct / 2))
            print("    [%d] %-45s %5.1f%%  %s" % (i, label, pct, bar))
    print("\n  [OK]  Energy analysis done")


def test5_visualise(scales, prefix="synthetic"):
    sep("TEST 5 - Saving visualisation  [%s]" % prefix)
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("TAMS-WDM Stage 1 -- Wavelet Subbands [%s]" % prefix, fontsize=14)
        level_titles = [
            "Level 1 - fine (120x120x78)",
            "Level 2 - medium (60x60x39)",
            "Level 3 - coarse (30x30x20)",
        ]
        for col, (scale, ltitle) in enumerate(zip(scales, level_titles)):
            mid = scale.shape[1] // 2
            axes[0, col].imshow(scale[0][mid], cmap='gray')
            axes[0, col].set_title("%s\naaa (approximation)" % ltitle, fontsize=9)
            axes[0, col].axis('off')
            axes[1, col].imshow(scale[7][mid] * 1000, cmap='hot')
            axes[1, col].set_title("%s\nddd x1000 (fine detail)" % ltitle, fontsize=9)
            axes[1, col].axis('off')
        plt.tight_layout()
        out = OUTPUT_DIR / ("wavelets_%s.png" % prefix)
        plt.savefig(str(out), dpi=150, bbox_inches='tight')
        plt.close()
        print("  [OK]  Saved -> %s" % out)
    except Exception as e:
        print("  [WARN]  Visualisation skipped: %s" % e)


def test6_real_nifti(t1c_path):
    sep("TEST 6 - Real NIfTI: %s" % t1c_path.name)
    if not t1c_path.exists():
        print("  [WARN]  Not found: %s  (skipping)" % t1c_path)
        return None

    print("  Loading...")
    img = nib.load(str(t1c_path))
    hdr = img.header
    vol = img.get_fdata().astype(np.float32)
    print("  Shape        : %s" % str(vol.shape))
    print("  Voxel spacing: %s" % str(tuple(float("%.2f" % x) for x in hdr.get_zooms())))
    print("  Value range  : [%.2f, %.2f]" % (vol.min(), vol.max()))

    mask = vol > 0
    if mask.sum() > 0:
        lo, hi = np.percentile(vol[mask], [0.5, 99.5])
        vol = np.clip(vol, lo, hi)
        mu, s = vol[mask].mean(), vol[mask].std()
        if s > 0:
            vol[mask] = (vol[mask] - mu) / s
    print("  After norm   : [%.2f, %.2f]" % (vol.min(), vol.max()))

    print("\n  Applying %d-level Haar DWT..." % NUM_LEVELS)
    t0     = time.time()
    scales = multiscale_dwt(vol)
    t_fwd  = time.time() - t0
    for i, s in enumerate(scales):
        print("    Level %d: %s" % (i+1, str(s.shape)))
    print("  DWT time : %.3fs" % t_fwd)

    print("\n  Applying inverse DWT...")
    t0  = time.time()
    rec = trim(multiscale_idwt(scales), vol.shape).astype(np.float32)
    t_inv = time.time() - t0

    err_max  = float(np.abs(vol - rec).max())
    err_mean = float(np.abs(vol - rec).mean())
    print("  iDWT time    : %.3fs" % t_inv)
    print("  Max  |error| : %.2e" % err_max)
    print("  Mean |error| : %.2e" % err_mean)
    assert err_max < 1e-4, "Real NIfTI reconstruction error: %e" % err_max
    print("  [OK]  Perfect reconstruction on real data!")

    # 3-view comparison plot
    D, H, W = vol.shape
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle("NIfTI Round-Trip | %s" % t1c_path.parent.name, fontsize=13)
    views = [
        ("Axial",    vol[D//2, :, :], rec[D//2, :, :]),
        ("Coronal",  vol[:, H//2, :], rec[:, H//2, :]),
        ("Sagittal", vol[:, :, W//2], rec[:, :, W//2]),
    ]
    for row, (view_name, orig_s, rec_s) in enumerate(views):
        diff_s = np.abs(orig_s - rec_s)
        axes[row, 0].imshow(orig_s.T, cmap='gray', origin='lower')
        axes[row, 0].set_title("%s - Original" % view_name, fontsize=10)
        axes[row, 0].axis('off')
        axes[row, 1].imshow(rec_s.T, cmap='gray', origin='lower')
        axes[row, 1].set_title("%s - Reconstructed" % view_name, fontsize=10)
        axes[row, 1].axis('off')
        # Amplify diff for visibility
        im = axes[row, 2].imshow(diff_s.T * 1e6, cmap='hot', origin='lower')
        axes[row, 2].set_title("%s - |Diff|x1e6 (max=%.1e)" % (view_name, diff_s.max()), fontsize=10)
        axes[row, 2].axis('off')
        plt.colorbar(im, ax=axes[row, 2], fraction=0.046)
    plt.tight_layout()
    out = OUTPUT_DIR / ("nifti_roundtrip_%s.png" % t1c_path.parent.name)
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved plot -> %s" % out)
    return scales


# --- Extended: all files, N cases ---------------------------------------------

def test8_all_files_n_cases(n_cases=50):
    """
    Test ALL NIfTI files per case (t1c + gtv) for the first n_cases
    training cases. Saves a CSV summary to OUTPUT_DIR.

    BraTS-MEN-RT files per case:
        *_t1c.nii.gz  -- T1 contrast-enhanced MRI
        *_gtv.nii.gz  -- GTV tumour segmentation mask
    """
    sep("TEST 8 - ALL Files, %d Cases (BraTS-GLI-Additional Training)" % n_cases)

    cases = find_cases(TRAIN_DIR)
    if not cases:
        print("  [WARN]  No cases found at %s  (skipping)" % TRAIN_DIR)
        return

    total_avail = len(cases)
    n = min(n_cases, total_avail)
    print("  Dataset      : %s" % TRAIN_DIR)
    print("  Cases        : testing %d of %d available" % (n, total_avail))
    print("  Files/case   : ALL .nii.gz (t1c, t1n, t2w, t2f, seg)")
    print("  Threshold    : max|error| < 1e-4\n")

    results  = []
    n_pass = n_fail = n_err = 0
    t_total_start = time.time()

    for idx, (case_dir, _t1c) in enumerate(cases[:n]):
        nifti_files = sorted(case_dir.glob("*.nii.gz"))
        print("  [%02d/%02d] %s  (%d files)"
              % (idx+1, n, case_dir.name, len(nifti_files)))

        for nii_path in nifti_files:
            # Detect modality from GLI naming: *-t1c.nii.gz, *-seg.nii.gz, etc.
            stem    = nii_path.name.replace(".nii.gz", "")
            suffix  = stem.split("-")[-1]          # 't1c', 'seg', 't1n', etc.
            is_seg  = (suffix == "seg")

            try:
                t0  = time.time()
                img = nib.load(str(nii_path))
                vol = img.get_fdata().astype(np.float32)
                shp = "%dx%dx%d" % vol.shape

                # Normalise MRI only
                if not is_seg:
                    mask = vol > 0
                    if mask.sum() > 0:
                        lo, hi = np.percentile(vol[mask], [0.5, 99.5])
                        vol = np.clip(vol, lo, hi)
                        mu, sig = vol[mask].mean(), vol[mask].std()
                        if sig > 0:
                            vol[mask] = (vol[mask] - mu) / sig

                # DWT -> iDWT round-trip
                scales  = multiscale_dwt(vol)
                rec     = trim(multiscale_idwt(scales), vol.shape).astype(np.float32)
                elapsed = time.time() - t0

                err_max  = float(np.abs(vol - rec).max())
                err_mean = float(np.abs(vol - rec).mean())
                status   = "PASS" if err_max < 1e-4 else "FAIL"
                mark     = "[OK]  " if status == "PASS" else "[FAIL]"

                print("         %s  %-5s  shape=%-14s  "
                      "max_err=%.2e  mean_err=%.2e  time=%.2fs  %s"
                      % (mark, suffix, shp, err_max, err_mean, elapsed, status))

                results.append({
                    "case":       case_dir.name,
                    "file":       nii_path.name,
                    "modality":   suffix,
                    "shape":      shp,
                    "max_error":  "%.6e" % err_max,
                    "mean_error": "%.6e" % err_mean,
                    "time_s":     "%.3f"  % elapsed,
                    "status":     status,
                })
                if status == "PASS":
                    n_pass += 1
                else:
                    n_fail += 1

            except Exception as e:
                elapsed = time.time() - t0
                print("         [ERR]  %-5s  ERROR: %s" % (suffix, e))
                results.append({
                    "case": case_dir.name, "file": nii_path.name,
                    "modality": suffix, "shape": "?",
                    "max_error": "?", "mean_error": "?",
                    "time_s": "%.3f" % elapsed, "status": "ERROR",
                })
                n_err += 1

    total_time = time.time() - t_total_start
    total_files = n_pass + n_fail + n_err

    # Save CSV
    csv_path = OUTPUT_DIR / ("wavelet_test_%dcases_all_files.csv" % n)
    fieldnames = ["case","file","modality","shape","max_error","mean_error","time_s","status"]
    with open(str(csv_path), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    # Print summary table
    print()
    print("  " + "=" * 61)
    print("  SUMMARY  |  %d cases  |  %d files tested  |  total time=%.1fs"
          % (n, total_files, total_time))
    print("  " + "=" * 61)
    print("    PASS   : %d  (%.1f%%)" % (n_pass,  100*n_pass/max(total_files,1)))
    print("    FAIL   : %d" % n_fail)
    print("    ERROR  : %d" % n_err)
    print("    CSV    : %s" % csv_path)
    print("  " + "=" * 61)

    if n_fail == 0 and n_err == 0:
        print("\n  [OK]  ALL %d files across %d cases PASSED!" % (total_files, n))
    else:
        print("\n  [WARN]  %d file(s) had issues." % (n_fail + n_err))


# --- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TAMS-WDM Stage 1 -- Wavelet DWT verification"
    )
    parser.add_argument(
        "--nifti", "-n", type=str, default=None,
        help="NIfTI to test. 'auto' = first t1c from training set."
    )
    parser.add_argument(
        "--all-modalities", action="store_true",
        help="Quick round-trip test on first 3 cases"
    )
    parser.add_argument(
        "--n-cases", type=int, default=0,
        help="Test ALL files in first N training cases + save CSV (e.g. --n-cases 50)"
    )
    args = parser.parse_args()

    print("[CONFIG] Wavelet      : %s" % WAVELET)
    print("[CONFIG] Levels       : %d" % NUM_LEVELS)
    print("[CONFIG] Dataset root : %s" % DATASET_ROOT)
    print("[CONFIG] Output dir   : %s" % OUTPUT_DIR)
    print()

    passed = [0]
    failed = [0]

    def run(name, fn, *a, **kw):
        try:
            result = fn(*a, **kw)
            passed[0] += 1
            return result
        except Exception as e:
            failed[0] += 1
            print("\n  [FAIL]  %s FAILED: %s" % (name, e))
            traceback.print_exc()
            return None

    # Core synthetic tests (always run)
    run("Test 1 - Single-level DWT",     test1_synthetic_single_level)
    scales = run("Test 2 - Multi-scale shapes", test2_multiscale_shapes)

    if scales is not None:
        run("Test 3 - Reconstruction",   test3_reconstruction, scales)
        run("Test 4 - Subband energy",   test4_subband_energy, scales)
        run("Test 5 - Visualise",        test5_visualise, scales, "synthetic")

    # Single real NIfTI
    nifti_path = None
    if args.nifti == "auto":
        cases = find_cases(TRAIN_DIR)
        if cases:
            nifti_path = cases[0][1]
            print("\n[INFO] Auto-selected: %s" % nifti_path)
        else:
            print("\n[WARN] No cases found at %s" % TRAIN_DIR)
    elif args.nifti:
        nifti_path = Path(args.nifti)

    if nifti_path:
        real_scales = run("Test 6 - Real NIfTI", test6_real_nifti, nifti_path)
        if real_scales is not None:
            run("Test 6b - Visualise real", test5_visualise,
                real_scales, nifti_path.parent.name)

    # Quick 3-case test
    if args.all_modalities:
        def quick3():
            sep("TEST 7 - Quick Round-Trip: First 3 Cases")
            cases = find_cases(TRAIN_DIR)
            n = min(3, len(cases))
            ok_ = 0
            print("  Testing first %d of %d cases...\n" % (n, len(cases)))
            for case_dir, t1c in cases[:n]:
                print("  Case: %s" % case_dir.name)
                vol, _ = load_and_normalise(t1c)
                print("    Shape: %s" % str(vol.shape), end="  ")
                t0 = time.time()
                scales = multiscale_dwt(vol)
                rec = trim(multiscale_idwt(scales), vol.shape).astype(np.float32)
                elapsed = time.time() - t0
                err = float(np.abs(vol - rec).max())
                mk = "[OK]" if err < 1e-4 else "[FAIL]"
                print("max|err|=%.2e  time=%.2fs  %s" % (err, elapsed, mk))
                if err < 1e-4:
                    ok_ += 1
            if ok_ == n:
                print("\n  [OK]  All %d cases passed!" % n)
        run("Test 7 - Three cases", quick3)

    # Full extended test
    if args.n_cases > 0:
        run("Test 8 - All files %d cases" % args.n_cases,
            test8_all_files_n_cases, args.n_cases)

    # Final
    print()
    print("=" * 65)
    total = passed[0] + failed[0]
    if failed[0] == 0:
        print("  [OK]  ALL %d TESTS PASSED - Stage 1 DWT is working correctly!" % total)
    else:
        print("  [WARN]   %d/%d passed  |  %d failed" % (passed[0], total, failed[0]))
    print("=" * 65)
    print()
    sys.exit(0 if failed[0] == 0 else 1)


if __name__ == "__main__":
    main()

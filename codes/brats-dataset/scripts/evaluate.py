"""
Evaluation script — computes MSE, PSNR, and SSIM between synthesised and ground-truth
MRI volumes produced by scripts/sample.py.

Expected output_dir layout (matches what sample.py saves):

    output_dir/
        BraTS-GLI-00000-000/
            sample.nii.gz   ← synthesised volume  (H, W, 155), float32, [0, 1]
            target.nii.gz   ← ground-truth volume (H, W, 155), float32, [0, 1]
        BraTS-GLI-00001-000/
            sample.nii.gz
            target.nii.gz
        ...

Metrics are computed over brain-masked voxels only (where target > 0), which is the
standard BraSyn evaluation protocol and avoids inflating PSNR/SSIM with background air.

Usage
-----
python scripts/evaluate.py --output_dir ./results/brats_1200000/

Optional flags:
    --contr     t1n     # label printed in summary (does not affect loading)
    --save_csv  metrics.csv   # write per-patient CSV alongside the summary

Example with CSV output:
    python scripts/evaluate.py \\
        --output_dir ./results/brats_1200000/ \\
        --contr t1n \\
        --save_csv ./results/brats_1200000/metrics.csv
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

# skimage is standard in most ML environments; install via:
#   pip install scikit-image
try:
    from skimage.metrics import (
        mean_squared_error,
        peak_signal_noise_ratio,
        structural_similarity,
    )
except ImportError:
    sys.exit(
        "scikit-image is required.  Install with:  pip install scikit-image"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_vol(path: Path) -> np.ndarray:
    """Load a NIfTI volume and return a float32 numpy array."""
    return nib.load(str(path)).get_fdata().astype(np.float32)


def compute_metrics(pred: np.ndarray, target: np.ndarray):
    """
    Compute MSE, PSNR, SSIM over brain-masked voxels.

    Both volumes are expected to be in [0, 1] and have the same shape.
    Brain mask = target > 0  (background is exactly 0 after clip-and-normalize).

    Returns
    -------
    mse   : float
    psnr  : float   (dB, data_range=1.0)
    ssim  : float   (computed on full 3-D volume, data_range=1.0)
    """
    assert pred.shape == target.shape, (
        f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
    )

    mask = target > 0  # brain-only voxels

    # Clamp prediction to [0, 1] — same post-processing as sample.py
    pred = np.clip(pred, 0.0, 1.0)

    # MSE over brain voxels
    mse = float(np.mean((pred[mask] - target[mask]) ** 2))

    # PSNR: use full-volume MSE (standard) with data_range=1.0
    # skimage's peak_signal_noise_ratio requires 2-D or n-D arrays; pass full vols.
    psnr = float(peak_signal_noise_ratio(target, pred, data_range=1.0))

    # SSIM: 3-D structural similarity, data_range=1.0
    # win_size must be odd and <= smallest dim; use 7 (standard for medical images)
    min_dim = min(pred.shape)
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
    ssim = float(
        structural_similarity(
            target,
            pred,
            data_range=1.0,
            win_size=win_size,
            channel_axis=None,  # 3-D volume, no channel axis
        )
    )

    return mse, psnr, ssim


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate synthesised MRI volumes: MSE, PSNR, SSIM."
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Root directory produced by scripts/sample.py (contains per-patient subdirs)."
    )
    parser.add_argument(
        "--contr", default="",
        help="Contrast label for display only (e.g. t1n, t1c, t2w, t2f)."
    )
    parser.add_argument(
        "--save_csv", default="",
        help="Optional path to save per-patient metrics as a CSV file."
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        sys.exit(f"output_dir not found: {output_dir}")

    # Collect all patient subdirectories that contain both sample and target
    patient_dirs = sorted([
        d for d in output_dir.iterdir()
        if d.is_dir()
        and (d / "sample.nii.gz").exists()
        and (d / "target.nii.gz").exists()
    ])

    if not patient_dirs:
        sys.exit(
            f"No patient directories with sample.nii.gz + target.nii.gz found under {output_dir}.\n"
            "Make sure you have run scripts/sample.py first."
        )

    print(f"\nEvaluating {len(patient_dirs)} patients"
          + (f" | contrast: {args.contr}" if args.contr else ""))
    print("-" * 60)

    results = []  # list of (patient_id, mse, psnr, ssim)

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        pred   = load_vol(patient_dir / "sample.nii.gz")
        target = load_vol(patient_dir / "target.nii.gz")

        mse, psnr, ssim = compute_metrics(pred, target)
        results.append((patient_id, mse, psnr, ssim))

        print(f"  {patient_id:30s}  MSE={mse:.6f}  PSNR={psnr:7.3f} dB  SSIM={ssim:.4f}")

    # ---------------------------------------------------------------------------
    # Summary statistics
    # ---------------------------------------------------------------------------
    mse_vals  = [r[1] for r in results]
    psnr_vals = [r[2] for r in results]
    ssim_vals = [r[3] for r in results]

    print("-" * 60)
    print(f"  {'MEAN':30s}  MSE={np.mean(mse_vals):.6f}  "
          f"PSNR={np.mean(psnr_vals):7.3f} dB  SSIM={np.mean(ssim_vals):.4f}")
    print(f"  {'STD':30s}  MSE={np.std(mse_vals):.6f}  "
          f"PSNR={np.std(psnr_vals):7.3f} dB  SSIM={np.std(ssim_vals):.4f}")
    print(f"  {'MEDIAN':30s}  MSE={np.median(mse_vals):.6f}  "
          f"PSNR={np.median(psnr_vals):7.3f} dB  SSIM={np.median(ssim_vals):.4f}")
    print()

    # ---------------------------------------------------------------------------
    # Optional CSV export
    # ---------------------------------------------------------------------------
    if args.save_csv:
        csv_path = Path(args.save_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["patient_id", "MSE", "PSNR_dB", "SSIM"])
            for row in results:
                writer.writerow([row[0], f"{row[1]:.8f}", f"{row[2]:.4f}", f"{row[3]:.4f}"])
            # Summary rows
            writer.writerow(["MEAN", f"{np.mean(mse_vals):.8f}",
                             f"{np.mean(psnr_vals):.4f}", f"{np.mean(ssim_vals):.4f}"])
            writer.writerow(["STD",  f"{np.std(mse_vals):.8f}",
                             f"{np.std(psnr_vals):.4f}",  f"{np.std(ssim_vals):.4f}"])
        print(f"Per-patient metrics saved to: {csv_path}")


if __name__ == "__main__":
    main()

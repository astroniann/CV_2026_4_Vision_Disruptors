
# Loads all modalities (+ optional segmentation) for a single BraTS20 patient.

import glob
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .io     import load_volume
from .labels import remap_labels

MODALITIES     = ["t2f", "t1n", "t1c", "t2w"]   # BraTS 2024: flair, t1, t1ce, t2
NUM_MODALITIES = len(MODALITIES)   # 4 input channels


def load_patient(
    patient_dir: str,
    load_seg: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load all four modalities and (optionally) the segmentation mask
    for a single patient.

    Parameters
    ----------
    patient_dir : path to the patient folder
    load_seg    : whether to load the segmentation mask

    Returns
    -------
    image : np.ndarray  shape (4, H, W, D)  float32
    seg   : np.ndarray  shape (H, W, D)     uint8, or None
    """
    patient_dir = Path(patient_dir)
    patient_id  = patient_dir.name

    modality_vols = []
    for mod in MODALITIES:
        files = glob.glob(str(patient_dir / f"{patient_id}-{mod}.nii*"))
        if not files:
            raise FileNotFoundError(
                f"Missing modality '{mod}' for {patient_id}"
            )
        modality_vols.append(load_volume(files[0]))

    image = np.stack(modality_vols, axis=0)   # (4, H, W, D)

    seg = None
    if load_seg:
        files = glob.glob(str(patient_dir / f"{patient_id}-seg.nii*"))
        if not files:
            raise FileNotFoundError(
                f"Missing segmentation mask for {patient_id}"
            )
        seg = load_volume(files[0]).astype(np.uint8)
        seg = remap_labels(seg)

    return image, seg
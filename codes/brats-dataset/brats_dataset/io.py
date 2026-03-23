
# Low-level NIfTI file I/O using SimpleITK.


import numpy as np
import SimpleITK as sitk


def load_volume(path: str) -> np.ndarray:
    """
    Load a .nii / .nii.gz file and return a (H, W, D) float32 numpy array.

    SimpleITK's GetArrayFromImage returns axes in (D, H, W) order,
    so we transpose to (H, W, D) for consistency throughout the pipeline.
    """
    img = sitk.ReadImage(str(path), sitk.sitkFloat32)
    arr = sitk.GetArrayFromImage(img)   # (D, H, W)
    return arr.transpose(1, 2, 0)      # → (H, W, D)
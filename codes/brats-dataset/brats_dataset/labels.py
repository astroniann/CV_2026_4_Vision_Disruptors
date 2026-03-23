"""
BraTS20 label remapping utilities.

BraTS20 segmentation masks use labels {0, 1, 2, 4} — there is no label 3
(a historical artefact of the challenge). Label 4 is remapped to 3 so that
model outputs use contiguous class indices [0, 1, 2, 3]:

    0 → background
    1 → necrotic core / non-enhancing tumour (NCR/NET)
    2 → peritumoral oedema (ED)
    4 → enhancing tumour (ET)  — remapped to 3
"""

import numpy as np

BRATS_LABEL_MAP = {4: 3}
NUM_CLASSES     = 4       # background, NCR/NET, ED, ET


def remap_labels(seg: np.ndarray) -> np.ndarray:
    """Remap BraTS label 4 → 3 for contiguous class indices."""
    seg = seg.astype(np.uint8)
    for src, dst in BRATS_LABEL_MAP.items():
        seg[seg == src] = dst
    return seg
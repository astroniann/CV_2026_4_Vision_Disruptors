
# DataLoader factory for all three BraTS20 splits.

from typing import Optional

from torch.utils.data import DataLoader

from .dataset import BraTS20Dataset


def get_dataloader(
    data_root: str,
    csv_path: Optional[str],
    split: str,
    batch_size: int = 1,
    num_workers: int = 4,
) -> DataLoader:
    """
    Build a DataLoader for a given split.

    Parameters
    ----------
    data_root  : root folder containing train/ validation/ additional/
    csv_path   : path to brats20_data.csv (or None)
    split      : 'train' | 'validation' | 'additional'
    batch_size : number of patients per batch (typically 1–2 for 3D MRI)
    num_workers: parallel workers for loading

    Returns
    -------
    torch.utils.data.DataLoader
    """
    dataset = BraTS20Dataset(data_root, csv_path, split)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
    )
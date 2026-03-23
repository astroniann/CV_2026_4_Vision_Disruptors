"""
main.py
-------
Entry point — loads all three splits and prints a batch summary.
Replace DATA_ROOT and CSV_PATH with your actual paths.
"""

from brats_dataset import get_dataloader

DATA_ROOT = r"D:\user\BraTS2024-GLI"
CSV_PATH  = None   # set to xlsx path to filter by patient ID; None loads all patients

if __name__ == "__main__":
    train_loader = get_dataloader(DATA_ROOT, CSV_PATH, split="train")
    val_loader   = get_dataloader(DATA_ROOT, CSV_PATH, split="validation")
    add_loader   = get_dataloader(DATA_ROOT, CSV_PATH, split="additional")

    for batch in train_loader:
        print("image :", batch["image"].shape)   # (B, 4, H, W, D)
        print("seg   :", batch["seg"].shape)     # (B, H, W, D)
        break
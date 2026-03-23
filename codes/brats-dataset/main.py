import torch
from brats_dataset import get_dataloader, DWT_3D, IDWT_3D

DATA_ROOT = r"D:\user\BraTS2024-GLI"
CSV_PATH  = None
WAVENAME  = "haar"

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dwt  = DWT_3D(WAVENAME).to(device)
    idwt = IDWT_3D(WAVENAME).to(device)

    train_loader = get_dataloader(DATA_ROOT, CSV_PATH, split="train")
    val_loader   = get_dataloader(DATA_ROOT, CSV_PATH, split="validation")

    for batch in train_loader:
        image = batch["image"].to(device)   # (B, 4, H, W, D)
        seg   = batch["seg"].to(device)     # (B, H, W, D)

        # Forward DWT
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(image)
        # each sub-band: (B, 4, H/2, W/2, D/2)

        print(f"[DWT]  input  : {image.shape}")
        print(f"[DWT]  LLL    : {LLL.shape}")
        print(f"[DWT]  HHH    : {HHH.shape}")
        print(f"[DWT]  status : OK")

        # ---- your model goes here ----
        # pred = model(LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH)

        # Inverse DWT
        image_recon = idwt(LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH)
        # (B, 4, H, W, D)

        max_err = (image - image_recon).abs().max().item()
        print(f"[IDWT] output : {image_recon.shape}")
        print(f"[IDWT] max reconstruction error : {max_err:.2e}")
        print(f"[IDWT] status : {'OK' if max_err < 1e-4 else 'FAILED'}")

        # ----loss and optimizer go here ----
        # loss = criterion(image_recon, image)
        # loss.backward()
        # optimizer.step()

if __name__ == "__main__":
    train()



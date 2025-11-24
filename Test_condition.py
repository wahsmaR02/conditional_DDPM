# Test_condition_3D.py
# Fully updated 3D inference script for your 3D DDPM + UNet

import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from Diffusion_condition_3D import GaussianDiffusionSampler_cond
from Model_condition_3D import UNet3D
from datasets_new import ImageDatasetNii3D

import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim


# --------------------
# CONFIG
# --------------------
dataset_root = "playground"
out_name = "trial_3d"
patch_size = (16, 32, 32)   # MUST match training!
batch_size = 1

T = 100
ch = 64
ch_mult = [1, 2, 4]
attn = [1]
num_res_blocks = 2
dropout = 0.1
beta_1 = 1e-4
beta_T = 0.02

save_dir = "./Checkpoints_3D"
output_dir = f"./test_{out_name}"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# --------------------
# LOAD DATASET (3D patches)
# --------------------
test_dataset = ImageDatasetNii3D(
    root=dataset_root,
    split="val",            # or "train" if you have no val set
    patch_size=patch_size,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

print(f"# Test samples: {len(test_dataset)}")


# --------------------
# LOAD MODEL
# --------------------
model = UNet3D(
    T=T,
    ch=ch,
    ch_mult=ch_mult,
    attn=attn,
    num_res_blocks=num_res_blocks,
    dropout=dropout
).to(device)

ckpt_path = os.path.join(save_dir, "model_final.pt")
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint)
model.eval()

print("Loaded model weights from:", ckpt_path)


# --------------------
# LOAD DIFFUSION SAMPLER
# --------------------
sampler = GaussianDiffusionSampler_cond(
    model=model,
    beta_1=beta_1,
    beta_T=beta_T,
    T=T
).to(device)


# --------------------
# UTILITY — SAVE MIDDLE SLICE
# --------------------
def save_middle_slice(cbct, pred_ct, gt_ct, idx):
    """
    Saves a visual PNG of:
    CBCT | Predicted CT | Ground Truth CT
    using the middle D slice
    """
    cbct = cbct[0]      # [1,D,H,W]
    pred = pred_ct[0]
    gt   = gt_ct[0]

    _, D, H, W = cbct.shape
    mid = D // 2

    # Extract slice [1, H, W]
    slice_cbct = cbct[:, mid, :, :]
    slice_pred = pred[:, mid, :, :]
    slice_gt   = gt[:, mid, :, :]

    # Concatenate horizontally: [1, H, 3W]
    img = torch.cat([slice_cbct, slice_pred, slice_gt], dim=2)

    # Convert [-1,1] → [0,1] for PNG saving
    img = (img + 1.0) / 2.0
    img = torch.clamp(img, 0.0, 1.0)

    out_path = os.path.join(output_dir, f"sample_{idx:04d}.png")
    save_image(img, out_path)
    print("Saved:", out_path)


# --------------------
# Function to compute metrics
# --------------------

def compute_metrics(pred, gt):
    # pred, gt: [B,1,D,H,W]
    mse = F.mse_loss(pred, gt).item()
    mae = F.l1_loss(pred, gt).item()

    # convert to numpy for SSIM
    pred_np = pred.cpu().numpy()[0,0]
    gt_np   = gt.cpu().numpy()[0,0]

    ssim_val = ssim(gt_np, pred_np, data_range=2.0)  # [-1,1] range = 2.0

    return mse, mae, ssim_val


# --------------------
# MAIN TEST LOOP
# --------------------
def test():
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            ct = batch["CT"].to(device)        # [B,1,D,H,W]
            cbct = batch["CBCT"].to(device)    # [B,1,D,H,W]

            # Start from pure noise CT
            noise = torch.randn_like(ct)       # [B,1,D,H,W]

            # Concatenate noise + CBCT condition
            x_T = torch.cat((noise, cbct), dim=1)   # [B,2,D,H,W]

            # DDPM reverse process
            x_0_pred = sampler(x_T)             # [B,2,D,H,W]

            # Extract predicted CT channel
            pred_ct = x_0_pred[:, 0:1, ...]

            # Calculate and print evaluation metrics
            mse, mae, ssim_val = compute_metrics(pred_ct, ct)
            print(f"[{i}] MSE={mse:.4f}, MAE={mae:.4f}, SSIM={ssim_val:.4f}")


            # Visualization
            save_middle_slice(cbct, pred_ct, ct, i+1)


if __name__ == "__main__":
    test()
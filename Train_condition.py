import os
import time
import datetime
import sys

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F  # for SSIM
import matplotlib.pyplot as plt

from Diffusion_condition import (
    GaussianDiffusionTrainer_cond,
    GaussianDiffusionSampler_cond,
)
from Model_condition import UNet
from datasets_3d import VolumePatchDataset3D

# --------------------------
# Configuration
# --------------------------

dataset_root = "playground"   # Root folder containing patient subfolders
patch_size = (12, 32, 32)     # 3D patch shape (D, H, W)
batch_size = 2                # Number of patches per batch
num_epochs = 5                # Total training epochs
learning_rate = 1e-4          # Optimizer learning rate
grad_clip = 1.0               # Max gradient norm for clipping

# Diffusion hyperparameters
T = 100                       # Number of diffusion steps
ch = 64                       # Base UNet channel count
ch_mult = [1, 2, 4]           # Channel multipliers per UNet level
attn = [1]                    # Levels with attention (index into ch_mult)
num_res_blocks = 2            # ResBlocks per level
dropout = 0.1                 # Dropout rate
beta_1 = 1e-4                 # Start of beta schedule
beta_T = 0.02                 # End of beta schedule

save_dir = "./Checkpoints_3D"  # Where to save all checkpoints and logs
os.makedirs(save_dir, exist_ok=True)


# --------------------------
# Device selection (GPU/CPU/MPS)
# --------------------------

if torch.cuda.is_available():
    device = torch.device("cuda")      # Prefer CUDA if available
elif torch.backends.mps.is_available():
    device = torch.device("mps")       # Mac M1/M2 GPU backend
else:
    device = torch.device("cpu")       # Fallback to CPU

print("Using device:", device)


# --------------------------
# Dataset and DataLoader
# --------------------------

# Training dataset (random 3D patches)
train_dataset = VolumePatchDataset3D(
    root=dataset_root,
    split="train",
    patch_size=patch_size,
    seed=123,                # For reproducible patch sampling
)

# Validation dataset (NO shuffling, different seed)
val_dataset = VolumePatchDataset3D(
    root=dataset_root,
    split="val",
    patch_size=patch_size,
    seed=999,
)

# PyTorch DataLoader wraps dataset into mini-batches
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,          # Shuffle patches during training
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,         # Validation must be deterministic
    num_workers=0,
)

# --------------------------
# Model, Optimizer, Diffusion
# --------------------------

# 3D UNet backbone
model = UNet(
    T=T,
    ch=ch,
    ch_mult=ch_mult,
    attn=attn,
    num_res_blocks=num_res_blocks,
    dropout=dropout,
).to(device)

# AdamW optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Diffusion TRAINING module (predicts noise)
trainer = GaussianDiffusionTrainer_cond(
    model=model,
    beta_1=beta_1,
    beta_T=beta_T,
    T=T,
).to(device)

# Diffusion SAMPLER module (reverse diffusion)
sampler = GaussianDiffusionSampler_cond(
    model=model,
    beta_1=beta_1,
    beta_T=beta_T,
    T=T,
).to(device)


# --------------------------
# SSIM helper for 3D volumes
# --------------------------

def ssim3d(x, y, C1=0.01**2, C2=0.03**2):
    """
    Computes a simple SSIM approximation for 3D tensors.
    x, y: [B,1,D,H,W]
    Using small avg_pool3d windows to estimate:
      - local mean
      - variance
      - covariance
    Returns mean SSIM over entire patch.
    """
    mu_x = F.avg_pool3d(x, 3, 1, 0)
    mu_y = F.avg_pool3d(y, 3, 1, 0)

    sigma_x = F.avg_pool3d(x * x, 3, 1, 0) - mu_x**2
    sigma_y = F.avg_pool3d(y * y, 3, 1, 0) - mu_y**2
    sigma_xy = F.avg_pool3d(x * y, 3, 1, 0) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))

    return ssim_map.mean()


# --------------------------
# Validation SSIM computation
# --------------------------

@torch.no_grad()
def compute_val_ssim(model, sampler, val_loader, device, max_batches=5):
    """
    Computes SSIM using the reverse diffusion sampler.
    Only uses the first few batches (max_batches) for speed.
    """
    model.eval()
    ssim_vals = []

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break

        ct = batch["pCT"].to(device)      # Ground truth CT
        cbct = batch["CBCT"].to(device)   # Conditioning CBCT

        # Start reverse diffusion with noise for CT + real CBCT
        noise = torch.randn_like(ct)
        x_T = torch.cat((noise, cbct), dim=1)

        # Sample reconstructed CT
        out = sampler(x_T)
        pred_ct = out[:, 0:1, ...]        # Extract CT channel

        # Compute SSIM between predicted CT and ground-truth CT
        ssim_val = ssim3d(pred_ct, ct)
        ssim_vals.append(ssim_val.item())

    return sum(ssim_vals) / len(ssim_vals) # Average over batches


# --------------------------
# Storage for logging
# --------------------------

train_losses = []
val_losses = []
val_ssims = []


# --------------------------
# Training Loop + Validation
# --------------------------

best_ssim = -1.0
prev_time = time.time()

for epoch in range(1, num_epochs + 1):

    # --------------------------
    # TRAINING PHASE
    # --------------------------
    model.train()
    train_loss = 0.0

    for batch in train_loader:
        ct = batch["pCT"].to(device)       # Ground truth CT
        cbct = batch["CBCT"].to(device)   # Condition CBCT

        x_0 = torch.cat((ct, cbct), dim=1)  # Inputs: CT + CBCT

        optimizer.zero_grad()
        loss, numel = trainer(x_0)          # Predict noise
        loss = loss / numel                 # Calculate mean loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)        # Compute average training loss per batch


    # --------------------------
    # VALIDATION LOSS
    # --------------------------
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            ct = batch["pCT"].to(device)
            cbct = batch["CBCT"].to(device)

            x_0 = torch.cat((ct, cbct), dim=1)
            loss, numel = trainer(x_0)
            loss = loss / numel
            val_loss += loss.item()

    val_loss /= len(val_loader)


    # --------------------------
    # VALIDATION SSIM (every 5 epochs)
    # --------------------------
    val_ssim = None
    if epoch % 5 == 0:
        print("Computing validation SSIM via sampler...")
        val_ssim = compute_val_ssim(model, sampler, val_loader, device)
        print(f"Epoch {epoch} — Validation SSIM: {val_ssim:.4f}")

        # Save best model based on SSIM score
        if val_ssim > best_ssim:
            best_ssim = val_ssim
            best_path = os.path.join(save_dir, "best_model.pt")
            torch.save(model.state_dict(), best_path)
            print(f"✓ Saved BEST model (SSIM={val_ssim:.4f}) → {best_path}")


    # --------------------------
    # LOG TRAIN/VAL STATS
    # --------------------------
    epoch_dur = datetime.timedelta(seconds=(time.time() - prev_time))
    prev_time = time.time()

    print(
        f"Epoch {epoch}/{num_epochs} | "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f} | "
        f"Duration: {epoch_dur}"
    )

    # Save logs for plotting
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    if val_ssim is not None:
        val_ssims.append(val_ssim)
    else:
        val_ssims.append(None)

    # Save checkpoint each epoch
    ckpt_path = os.path.join(save_dir, f"ckpt_epoch_{epoch}.pt")
    torch.save(model.state_dict(), ckpt_path)


# --------------------------
# Save final model
# --------------------------
final_path = os.path.join(save_dir, "model_final.pt")
torch.save(model.state_dict(), final_path)
print(f"Final model saved to: {final_path}")


# --------------------------
# Plot Loss Curves
# --------------------------
plt.figure(figsize=(10,5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Diffusion Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, "loss_curve.png"))
plt.close()

# --------------------------
# Plot SSIM Curve
# --------------------------
plt.figure(figsize=(10,5))
epochs_ssim = [e for e in range(1, num_epochs+1)]
plt.plot(epochs_ssim, val_ssims, marker='o')
plt.xlabel('Epoch')
plt.ylabel('SSIM')
plt.title('Validation SSIM (every 5 epochs)')
plt.grid(True)
plt.savefig(os.path.join(save_dir, "ssim_curve.png"))
plt.close()

print("Saved training curves to:", save_dir)

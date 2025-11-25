import os
import time
import datetime
import sys

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F  # for SSIM
import matplotlib.pyplot as plt

from Diffusion_condition_3D import (
    GaussianDiffusionTrainer_cond,
    GaussianDiffusionSampler_cond,
)
from Model_condition_3D import UNet3D
from datasets_new import ImageDatasetNii3D


# --------------------------
# Configuration
# --------------------------

dataset_root = "playground"   # root folder containing HN/TH/AB
patch_size = (12, 32, 32)
batch_size = 2
num_epochs = 20
learning_rate = 1e-4
grad_clip = 1.0

T = 100
ch = 64
ch_mult = [1, 2, 4]
attn = [1]
num_res_blocks = 2
dropout = 0.1
beta_1 = 1e-4
beta_T = 0.02

save_dir = "./Checkpoints_3D"
os.makedirs(save_dir, exist_ok=True)

# --------------------------
# Device selection (GPU/CPU/MPS)
# --------------------------

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)

# --------------------------
# Dataset and DataLoader
# --------------------------

train_dataset = ImageDatasetNii3D(
    root=dataset_root,
    split="train",
    patch_size=patch_size,
    seed=123,
)

val_dataset = ImageDatasetNii3D(
    root=dataset_root,
    split="val",
    patch_size=patch_size,
    seed=999,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
)

# --------------------------
# Model, Optimizer, Diffusion
# --------------------------

model = UNet3D(
    T=T,
    ch=ch,
    ch_mult=ch_mult,
    attn=attn,
    num_res_blocks=num_res_blocks,
    dropout=dropout,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

trainer = GaussianDiffusionTrainer_cond(
    model=model,
    beta_1=beta_1,
    beta_T=beta_T,
    T=T,
).to(device)

sampler = GaussianDiffusionSampler_cond(
    model=model,
    beta_1=beta_1,
    beta_T=beta_T,
    T=T,
).to(device)


# --------------------------
# SSIM helper for 3D volumes (patch-level, simple version)
# --------------------------

def ssim3d(x, y, C1=0.01**2, C2=0.03**2):
    """
    x, y: [B,1,D,H,W]
    Computes SSIM per batch, returns average over batch.
    Assumes x, y are roughly normalized to a similar range.
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
# Computing SSIM on validation
# --------------------------

@torch.no_grad()
def compute_val_ssim(model, sampler, val_loader, device, max_batches=5):
    """
    Computes SSIM on reconstructed CT using the existing sampler.
    Uses only the first few val batches for speed.
    """
    model.eval()
    ssim_vals = []

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break

        ct = batch["CT"].to(device)      # [B,1,D,H,W]
        cbct = batch["CBCT"].to(device)  # [B,1,D,H,W]

        # --- prepare x_T: random noise for CT + CBCT condition ---
        noise = torch.randn_like(ct)
        x_T = torch.cat((noise, cbct), dim=1)  # [B,2,D,H,W]

        # --- run sampler (reverse diffusion) ---
        out = sampler(x_T)               # [B,2,D,H,W], CT in channel 0
        pred_ct = out[:, 0:1, ...]       # only CT channel

        # --- compute SSIM between predicted CT and ground-truth CT ---
        ssim_val = ssim3d(pred_ct, ct)
        ssim_vals.append(ssim_val.item())

    return sum(ssim_vals) / len(ssim_vals)

# --------------------------
# Save losses for plotting
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
    # TRAINING
    # --------------------------
    model.train()
    train_loss = 0.0

    for batch in train_loader:
        ct = batch["CT"].to(device)
        cbct = batch["CBCT"].to(device)

        x_0 = torch.cat((ct, cbct), dim=1)  # [B,2,D,H,W]

        optimizer.zero_grad()
        loss = trainer(x_0)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # --------------------------
    # VALIDATION LOSS
    # --------------------------
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            ct = batch["CT"].to(device)
            cbct = batch["CBCT"].to(device)

            x_0 = torch.cat((ct, cbct), dim=1)
            loss = trainer(x_0)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    # --------------------------
    # SSIM (Every 5 epochs)
    # --------------------------
    val_ssim = None
    if epoch % 5 == 0:
        print("Computing validation SSIM via sampler...")
        val_ssim = compute_val_ssim(model, sampler, val_loader, device)
        print(f"Epoch {epoch} — Validation SSIM: {val_ssim:.4f}")

        # Best checkpoint by SSIM
        if val_ssim > best_ssim:
            best_ssim = val_ssim
            best_path = os.path.join(save_dir, "best_model.pt")
            torch.save(model.state_dict(), best_path)
            print(f"✓ Saved BEST model (SSIM={val_ssim:.4f}) → {best_path}")

    # --------------------------
    # LOGGING
    # --------------------------
    epoch_dur = datetime.timedelta(seconds=(time.time() - prev_time))
    prev_time = time.time()

    print(
        f"Epoch {epoch}/{num_epochs} | "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f} | "
        f"Duration: {epoch_dur}"
    )
    # --------------------------
    # Save loss & ssim
    # --------------------------
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    if val_ssim is not None:
        val_ssims.append(val_ssim)
    else:
        val_ssims.append(None)


    # --------------------------
    # Save checkpoint every epoch (optional)
    # --------------------------
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
# Plot SSIM (only every 5 epochs)
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
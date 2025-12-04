# Train_condition.py
import os
import time
import datetime
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

from Diffusion_condition import (
    GaussianDiffusionTrainer_cond,
    GaussianDiffusionSampler_cond,
)
from Model_condition import UNet
from datasets_3d import VolumePatchDataset3D

# --------------------------
# ADD: Path to SynthRAD2025 metrics
# --------------------------
# Adjust this to wherever you cloned SynthRAD2025/metrics 
# Example: sys.path.append("/home/you/SynthRAD2025/metrics")
sys.path.append("/content/drive/MyDrive/Project_in_Scientific_Computing/metrics") # !! Change this to the correct path 
from SynthRAD_metrics import ImageMetrics  # MAE, PSNR, MS-SSIM

metrics = ImageMetrics(debug=False)

# --------------------------
# Configuration
# --------------------------

dataset_root = "/mnt/asgard0/users/p25_2025/synthRAD2025_Task2_Train/synthRAD2025_Task2_Train/Task2"   # Root folder containing patient subfolders
patch_size = (16, 64, 64)     # 3D patch shape (D, H, W)
batch_size = 2                # Number of patches per batch
num_epochs = 30               # Total training epochs
learning_rate = 1e-4          # Optimizer learning rate
grad_clip = 1.0               # Max gradient norm for clipping

# Diffusion hyperparameters
T = 1000                       # Number of diffusion steps
ch = 64                      # Base UNet channel count
ch_mult = [1, 2, 3, 4]           # Channel multipliers per UNet level
attn = [2]                    # Levels with attention (index into ch_mult)
num_res_blocks = 2            # ResBlocks per level
dropout = 0.3                 # Dropout rate
beta_1 = 1e-4                 # Start of beta schedule
beta_T = 0.02                 # End of beta schedule

save_dir = "./Checkpoints_3D"  # Where to save all checkpoints and logs
os.makedirs(save_dir, exist_ok=True)


# --------------------------
# Device selection (GPU/CPU/MPS)
# --------------------------

if torch.cuda.is_available():
    device = torch.device("cuda")      # Prefer CUDA if available
    print(f"[INFO] Using GPU: {torch.cuda.get_device_name(device)}")
else:
    print("[ERROR] No CUDA-compatible GPU found. "
          "Make sure you requested a GPU node...")
    sys.exit(1)

print("Using device:", device)

import gc

TARGET_MAX_GB = 12  # you want to use max half of 24GB

def gpu_memory_gb():
    return torch.cuda.memory_allocated() / 1024**3

def enforce_memory_limit(batch_size):
    current = gpu_memory_gb()
    if current > TARGET_MAX_GB:
        print(f"GPU memory high: {current:.2f} GB > {TARGET_MAX_GB} GB.")
        print("→ Automatically reducing batch size and enabling gradient accumulation.")

        new_batch = max(1, batch_size // 2)
        torch.cuda.empty_cache()
        gc.collect()
        return new_batch, True  # batch_size, accumulate_gradients
    return batch_size, False


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
    pin_memory=True,
    persistent_workers=False,
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

# Cosine LR scheduler
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,     # total number of epochs
    eta_min=1e-6          # minimum LR at the end
)

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
# Function to save space
# --------------------------
def save_clean(model, path):
    """Save CPU-only, grad-free state dict (small file)."""
    sd = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(sd, path)
    model.to(device)

# --------------------------
# Denormalization: [-1,1] -> HU
# (inverse of datasets_3d.norm_hu)
# --------------------------
def denorm_hu(x_norm: torch.Tensor,
              lo: float = -1000.0,
              hi: float = 2000.0) -> np.ndarray:
    """
    x_norm: torch tensor in [-1,1]
    returns: numpy array in HU
    """
    x = x_norm.cpu().numpy()
    hu = ( (x + 1.0) / 2.0 ) * (hi - lo) + lo
    return hu.astype(np.float32)


# --------------------------
# Validation metrics (MAE, PSNR, MS-SSIM)
# --------------------------

@torch.no_grad()
def compute_val_metrics(model,
                        sampler,
                        val_loader,
                        device,
                        max_batches=5,
                        seed: int = 42):
    """
    Computes MAE, PSNR, MS-SSIM using SynthRAD2025 ImageMetrics
    on a subset of validation batches.

    Returns:
        (mean_mae, mean_psnr, mean_ms_ssim_masked)
    """
    model.eval()

    mae_vals = []
    psnr_vals = []
    msssim_vals = []

    # Fix RNG for deterministic sampling across epochs
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break

        ct = batch["pCT"].to(device)      # [B,1,D,H,W], normalized [-1,1]
        cbct = batch["CBCT"].to(device)   # [B,1,D,H,W], normalized [-1,1]

        # Start reverse diffusion with noise for CT + real CBCT
        noise = torch.randn_like(ct)
        x_T = torch.cat((noise, cbct), dim=1)  # [B,2,D,H,W]

        # Sample reconstructed CT
        x_0 = sampler(x_T)                 # [B,2,D,H,W]
        pred_ct = x_0[:, 0:1, ...]         # [B,1,D,H,W]

        # Denormalize to HU for metrics
        gt_np = denorm_hu(ct).squeeze(1)       # [B,D,H,W]
        pred_np = denorm_hu(pred_ct).squeeze(1)

        B = gt_np.shape[0]
        for b in range(B):
            gt_vol = gt_np[b]
            pred_vol = pred_np[b]

            # Use a mask of all ones (patch-based; no body mask here)
            mask = np.ones_like(gt_vol, dtype=np.float32)

            mae = metrics.mae(gt_vol, pred_vol, mask)
            psnr = metrics.psnr(gt_vol, pred_vol, mask, use_population_range=True)
            _, ms_ssim_mask = metrics.ms_ssim(gt_vol, pred_vol, mask)

            mae_vals.append(mae)
            psnr_vals.append(psnr)
            msssim_vals.append(ms_ssim_mask)

    # Restore RNG
    torch.set_rng_state(cpu_state)
    if torch.cuda.is_available() and cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state)

    mean_mae = float(np.mean(mae_vals)) if mae_vals else float("nan")
    mean_psnr = float(np.mean(psnr_vals)) if psnr_vals else float("nan")
    mean_msssim = float(np.mean(msssim_vals)) if msssim_vals else float("nan")

    return mean_mae, mean_psnr, mean_msssim


# --------------------------
# Storage for logging
# --------------------------

train_losses = []
val_losses = []

val_maes = []
val_psnrs = []
val_msssims = []


# --------------------------
# Training Loop + Validation
# --------------------------

best_msssim = -1.0
prev_time = time.time()

accum_steps = 1  # will increase automatically if needed

for epoch in range(1, num_epochs + 1):

    # --------------------------
    # TRAINING PHASE
    # --------------------------
    model.train()
    train_loss = 0.0

    for batch in train_loader:

        # --- Memory governor (only meaningful on GPU) ---
        used = gpu_memory_gb()
        if used > TARGET_MAX_GB and torch.cuda.is_available():
            print(f"⚠️ GPU at {used:.2f} GB > {TARGET_MAX_GB} GB → Reducing load")
            accum_steps *= 2
            torch.cuda.empty_cache()
            gc.collect()
            print(f"→ Gradient accumulation increased to {accum_steps}")

        ct = batch["pCT"].to(device)
        cbct = batch["CBCT"].to(device)
        x_0 = torch.cat((ct, cbct), dim=1)  # [B,2,D,H,W]

        optimizer.zero_grad()

        # Scale loss if using accumulation
        loss, numel = trainer(x_0)
        loss = loss / numel
        loss = loss / accum_steps

        loss.backward()

        # NOTE: train_loader._index is a bit hacky; if it breaks, replace with a manual counter.
        if (batch_idx := getattr(train_loader, "_index", 0)) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        train_loss += loss.item()

    train_loss /= len(train_loader)

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
    # VALIDATION METRICS (every 5 epochs)
    # --------------------------
    epoch_mae = None
    epoch_psnr = None
    epoch_msssim = None

    if epoch % 5 == 0:
        print("Computing validation MAE / PSNR / MS-SSIM via SynthRAD metrics...")
        epoch_mae, epoch_psnr, epoch_msssim = compute_val_metrics(
            model, sampler, val_loader, device, max_batches=5
        )
        print(
            f"Epoch {epoch} — "
            f"Val MAE: {epoch_mae:.4f} | "
            f"Val PSNR: {epoch_psnr:.4f} | "
            f"Val MS-SSIM(mask): {epoch_msssim:.4f}"
        )

        # Track best model by MS-SSIM (higher is better)
        if epoch_msssim > best_msssim:
            best_msssim = epoch_msssim
            best_path = os.path.join(save_dir, "best_model.pt")
            save_clean(model, best_path)
            print(f"✓ Saved BEST model (MS-SSIM={epoch_msssim:.4f}) → {best_path}")

    # --------------------------
    # LOG TRAIN/VAL STATS
    # --------------------------
    epoch_dur = datetime.timedelta(seconds=(time.time() - prev_time))
    prev_time = time.time()

    current_lr = scheduler.get_last_lr()[0]
    print(
        f"Epoch {epoch}/{num_epochs} | "
        f"LR: {current_lr:.6e} | "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f} | "
        f"Duration: {epoch_dur}"
    )

    scheduler.step()

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_maes.append(epoch_mae)
    val_psnrs.append(epoch_psnr)
    val_msssims.append(epoch_msssim)

# --------------------------
# Save final model
# --------------------------
final_path = os.path.join(save_dir, "model_final.pt")
save_clean(model, final_path)
print(f"Final model saved to: {final_path}")

# --------------------------
# Plot Loss Curves
# --------------------------
plt.figure(figsize=(10, 5))
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
# Plot Metric Curves (only where computed)
# --------------------------
epochs = np.arange(1, num_epochs + 1)

def masked_plot(values, label):
    xs = [e for e, v in zip(epochs, values) if v is not None]
    ys = [v for v in values if v is not None]
    if xs:
        plt.plot(xs, ys, marker='o', label=label)

plt.figure(figsize=(10, 5))
masked_plot(val_maes, "Val MAE")
masked_plot(val_psnrs, "Val PSNR")
masked_plot(val_msssims, "Val MS-SSIM(mask)")
plt.xlabel('Epoch')
plt.ylabel('Metric value')
plt.title('Validation Metrics (SynthRAD-style, patch-based)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, "metrics_curve.png"))
plt.close()

print("Saved training curves to:", save_dir)


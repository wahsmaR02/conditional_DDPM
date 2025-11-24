# Training script for 3D conditional DDPM (updated for patch-based 3D training)

import os
import time
import datetime
import sys

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from Diffusion_condition_3D import GaussianDiffusionTrainer_cond
from Model_condition_3D import UNet3D
from datasets_new import ImageDatasetNii3D


# --------------------------
# Configuration
# --------------------------

dataset_root = "synthRAD2025_Task2_Train/playground"   # root folder containing HN/TH/AB
patch_size = (64,128,128)
batch_size = 2
num_epochs = 200
learning_rate = 1e-4
grad_clip = 1.0

T = 1000
ch = 64
ch_mult = [1, 2, 4, 4]
attn = [1]
num_res_blocks = 2
dropout = 0.1
beta_1 = 1e-4
beta_T = 0.02

save_dir = "./Checkpoints_3D"
os.makedirs(save_dir, exist_ok=True)

# --------------------------
# Device selection (CPU/MPS)
# --------------------------

if torch.backends.mps.is_available():
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
    seed=123
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
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
    dropout=dropout
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

trainer = GaussianDiffusionTrainer_cond(
    model=model,
    beta_1=beta_1,
    beta_T=beta_T,
    T=T
).to(device)


# --------------------------
# Training Loop
# --------------------------

prev_time = time.time()

for epoch in range(1, num_epochs + 1):
    running_loss = 0.0

    for batch_idx, batch in enumerate(train_loader, start=1):

        ct = batch["pCT"].to(device)       # [B, 1, D, H, W]
        cbct = batch["CBCT"].to(device)    # [B, 1, D, H, W]

        x_0 = torch.cat((ct, cbct), dim=1)  # [B, 2, D, H, W]

        optimizer.zero_grad()
        loss = trainer(x_0) / (batch_size * 256 * 256 * 96)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running_loss += loss.item()

    epoch_duration = datetime.timedelta(seconds=(time.time() - prev_time))
    prev_time = time.time()

    print(f"Epoch {epoch}/{num_epochs} | Duration: {epoch_duration} | Loss: {running_loss:.6f}")

    if epoch % 10 == 0:
        ckpt_path = os.path.join(save_dir, f"ckpt_epoch_{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")
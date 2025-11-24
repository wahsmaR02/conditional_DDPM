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

dataset_root = "playground"   # root folder containing HN/TH/AB
patch_size = (12,32,32)
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
    seed=123
)

val_dataset = ImageDatasetNii3D(
    root=dataset_root,
    split="val",
    patch_size=patch_size,
    seed=999
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
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
# Training Loop + Validation
# --------------------------

best_val_loss = float("inf")
prev_time = time.time()

for epoch in range(1, num_epochs + 1):
    # --------------------------
    # TRAINING
    # --------------------------
    model.train()
    train_loss = 0.0

    for batch in train_loader:
        ct = batch["CT"].to(device)       # [B,1,D,H,W]
        cbct = batch["CBCT"].to(device)   # [B,1,D,H,W]

        x_0 = torch.cat((ct, cbct), dim=1)  # [B,2,D,H,W]

        optimizer.zero_grad()
        loss = trainer(x_0)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)


    # --------------------------
    # VALIDATION
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
    # Logging
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
    # Save "best" model (lowest val loss)
    # --------------------------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_path = os.path.join(save_dir, "best_model.pt")
        torch.save(model.state_dict(), best_path)
        print(f"✓ Saved BEST model to {best_path}")


    # --------------------------
    # Optional: Save checkpoint every epoch
    # --------------------------
    ckpt_path = os.path.join(save_dir, f"ckpt_epoch_{epoch}.pt")
    torch.save(model.state_dict(), ckpt_path)


# --------------------------
# Save final model
# --------------------------
final_path = os.path.join(save_dir, "model_final.pt")
torch.save(model.state_dict(), final_path)
print(f"Final model saved to: {final_path}")
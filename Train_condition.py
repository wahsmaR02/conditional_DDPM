# Jan. 2023, by Junbo Peng, PhD Candidate, Georgia Tech
# Training script for 3D conditional DDPM (updated for patch-based 3D training)

import os
import time
import datetime
import sys

import torch
from torch.utils.data import DataLoader

from Diffusion_condition import GaussianDiffusionTrainer_cond
from Model_condition import UNet
from datasets import VolumePatchDataset3D


# --------------------------
# Configuration
# --------------------------

dataset_root = "/content/drive/MyDrive/Project in Scientific Computing/Train/conditional_DDPM/playground"   # root folder containing HN/TH/AB 
patch_size = (16, 32, 32)
batch_size = 2
num_epochs = 5
learning_rate = 1e-4
grad_clip = 1.0

T = 100
ch = 64
ch_mult = [1, 2, 3, 4]
attn = []
num_res_blocks = 1
dropout = 0.0
beta_1 = 1e-4
beta_T = 0.02

save_dir = "./Checkpoints_3D"
os.makedirs(save_dir, exist_ok=True)

# --------------------------
# Device selection (CPU/MPS)
# --------------------------

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")


# --------------------------
# Dataset and DataLoader
# --------------------------

train_dataset = VolumePatchDataset3D(
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

model = UNet(
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

       # print("\n[Train] Batch CT:", ct.shape)
        #print("[Train] Batch CBCT:", cbct.shape)

        x_0 = torch.cat((ct, cbct), dim=1)  # [B, 2, D, H, W]

        optimizer.zero_grad()
        loss, n_vox = trainer(x_0) #train the model to predict the noise
        loss = loss/ n_vox # normalize by number of voxels

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
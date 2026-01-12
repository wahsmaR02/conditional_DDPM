# plot_hu_hist_cbct_vs_ct.py
# Plots HU histogram: CBCT vs CT (RAW values, before any normalization)

import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

# --------------------------
# Configuration
# --------------------------
dataset_root = "/mnt/asgard0/users/p25_2025/synthRAD2025_Task2_Train/synthRAD2025_Task2_Train/Task2"  # <-- CHANGE THIS
cohorts = ("HN", "TH", "AB")

use_mask = True            # True = only voxels inside mask.mha (recommended)
max_patients = None        # e.g. 10 for quick test, or None for all
max_voxels_total = 2_000_000  # cap total sampled voxels (keeps memory + speed sane)
hist_range = (-1200, 3200) # HU range shown in plot
bins = 400                 # number of bins

# --------------------------
# Helpers
# --------------------------
def load_arr(path, dtype=np.float32):
    img = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(img).astype(dtype)

def iter_patients(root, cohorts):
    for cohort in cohorts:
        cohort_dir = os.path.join(root, cohort)
        if not os.path.isdir(cohort_dir):
            continue
        for pid in sorted(os.listdir(cohort_dir)):
            pdir = os.path.join(cohort_dir, pid)
            if not os.path.isdir(pdir):
                continue
            cbct_path = os.path.join(pdir, "cbct.mha")
            ct_path   = os.path.join(pdir, "ct.mha")
            mask_path = os.path.join(pdir, "mask.mha")
            if not (os.path.exists(cbct_path) and os.path.exists(ct_path)):
                continue
            if use_mask and not os.path.exists(mask_path):
                continue
            yield cohort, pid, cbct_path, ct_path, mask_path

# --------------------------
# Collect sampled voxels
# --------------------------
rng = np.random.default_rng(42)

cbct_samples = []
ct_samples = []
collected = 0
n_pat = 0

for cohort, pid, cbct_path, ct_path, mask_path in iter_patients(dataset_root, cohorts):
    cbct = load_arr(cbct_path, np.float32)
    ct   = load_arr(ct_path,   np.float32)

    if use_mask:
        mask = load_arr(mask_path, np.uint8) > 0
        cbct = cbct[mask]
        ct   = ct[mask]
    else:
        cbct = cbct.ravel()
        ct   = ct.ravel()

    n = cbct.size
    if n == 0:
        continue

    remaining = max_voxels_total - collected
    if remaining <= 0:
        break

    take = min(n, remaining)

    # uniform random subsample of voxels from this patient (keeps distribution roughly right)
    if take < n:
        idx = rng.choice(n, size=take, replace=False)
        cbct_take = cbct[idx]
        ct_take   = ct[idx]
    else:
        cbct_take = cbct
        ct_take   = ct

    cbct_samples.append(cbct_take)
    ct_samples.append(ct_take)
    collected += take
    n_pat += 1

    print(f"{cohort}/{pid}: took {take:,} voxels (total {collected:,})")

    if max_patients is not None and n_pat >= max_patients:
        break

cbct_samples = np.concatenate(cbct_samples) if cbct_samples else np.array([], dtype=np.float32)
ct_samples   = np.concatenate(ct_samples)   if ct_samples   else np.array([], dtype=np.float32)

print("\nDone.")
print(f"Patients used: {n_pat}")
print(f"Total voxels:  {collected:,}")
print(f"CBCT raw range: [{cbct_samples.min():.1f}, {cbct_samples.max():.1f}] HU" if cbct_samples.size else "No CBCT samples")
print(f"CT   raw range: [{ct_samples.min():.1f},   {ct_samples.max():.1f}] HU"   if ct_samples.size else "No CT samples")

# --------------------------
# Plot histograms
# --------------------------
plt.figure(figsize=(10, 5))
plt.hist(cbct_samples, bins=bins, range=hist_range, density=True, histtype="step", linewidth=1.5, label="CBCT (raw HU)")
plt.hist(ct_samples,   bins=bins, range=hist_range, density=True, histtype="step", linewidth=1.5, label="CT (raw HU)")

plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Density")
plt.title("HU Histogram (raw, before normalization)" + (" — masked" if use_mask else " — full volume"))
plt.legend()
plt.tight_layout()

# ---- Save figure ----
out_path = "/mnt/asgard0/users/p25_2025/synthRAD2025_Task2_Train/Checkpoints_3D/hu_hist_cbct_vs_ct.png"
plt.savefig(out_path, dpi=300)
print("Saved figure to:", out_path)

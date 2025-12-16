# Test_condition.py
import os
import json
import numpy as np
import torch
import SimpleITK as sitk
from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy.signal import windows

from Model_condition import UNet
from Diffusion_condition import GaussianDiffusionSampler_cond
from datasets_3d import VolumePatchDataset3D
from SynthRAD_metrics import ImageMetrics


# --------------------------
# Configuration
# --------------------------
dataset_root = "/mnt/asgard0/users/p25_2025/synthRAD2025_Task2_Train/synthRAD2025_Task2_Train/Task2"
save_dir = "./Checkpoints_3D"
output_dir = "./test_results_3d"
os.makedirs(output_dir, exist_ok=True)

#patch_size = (64, 128, 128)
#stride = (58, 115, 115)  # overlap sliding window
patch_size = (32, 64, 64)
stride = (28, 57, 57) 
T = 1000
ch = 64
ch_mult = [1, 2, 3, 4]
attn = [] #[2]
num_res_blocks = 2
dropout = 0.3

metrics = ImageMetrics(debug=False)

N_SAMPLES_TO_AVG = 5 # <-- CISSI ADDED THIS LINE FOR stochastic DDPM-AVG

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)  


# --------------------------
# Utility functions
# --------------------------

# ADD GAUSSIAN HELPER (for weighted average)
def gaussian_weight_3d(patch_size, sigma_frac=0.125):
    """Creates a 3D Gaussian weight map for smooth patch blending."""
    pD, pH, pW = patch_size
    
    def gaussian_1d(length):
        if length == 1: return np.array([1.0])
        # Use sigma based on a fraction of the length for smooth decay
        sigma = length * sigma_frac
        g = windows.gaussian(length, std=sigma)
        return g

    gD = gaussian_1d(pD)
    gH = gaussian_1d(pH)
    gW = gaussian_1d(pW)
    
    # Outer product to create 3D weight map
    weights = gD[:, None, None] * gH[None, :, None] * gW[None, None, :]
    # Normalize to ensure max value is 1.0
    return (weights / weights.max()).astype(np.float32)

def norm_hu(x):
    lo, hi = -1024, 2000
    #x = np.clip(x, lo, hi)
    return 2 * (x - lo) / (hi - lo) - 1


def denorm_hu(x):
    lo, hi = -1024, 2000
    return ((x + 1) / 2) * (hi - lo) + lo


def sliding_window_inference(model, sampler, cbct_norm, device, mask: np.ndarray, batch_size=4, num_samples_to_avg=N_SAMPLES_TO_AVG): # <-- CISSI MODIFIED THIS LINE FOR stochastic DDPM-AVG 
    """
    Runs sliding-window inference over the full CBCT volume,
    but only processes patches that intersect the mask.
    Args:
        batch_size: Number of patches to process simultaneously (default=4)
                   Increase for better speed, decrease if running out of memory
    """
    D, H, W = cbct_norm.shape
    pD, pH, pW = patch_size
    sD, sH, sW = stride

    cbct_t = torch.from_numpy(cbct_norm).float().to(device)
    # noise_generator = torch.Generator(device=device).manual_seed(SEED) # <-- CISSI REMOVED THIS LINE for DDPM-AVG

    # 1. NEW: Initialize Gaussian weights once
    patch_weights_np = gaussian_weight_3d(patch_size)
    patch_weights_t = torch.from_numpy(patch_weights_np).float().to(device)

    # 2. NEW: output_weights replaces output_count
    output_sum  = torch.zeros((D, H, W), device=device)
    output_weights = torch.zeros((D, H, W), device=device)
    
    #output_sum  = torch.zeros((D, H, W), device=device)
    #output_count = torch.zeros((D, H, W), device=device)

    # Compute valid patch starting indices
    z_idx = list(range(0, D - pD + 1, sD))
    y_idx = list(range(0, H - pH + 1, sH))
    x_idx = list(range(0, W - pW + 1, sW))

    # Add boundary patches if needed
    if z_idx[-1] != D - pD: z_idx.append(D - pD)
    if y_idx[-1] != H - pH: y_idx.append(H - pH)
    if x_idx[-1] != W - pW: x_idx.append(W - pW)

    # Removes this to ignore background patches
    #patches = [(z, y, x) for z in z_idx for y in y_idx for x in x_idx]
    #patches = sorted(set(patches))

    # Generate all potential patch start coordinates
    all_patches = [(z, y, x) for z in z_idx for y in y_idx for x in x_idx]
    all_patches = sorted(set(all_patches)) # Removes duplicates, crucial if stride causes overlap at boundaries

    # -------------------------------------------------------------
    # NEW STEP: FILTER PATCHES TO ONLY INCLUDE THOSE INTERSECTING MASK
    # -------------------------------------------------------------
    
    # A patch is considered relevant if any voxel within its bounding box 
    # is part of the patient mask (value > 0).
    relevant_patches = []
    
    for z, y, x in all_patches:
        # Extract the patch region from the mask
        mask_patch = mask[z:z+pD, y:y+pH, x:x+pW]
        
        # Check if any part of the mask patch is non-zero
        if np.any(mask_patch > 0):
            relevant_patches.append((z, y, x))
            
    patches = relevant_patches # Use the filtered list for the loop
    
    print(f"  -> Processing {len(patches)} relevant patches (skipping {len(all_patches) - len(patches)} background patches)...")
    
    # print(f"  -> Running {len(patches)} patches in batches of {batch_size}...")

    model.eval()
    with torch.no_grad():
       # Process patches in batches
        for batch_start in tqdm(range(0, len(patches), batch_size)):
            batch_end = min(batch_start + batch_size, len(patches))
            batch_coords = patches[batch_start:batch_end]
            current_batch_size = len(batch_coords)
            
            # Collect all patches in this batch
            batch_cbct_patches = []
            # batch_noises = [] # <-- CISSI REMOVED THIS LINE for DDPM-AVG
            
            for z, y, x in batch_coords:
                # Extract patch
                patch = cbct_t[z:z+pD, y:y+pH, x:x+pW]
                batch_cbct_patches.append(patch)
                
                # Generate noise for this patch <-- CISSI REMOVED for DDPM-AVG
                #noise = torch.randn((pD, pH, pW), device=device, generator=noise_generator)
                #batch_noises.append(noise)
            
            # Stack into batch dimension: (batch_size, D, H, W)
            batch_cbct = torch.stack(batch_cbct_patches, dim=0).unsqueeze(1)  # (B, 1, D, H, W)
            # Cissi removes noise below
            #batch_noise = torch.stack(batch_noises, dim=0).unsqueeze(1)       # (B, 1, D, H, W)

            patch_sum_sct_batch = torch.zeros_like(batch_cbct) # CISSI ADDED: Accumulator

            # Cissi adds loop to perform DDPM-AVG
            for n in range(num_samples_to_avg): # ADDED: DDPM-avg loop start
            
                # CISSI ADDS STOCHASTIC NOISE FOR DDPM-AVG
                batch_noise = torch.randn_like(batch_cbct) # ADD: Stochastic noise x_T

                # Concatenate condition and noise
                x_in = torch.cat((batch_noise, batch_cbct), dim=1)  # (B, 2, D, H, W)
            
                # Process entire batch at once! 
                x_out = sampler(x_in)[:, 0:1]  # (B, 1, D, H, W)

                # Added for stochastic DDPM-AVG
                patch_sum_sct_batch += x_out # ADD: Accumulate stochastic sample
            
            avg_pred_patch_batch = patch_sum_sct_batch / num_samples_to_avg # ADD: Compute average
            
            # Distribute results back to output volume
            for i, (z, y, x) in enumerate(batch_coords):
                #pred_patch = x_out[i, 0, :, :, :]  # Extract i-th result <---- CISSI REMOVED FOR DDPM-AVG
                pred_patch = avg_pred_patch_batch[i, 0, :, :, :] # ADDED for DDPM-AVG
                # 3. Apply weights and accumulate
                weighted_pred_patch = pred_patch * patch_weights_t
                
                output_sum[z:z+pD, y:y+pH, x:x+pW] += weighted_pred_patch
                # 4. Accumulate the weights (for final division)
                output_weights[z:z+pD, y:y+pH, x:x+pW] += patch_weights_t
                
                #output_sum[z:z+pD, y:y+pH, x:x+pW] += pred_patch
                #output_count[z:z+pD, y:y+pH, x:x+pW] += 1
    
    #result = output_sum / output_count
    # Final result: Weighted sum divided by accumulated weights
    result = torch.where(output_weights > 0, output_sum / output_weights, output_sum)
    
    return result.cpu().numpy()


# --------------------------
# Main Testing
# --------------------------
def main():

    # --------------------------
    # Load test split IDs
    # --------------------------
    split_file = os.path.join(save_dir, "test_split.json")
    with open(split_file) as f:
        # CISSI EDITS TO HAVE COHORT + PID
        # test_ids = set(json.load(f))
        test_entries = json.load(f)
        split_keys = {(entry["cohort"], entry["pid"]) for entry in test_entries}


    # print(f"Loaded {len(test_ids)} test IDs from JSON.")
    print(f"Loaded {len(split_keys)} test patients from JSON.")


    # --------------------------
    # Create dataset using split='test'
    # --------------------------
    test_dataset = VolumePatchDataset3D(
        root=dataset_root,
        split="test",
        patch_size=patch_size,
        train_frac=0.6,
        val_frac=0.2,
        test_frac=0.2,
        seed=42,
    )
    # CISSI EDITS TO GET COHORT + PID
    # Filter to ensure exact match with saved split (cohort + pid)
    test_dataset.patients = [
        #p for p in test_dataset.patients if p["pid"] in test_ids
        p for p in test_dataset.patients if (p["cohort"], p["pid"]) in split_keys
    ]

    print(f"Filtered dataset contains {len(test_dataset.patients)} patients.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --------------------------
    # Load trained model
    # --------------------------
    print("Loading model...")
    model = UNet(T, ch, ch_mult, attn, num_res_blocks, dropout).to(device)

    ckpt_path = os.path.join(save_dir, "model_final.pt")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)

    sampler = GaussianDiffusionSampler_cond(model, 1e-4, 0.02, T).to(device)

    results = []

    # --------------------------
    # Inference loop over test patients
    # --------------------------
    for p in test_dataset.patients:

        pid = p["pid"]
        p_dir = p["path"]
        cohort = p["cohort"]

        print(f"\nProcessing: {cohort}/{pid}")

        cbct_path = os.path.join(p_dir, "cbct.mha")
        ct_path   = os.path.join(p_dir, "ct.mha")
        mask_path = os.path.join(p_dir, "mask.mha")

        if not os.path.exists(cbct_path):
            print("  Missing CBCT — skipping.")
            continue

        if not os.path.exists(ct_path):
            print("  Missing CT — skipping metrics.")
            continue

        # Load volumes
        cbct = sitk.GetArrayFromImage(sitk.ReadImage(cbct_path))
        gt   = sitk.GetArrayFromImage(sitk.ReadImage(ct_path))

        if os.path.exists(mask_path):
            mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path)).astype(np.float32)
        else:
            print("  Warning: no mask found — using full volume mask.")
            mask = np.ones_like(gt, dtype=np.float32)

        # Normalize
        cbct_norm = norm_hu(cbct)

        # Predict full synthetic CT
        pred_norm = sliding_window_inference(model, sampler, cbct_norm, device, mask)
        pred_hu   = denorm_hu(pred_norm)

        # ------------------------------------------------------------------
        # NEW STEP: Enforce Black Background (HU = -1000) outside the mask
        # ------------------------------------------------------------------
        BACKGROUND_HU = -1000.0

        # Ensure mask is a binary float array (1=inside, 0=outside)
        mask_binary = (mask > 0).astype(np.float32)
        
        # Apply the mask: keep prediction inside mask, set to -1000 outside
        pred_hu_masked = pred_hu * mask_binary + BACKGROUND_HU * (1 - mask_binary)
        pred_hu = pred_hu_masked 
        # ------------------------------------------------------------------

        # Evaluate metrics
        mae = metrics.mae(gt, pred_hu, mask)
        print(f"  → MAE = {mae:.2f}")

        results.append({"pid": pid, "mae": mae})

        # Save synthetic CT
        out_img = sitk.GetImageFromArray(pred_hu)
        out_img.CopyInformation(sitk.ReadImage(cbct_path))
        sitk.WriteImage(out_img, os.path.join(output_dir, f"{pid}_sct2.mha"))

    # --------------------------
    # Summary
    # --------------------------
    if results:
        avg_mae = np.mean([r["mae"] for r in results])
        print("\n===================================")
        print(f" FINAL AVERAGE MAE on TEST SET = {avg_mae:.2f}")
        print("===================================")


if __name__ == "__main__":
    main()

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


def sliding_window_inference(model, sampler, cbct_norm, device, mask: np.ndarray, batch_size=4):
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

    # 1. Prepare Gaussian Weight Map (Transfer to GPU once)
    weight_map_np = gaussian_weight_3d(patch_size)
    weight_map = torch.from_numpy(weight_map_np).float().to(device)

    # Global noise (deterministic)
    noise_generator = torch.Generator(device=device).manual_seed(SEED)
    noise_full = torch.randn((1, 1, D, H, W), device=device, generator=noise_generator)

    output_sum   = torch.zeros((D, H, W), device=device)
    output_weights = torch.zeros((D, H, W), device=device)

    # Compute valid patch starting indices
    z_idx = list(range(0, D - pD + 1, sD))
    y_idx = list(range(0, H - pH + 1, sH))
    x_idx = list(range(0, W - pW + 1, sW))

    # Add boundary patches if needed
    if z_idx[-1] != D - pD: z_idx.append(D - pD)
    if y_idx[-1] != H - pH: y_idx.append(H - pH)
    if x_idx[-1] != W - pW: x_idx.append(W - pW)

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
            batch_coord_patches = [] # <--- Store coords here
            batch_noises = []
            
            for z, y, x in batch_coords:
                # 1. Extract CBCT Patch
                patch = cbct_t[z:z+pD, y:y+pH, x:x+pW]
                batch_cbct_patches.append(patch)
                
                # 2. Extract Noise Patch
                noise = noise_full[:, :, z:z+pD, y:y+pH, x:x+pW].squeeze(0)
                batch_noises.append(noise)

                # 3. GENERATE COORDINATES ON THE FLY
                # Create meshgrid for this specific patch location
                z_range = np.arange(z, z + pD, dtype=np.float32)
                y_range = np.arange(y, y + pH, dtype=np.float32)
                x_range = np.arange(x, x + pW, dtype=np.float32)

                # Normalize to [-1, 1] using FULL volume dimensions (D,H,W)
                z_norm = 2.0 * z_range / (D - 1) - 1.0
                y_norm = 2.0 * y_range / (H - 1) - 1.0
                x_norm = 2.0 * x_range / (W - 1) - 1.0

                Z_grid, Y_grid, X_grid = np.meshgrid(z_norm, y_norm, x_norm, indexing='ij')
                coord_patch = np.stack([Z_grid, Y_grid, X_grid], axis=0) # [3, pD, pH, pW]
                
                # Convert to tensor
                batch_coord_patches.append(torch.from_numpy(coord_patch).float().to(device))
            
            # Stack batches
            batch_cbct = torch.stack(batch_cbct_patches, dim=0).unsqueeze(1) # (B, 1, ...)
            batch_noise = torch.stack(batch_noises, dim=0).unsqueeze(1)      # (B, 1, ...)
            batch_coords = torch.stack(batch_coord_patches, dim=0)           # (B, 3, ...)

            # Concatenate: Noise(1) + CBCT(1) + Coords(3) = 5 Channels
            x_in = torch.cat((batch_noise, batch_cbct, batch_coords), dim=1) 
            
            # Run Inference
            x_out = sampler(x_in) # (B, 5, ...)

            # Extract only the CT channel (Channel 0)
            pred_ct_batch = x_out[:, 0, ...]
            
            # Distribute results with GAUSSIAN WEIGHTING
            for i, (z, y, x) in enumerate(batch_coords_list):
                pred_patch = pred_ct_batch[i]
                
                # Accumulate Weighted Prediction
                output_sum[z:z+pD, y:y+pH, x:x+pW] += pred_patch * weight_map
                
                # Accumulate Weights
                output_weights[z:z+pD, y:y+pH, x:x+pW] += weight_map
    
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

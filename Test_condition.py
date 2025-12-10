import os
import sys
import numpy as np
import torch
import SimpleITK as sitk
from tqdm import tqdm
import random
import json

# --------------------------
# 1. Imports (Match Train_condition.py)
# --------------------------
from Model_condition import UNet
from Diffusion_condition import GaussianDiffusionSampler_cond
from datasets_3d import VolumePatchDataset3D
from torch.utils.data import DataLoader


# Add path to metrics (Same as in your Train script)
#sys.path.append("/content/drive/MyDrive/Project_in_Scientific_Computing/metrics") 
from SynthRAD_metrics import ImageMetrics

# Initialize Metrics
metrics = ImageMetrics(debug=False)

# --------------------------
# 2. Configuration (MUST Match Train_condition.py)
# --------------------------
# Paths
dataset_root = "/mnt/asgard0/users/p25_2025/synthRAD2025_Task2_Train/synthRAD2025_Task2_Train/Task2" # Point to Validation/Test folder
save_dir = "./Checkpoints_3D"
output_dir = "./test_results_3d"
os.makedirs(output_dir, exist_ok=True)

# Model Params
patch_size = (32, 64, 64)  # (D, H, W) -> MUST MATCH TRAINING
stride = (16, 32, 32)      # 50% overlap for smoothing
T = 1000                   # MUST MATCH TRAINING
ch = 64
ch_mult = [1, 2, 3, 4]
attn = [2]
num_res_blocks = 2
dropout = 0.3


SEED = 42
torch.manual_seed(SEED)

with open("Checkpoints_3D/test_split.json") as f:
    test_ids = json.load(f)

# Create evaluation dataset using the saved test IDs
eval_dataset = VolumePatchDataset3D(
    root=dataset_root,
    split="test",
    patch_size=patch_size,
    seed=42,
)

# Filter dataset to only those test IDs
eval_dataset.patients = [p for p in eval_dataset.patients if p["pid"] in test_ids]
eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------
# 3. Helpers
# --------------------------
def norm_hu_inference(arr: np.ndarray, lo: float = -1000, hi: float = 2000) -> np.ndarray:
    """Normalize HU [-1000, 2000] to [-1, 1]"""
    #arr = np.clip(arr, lo, hi)   <--- REMOVING CLIPPING
    return (2.0 * (arr - lo) / (hi - lo) - 1.0)

def denorm_hu(arr):
    """Denormalize [-1, 1] to HU"""
    lo, hi = -1000, 2000
    return ((arr + 1.0) / 2.0) * (hi - lo) + lo

def predict_sliding_window(model, sampler, cbct_vol):
    """
    Performs sliding window inference on a full volume, ensuring all patches 
    are exactly the required training size (32, 64, 64) by carefully handling
    indices at the boundaries.
    """
    D_orig, H_orig, W_orig = cbct_vol.shape
    pD, pH, pW = patch_size
    sD, sH, sW = stride
    
    # Track sum and count for averaging overlaps
    output_sum = torch.zeros((D_orig, H_orig, W_orig), device=device)
    output_count = torch.zeros((D_orig, H_orig, W_orig), device=device)
    
    cbct_tensor = torch.from_numpy(cbct_vol).float().to(device)
    
    # New Robust Indexing (Safe: Only generates indices that allow a full-sized patch)
    
    # Indices up to the last full stride step
    dz = list(range(0, D_orig - pD + 1, sD))
    dy = list(range(0, H_orig - pH + 1, sH))
    dx = list(range(0, W_orig - pW + 1, sW))
    
    # Add the final boundary index ONLY if the volume edge hasn't been covered by the stride.
    # The index must be D_orig - pD, ensuring the patch ends exactly at the boundary.
    if not dz or dz[-1] < D_orig - pD:
        dz.append(D_orig - pD)
    if not dy or dy[-1] < H_orig - pH:
        dy.append(H_orig - pH)
    if not dx or dx[-1] < W_orig - pW:
        dx.append(W_orig - pW)
        
    # Clean up (ensure indices are valid, unique, and non-negative)
    dz = sorted(list(set(z for z in dz if z >= 0)))
    dy = sorted(list(set(y for y in dy if y >= 0)))
    dx = sorted(list(set(x for x in dx if x >= 0)))

    patches = sorted(list(set((z, y, x) for z in dz for y in dy for x in dx)))
    print(f"  -> Inferencing {len(patches)} full-sized patches...")
    
    model.eval()
    with torch.no_grad():
        for (z, y, x) in tqdm(patches):
            # 1. Extract Patch (Guaranteed size pD, pH, pW)
            # Since the indexing ensures z+pD <= D_orig, this extracts a full patch.
            patch_cbct = cbct_tensor[z:z+pD, y:y+pH, x:x+pW]
            patch_cbct = patch_cbct.unsqueeze(0).unsqueeze(0) # [1, 1, D, H, W]
            
            # ... (2. Diffusion Sampling and 3. Accumulate remains the same) ...
            #noise = torch.randn_like(patch_cbct, generator=torch.Generator(device=device).manual_seed(SEED))
            # Older PyTorch compatibility: randn_like() cannot accept a generator
            g = torch.Generator(device=device)
            g.manual_seed(SEED)
            noise = torch.randn(patch_cbct.shape, dtype=patch_cbct.dtype, device=device, generator=g)

            x_in = torch.cat((noise, patch_cbct), dim=1) 
            
            x_out = sampler(x_in)
            
            pred_patch = x_out[:, 0, :, :, :] 
            
            output_sum[z:z+pD, y:y+pH, x:x+pW] += pred_patch.squeeze(0)
            output_count[z:z+pD, y:y+pH, x:x+pW] += 1.0

    # Average and return
    avg_vol = output_sum / output_count
    return avg_vol.cpu().numpy()
    
# --------------------------
# 4. Main Test Loop
# --------------------------
def main():
    # A. Load Model
    print("Loading Model...")
    model = UNet(T, ch, ch_mult, attn, num_res_blocks, dropout).to(device)
    
    ckpt_path = os.path.join(save_dir, "model_final.pt") # Or best_model.pt
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    
    sampler = GaussianDiffusionSampler_cond(model, 1e-4, 0.02, T).to(device)
    
    # B. Find Patients
    # Assumes structure: root/Cohort/PatientID/{cbct.mha, ct.mha, mask.mha}

    cohorts = ["AB", "HN", "TH"]
    patients = []

    for cohort in cohorts:
        cohort_dir = os.path.join(dataset_root, cohort)
        if not os.path.isdir(cohort_dir):
            continue
        for pid in os.listdir(cohort_dir):
            p_dir = os.path.join(cohort_dir, pid)
            if os.path.isdir(p_dir):
                patients.append((cohort, pid, p_dir))


    print(f"Found {len(patients)} patients in {dataset_root}")

    results = []

    for cohort, pid, p_dir in patients:
        print(f"\nProcessing ({cohort}, {pid})...")

        #p_dir = os.path.join(dataset_root, pid)
        
        # Paths
        cbct_path = os.path.join(p_dir, "cbct.mha")
        ct_path = os.path.join(p_dir, "ct.mha")
        mask_path = os.path.join(p_dir, "mask.mha")
        
        if not os.path.exists(cbct_path):
            print("  Skipping: Missing CBCT.")
            continue

        has_gt = os.path.exists(ct_path)
        if not has_gt:
            print("  No CT ground truth found — running inference only.")


        # C. Load Data
        cbct_img = sitk.ReadImage(cbct_path)
        #ct_img = sitk.ReadImage(ct_path)

        cbct_arr = sitk.GetArrayFromImage(cbct_img) # [D, H, W]
        #gt_arr = sitk.GetArrayFromImage(ct_img)

        # ---- this was added ------------------------
        if has_gt:
            ct_img = sitk.ReadImage(ct_path)
            gt_arr = sitk.GetArrayFromImage(ct_img)
        else:
            gt_arr = None
        # ---------------------------------------------
        
        
        # Load Mask (if exists, else None)
        #if os.path.exists(mask_path):
        #    mask_img = sitk.ReadImage(mask_path)
        #    mask_arr = sitk.GetArrayFromImage(mask_img)
        #else:
        #    print("  Warning: No mask found. Metrics will use full volume.")
        #    mask_arr = np.ones_like(gt_arr)

        if os.path.exists(mask_path) and has_gt:
            mask_img = sitk.ReadImage(mask_path)
            mask_arr = sitk.GetArrayFromImage(mask_img)
        elif has_gt:
            print("  Warning: No mask found. Metrics will use full volume.")
            mask_arr = np.ones_like(gt_arr, dtype = np.uint8)
        else:
            mask_arr = None

        # D. Normalize & Predict
        cbct_norm = norm_hu_inference(cbct_arr)
        pred_norm = predict_sliding_window(model, sampler, cbct_norm)
        
        # E. Denormalize
        pred_hu = denorm_hu(pred_norm)

        # F. Compute Metrics (Using your custom class)
        # Note: score_patient expects (GT, Pred, Mask)
        #scores = metrics.score_patient(gt_arr, pred_hu, mask_arr)
        if has_gt:
            scores = metrics.score_patient(gt_arr, pred_hu, mask_arr)
            print(f"  Result: MAE={scores['mae']:.2f} | PSNR={scores['psnr']:.2f} | MS-SSIM={scores['ms_ssim']:.4f}")
            results.append(scores)
        else:
            print("  (No ground truth: metrics skipped)")
        
        #print(f"  Result: MAE={scores['mae']:.2f} | PSNR={scores['psnr']:.2f} | MS-SSIM={scores['ms_ssim']:.4f}")
        #results.append(scores)

        # G. Save Prediction
        out_img = sitk.GetImageFromArray(pred_hu)
        out_img.CopyInformation(cbct_img) # Copy geometry
        sitk.WriteImage(out_img, os.path.join(output_dir, f"{pid}_sct.mha"))

    # Summary
    if results:
        avg_mae = np.mean([r['mae'] for r in results])
        avg_ssim = np.mean([r['ms_ssim'] for r in results])
        avg_psnr = np.mean([r['psnr'] for r in results])
        print(f"\nFinal Average: MAE={avg_mae:.2f} | MS-SSIM={avg_ssim:.4f} | PSNR={avg_psnr:.2f}")

if __name__ == "__main__":
    main()

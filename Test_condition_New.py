import os
import sys
import numpy as np
import torch
import SimpleITK as sitk
from tqdm import tqdm

# --------------------------
# 1. Imports (Match Train_condition.py)
# --------------------------
from Model_condition import UNet
from Diffusion_condition import GaussianDiffusionSampler_cond

# Add path to metrics (Same as in your Train script)
sys.path.append("/content/drive/MyDrive/Project_in_Scientific_Computing/metrics") 
from functions.image_metrics import ImageMetrics

# Initialize Metrics
metrics = ImageMetrics(debug=False)

# --------------------------
# 2. Configuration (MUST Match Train_condition.py)
# --------------------------
# Paths
dataset_root = "/content/drive/MyDrive/Project_in_Scientific_Computing/playground/val" # Point to Validation/Test folder
save_dir = "./Checkpoints_3D"
output_dir = "./test_results_3d"
os.makedirs(output_dir, exist_ok=True)

# Model Params
patch_size = (32, 64, 64)  # (D, H, W) -> MUST MATCH TRAINING
stride = (16, 32, 32)      # 50% overlap for smoothing
T = 1000                   # MUST MATCH TRAINING
ch = 64
ch_mult = [1, 2, 4]
attn = [1]
num_res_blocks = 2
dropout = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------
# 3. Helpers
# --------------------------
def norm_hu(arr):
    """Normalize HU [-1000, 2000] to [-1, 1]"""
    lo, hi = -1000, 2000
    arr = np.clip(arr, lo, hi)
    return (2.0 * (arr - lo) / (hi - lo) - 1.0)

def denorm_hu(arr):
    """Denormalize [-1, 1] to HU"""
    lo, hi = -1000, 2000
    return ((arr + 1.0) / 2.0) * (hi - lo) + lo

def predict_sliding_window(model, sampler, cbct_vol):
    """
    Performs sliding window inference on a full volume.
    cbct_vol: [D, H, W] numpy array (normalized)
    """
    D, H, W = cbct_vol.shape
    pD, pH, pW = patch_size
    sD, sH, sW = stride
    
    # Track sum and count for averaging overlaps
    output_sum = torch.zeros((D, H, W), device=device)
    output_count = torch.zeros((D, H, W), device=device)
    
    cbct_tensor = torch.from_numpy(cbct_vol).float().to(device)
    
    # Generate grid coordinates
    dz = list(range(0, D - pD + 1, sD)) + ([D-pD] if (D-pD) % sD != 0 else [])
    dy = list(range(0, H - pH + 1, sH)) + ([H-pH] if (H-pH) % sH != 0 else [])
    dx = list(range(0, W - pW + 1, sW)) + ([W-pW] if (W-pW) % sW != 0 else [])
    
    # Create unique patch list
    patches = sorted(list(set((z, y, x) for z in dz for y in dy for x in dx)))
    print(f"  -> Inferencing {len(patches)} patches...")
    
    model.eval()
    with torch.no_grad():
        for (z, y, x) in tqdm(patches):
            # 1. Extract Patch
            patch_cbct = cbct_tensor[z:z+pD, y:y+pH, x:x+pW]
            patch_cbct = patch_cbct.unsqueeze(0).unsqueeze(0) # [1, 1, D, H, W]
            
            # 2. Diffusion Sampling
            noise = torch.randn_like(patch_cbct)
            x_in = torch.cat((noise, patch_cbct), dim=1) # [1, 2, D, H, W]
            
            # Sampler returns [1, 2, D, H, W] (CT, CBCT)
            x_out = sampler(x_in)
            
            # Extract CT channel only
            pred_patch = x_out[:, 0, :, :, :] # [1, D, H, W]
            
            # 3. Accumulate
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
    # Or root/PatientID/... depending on your folder structure
    # Adapting to your 'playground' structure (likely root/PatientID directly)
    patients = sorted([p for p in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, p))])
    print(f"Found {len(patients)} patients in {dataset_root}")

    results = []

    for pid in patients:
        print(f"\nProcessing {pid}...")
        p_dir = os.path.join(dataset_root, pid)
        
        # Paths
        cbct_path = os.path.join(p_dir, "cbct.mha")
        ct_path = os.path.join(p_dir, "ct.mha")
        mask_path = os.path.join(p_dir, "mask.mha")
        
        if not (os.path.exists(cbct_path) and os.path.exists(ct_path)):
            print("  Skipping: Missing files.")
            continue

        # C. Load Data
        cbct_img = sitk.ReadImage(cbct_path)
        ct_img = sitk.ReadImage(ct_path)
        
        cbct_arr = sitk.GetArrayFromImage(cbct_img) # [D, H, W]
        gt_arr = sitk.GetArrayFromImage(ct_img)
        
        # Load Mask (if exists, else None)
        if os.path.exists(mask_path):
            mask_img = sitk.ReadImage(mask_path)
            mask_arr = sitk.GetArrayFromImage(mask_img)
        else:
            print("  Warning: No mask found. Metrics will use full volume.")
            mask_arr = np.ones_like(gt_arr)

        # D. Normalize & Predict
        cbct_norm = norm_hu(cbct_arr)
        pred_norm = predict_sliding_window(model, sampler, cbct_norm)
        
        # E. Denormalize
        pred_hu = denorm_hu(pred_norm)

        # F. Compute Metrics (Using your custom class)
        # Note: score_patient expects (GT, Pred, Mask)
        scores = metrics.score_patient(gt_arr, pred_hu, mask_arr)
        
        print(f"  Result: MAE={scores['mae']:.2f} | PSNR={scores['psnr']:.2f} | MS-SSIM={scores['ms_ssim']:.4f}")
        results.append(scores)

        # G. Save Prediction
        out_img = sitk.GetImageFromArray(pred_hu)
        out_img.CopyInformation(cbct_img) # Copy geometry
        sitk.WriteImage(out_img, os.path.join(output_dir, f"{pid}_sct.mha"))

    # Summary
    if results:
        avg_mae = np.mean([r['mae'] for r in results])
        avg_ssim = np.mean([r['ms_ssim'] for r in results])
        print(f"\nFinal Average: MAE={avg_mae:.2f} | MS-SSIM={avg_ssim:.4f}")

if __name__ == "__main__":
    main()

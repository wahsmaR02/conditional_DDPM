# Test_condition.py
import os
import json
import numpy as np
import torch
import SimpleITK as sitk
from tqdm import tqdm
from torch.utils.data import DataLoader

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

patch_size = (32, 64, 64)
stride = (16, 32, 32)  # overlap sliding window
T = 1000
ch = 64
ch_mult = [1, 2, 3, 4]
attn = [2]
num_res_blocks = 2
dropout = 0.3

metrics = ImageMetrics(debug=False)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# --------------------------
# Utility functions
# --------------------------

def norm_hu(x):
    lo, hi = -1000, 2000
    x = np.clip(x, lo, hi)
    return 2 * (x - lo) / (hi - lo) - 1


def denorm_hu(x):
    lo, hi = -1000, 2000
    return ((x + 1) / 2) * (hi - lo) + lo


def sliding_window_inference(model, sampler, cbct_norm, device):
    """
    Runs sliding-window inference over the full CBCT volume.
    """
    D, H, W = cbct_norm.shape
    pD, pH, pW = patch_size
    sD, sH, sW = stride

    cbct_t = torch.from_numpy(cbct_norm).float().to(device)

    # Added this for deterministic noise
    noise_generator = torch.Generator(device=device).manual_seed(SEED)

    output_sum  = torch.zeros((D, H, W), device=device)
    output_count = torch.zeros((D, H, W), device=device)

    # Compute valid patch starting indices
    z_idx = list(range(0, D - pD + 1, sD))
    y_idx = list(range(0, H - pH + 1, sH))
    x_idx = list(range(0, W - pW + 1, sW))

    # Add boundary patches if needed
    if z_idx[-1] != D - pD: z_idx.append(D - pD)
    if y_idx[-1] != H - pH: y_idx.append(H - pH)
    if x_idx[-1] != W - pW: x_idx.append(W - pW)

    patches = [(z, y, x) for z in z_idx for y in y_idx for x in x_idx]
    patches = sorted(set(patches))

    print(f"  -> Running {len(patches)} patches...")

    model.eval()
    with torch.no_grad():
        for z, y, x in tqdm(patches):
            patch_cbct = cbct_t[z:z+pD, y:y+pH, x:x+pW].unsqueeze(0).unsqueeze(0)

            # Generate deterministic noise   <----- removing this here (Cissi 10/12)
            #g = torch.Generator(device=device)# <-- DELETE
            #g.manual_seed(SEED)# <-- DELETE
            noise = torch.randn(patch_cbct.shape, device=device, generator=noise_generator) # <-- REPLACED with persistent generator)

            x_in = torch.cat((noise, patch_cbct), dim=1)
            x_out = sampler(x_in)

            pred_patch = x_out[:, 0, :, :, :].squeeze(0)

            output_sum[z:z+pD, y:y+pH, x:x+pW] += pred_patch
            output_count[z:z+pD, y:y+pH, x:x+pW] += 1

    result = output_sum / output_count
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
        test_ids = set(json.load(f))

    print(f"Loaded {len(test_ids)} test IDs from JSON.")

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

    # Filter to ensure exact match with saved split
    test_dataset.patients = [
        p for p in test_dataset.patients if p["pid"] in test_ids
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
        pred_norm = sliding_window_inference(model, sampler, cbct_norm, device)
        pred_hu   = denorm_hu(pred_norm)

        # Evaluate metrics
        mae = metrics.mae(gt, pred_hu, mask)
        print(f"  → MAE = {mae:.2f}")

        results.append({"pid": pid, "mae": mae})

        # Save synthetic CT
        out_img = sitk.GetImageFromArray(pred_hu)
        out_img.CopyInformation(sitk.ReadImage(cbct_path))
        sitk.WriteImage(out_img, os.path.join(output_dir, f"{pid}_sct.mha"))

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

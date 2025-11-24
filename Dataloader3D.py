import os
import glob
import random
import argparse
import numpy as np
import SimpleITK as sitk

# ======================================
# HU normalization
# ======================================
def norm_hu(arr, lo=-1000, hi=2000):
    """
    Clip Hounsfield units to [lo, hi] and normalize to [-1,1].
    -1000 ~ air, +2000 ~ dense bone.
    """
    arr = np.clip(arr, lo, hi)
    return (2.0 * (arr - lo) / (hi - lo) - 1.0).astype(np.float32)


# ======================================
# Patient discovery and splitting
# ======================================
def collect_patients(root, cohorts=("AB", "HN", "TH")):
    """
    Find all patient folders under root/<cohort>/*.
    Returns a list of dicts with {cohort, pid, path}.
    """
    patients = []
    for cohort in cohorts:
        cohort_dir = os.path.join(root, cohort)
        if not os.path.isdir(cohort_dir):
            print(f"⚠️ Cohort folder missing: {cohort_dir}")
            continue
        for pid in sorted(os.listdir(cohort_dir)):
            p_dir = os.path.join(cohort_dir, pid)
            if not os.path.isdir(p_dir):
                continue
            patients.append({"cohort": cohort, "pid": pid, "path": p_dir})
    return patients


def split_patients(patients, train_frac=0.8, seed=42):
    """
    Patient-wise random split into train / val.
    """
    random.seed(seed)
    random.shuffle(patients)
    N = len(patients)
    n_train = int(round(train_frac * N))
    train_pat = patients[:n_train]
    val_pat   = patients[n_train:]
    return train_pat, val_pat


# ======================================
# Export full 3D volumes as .nii.gz (HU-normalized)
# ======================================
def export_patient_volumes(pinfo, out_root, split):
    """
    For one patient:
      - load cbct.mha, ct.mha, mask.mha
      - HU-normalize CT + CBCT
      - save as 3D .nii.gz under:
            out_root/split/a (CT)
            out_root/split/b (CBCT)
            out_root/split/m (MASK)
    """
    cohort = pinfo["cohort"]
    pid    = pinfo["pid"]
    p_dir  = pinfo["path"]

    cbct_path = os.path.join(p_dir, "cbct.mha")
    ct_path   = os.path.join(p_dir, "ct.mha")
    mask_path = os.path.join(p_dir, "mask.mha")

    if not (os.path.exists(cbct_path) and os.path.exists(ct_path) and os.path.exists(mask_path)):
        print(f"⚠️ Missing files for {cohort}/{pid}, skipping.")
        return

    # Load raw images
    cbct_img = sitk.ReadImage(cbct_path)
    ct_img   = sitk.ReadImage(ct_path)
    mask_img = sitk.ReadImage(mask_path)

    cbct_arr = sitk.GetArrayFromImage(cbct_img).astype(np.float32)  # [D,H,W]
    ct_arr   = sitk.GetArrayFromImage(ct_img).astype(np.float32)
    mask_arr = sitk.GetArrayFromImage(mask_img).astype(np.uint8)

     # 👉 Print shapes here
    print(f"\nPatient {cohort}/{pid}:")
    print(f"  CT   shape  = {ct_arr.shape}  (D,H,W)")
    print(f"  CBCT shape  = {cbct_arr.shape}  (D,H,W)")
    print(f"  MASK shape  = {mask_arr.shape}  (D,H,W)")

    # HU-normalize CT and CBCT
    cbct_norm = norm_hu(cbct_arr)
    ct_norm   = norm_hu(ct_arr)

    # Create new SITK images from normalized arrays
    cbct_norm_img = sitk.GetImageFromArray(cbct_norm)
    ct_norm_img   = sitk.GetImageFromArray(ct_norm)
    mask_out_img  = sitk.GetImageFromArray(mask_arr)

    # Preserve original spatial metadata
    cbct_norm_img.CopyInformation(cbct_img)
    ct_norm_img.CopyInformation(ct_img)
    mask_out_img.CopyInformation(mask_img)

    # Output dirs
    out_a = os.path.join(out_root, split, "a")  # CT
    out_b = os.path.join(out_root, split, "b")  # CBCT
    out_m = os.path.join(out_root, split, "m")  # MASK
    os.makedirs(out_a, exist_ok=True)
    os.makedirs(out_b, exist_ok=True)
    os.makedirs(out_m, exist_ok=True)

    # Keep patient name (no cohort prefix if you prefer)
    base = pid

    sitk.WriteImage(ct_norm_img,   os.path.join(out_a, f"{base}_ct.nii.gz"))
    sitk.WriteImage(cbct_norm_img, os.path.join(out_b, f"{base}_cbct.nii.gz"))
    sitk.WriteImage(mask_out_img,  os.path.join(out_m, f"{base}_mask.nii.gz"))

    print(f"✅ Saved 3D volumes for {cohort}/{pid} → {split}")


def export_all_volumes(root, out_root, train_frac=0.8, seed=42):
    patients = collect_patients(root)
    if len(patients) == 0:
        raise RuntimeError(f"No patients found under {root}")

    print(f"📁 Found {len(patients)} patients")
    train_pat, val_pat = split_patients(patients, train_frac=train_frac, seed=seed)
    print(f"  Train: {len(train_pat)} patients")
    print(f"  Val:   {len(val_pat)} patients")

    print("\n▶ Exporting TRAIN volumes...")
    for p in train_pat:
        export_patient_volumes(p, out_root=out_root, split="train")

    print("\n▶ Exporting VAL volumes...")
    for p in val_pat:
        export_patient_volumes(p, out_root=out_root, split="val")

    print("\n✅ Volume export done.")
    return train_pat, val_pat


# ======================================
# Patch extraction from TRAIN set only
# ======================================
def extract_random_patches_for_train(out_root,
                                     patch_size=(64, 128, 128),
                                     patches_per_patient=10,
                                     seed=42,
                                     max_tries=100):
    """
    From the already-saved train volumes (HU-normalized), extract random 3D patches.

    patch_size: (D, H, W) = e.g. (64, 128, 128)
    - Pick random corner with RNG seeded by `seed`
    - Ensure center voxel of patch is inside the mask
    - If not, retry (while loop) up to max_tries
    - Save patches as .npz under out_root/patches/train/
    """
    rng = np.random.RandomState(seed)

    ct_dir   = os.path.join(out_root, "train", "a")
    cbct_dir = os.path.join(out_root, "train", "b")
    mask_dir = os.path.join(out_root, "train", "m")

    ct_files = sorted(glob.glob(os.path.join(ct_dir, "*_ct.nii.gz")))
    os.makedirs(os.path.join(out_root, "patches", "train"), exist_ok=True)

    pd, ph, pw = patch_size  # (D, H, W)

    for ct_path in ct_files:
        base = os.path.basename(ct_path).replace("_ct.nii.gz", "")
        cbct_path = os.path.join(cbct_dir, f"{base}_cbct.nii.gz")
        mask_path = os.path.join(mask_dir, f"{base}_mask.nii.gz")

        if not (os.path.exists(cbct_path) and os.path.exists(mask_path)):
            print(f"⚠️ Missing cbct/mask for {base}, skipping patches.")
            continue

        # Load volumes (these are already HU-normalized)
        ct_img   = sitk.ReadImage(ct_path)
        cbct_img = sitk.ReadImage(cbct_path)
        mask_img = sitk.ReadImage(mask_path)

        ct   = sitk.GetArrayFromImage(ct_img).astype(np.float32)   # [D,H,W]
        cbct = sitk.GetArrayFromImage(cbct_img).astype(np.float32)
        mask = sitk.GetArrayFromImage(mask_img).astype(np.uint8)

        D, H, W = ct.shape

        # Ensure volume is large enough
        if D < pd or H < ph or W < pw:
            print(f"⚠️ Volume too small for patch size for {base}, skipping.")
            continue

        print(f"🔹 Extracting patches for {base} ...")

        for p_idx in range(patches_per_patient):
            tries = 0
            patch_ct = patch_cbct = patch_mask = None

            while True:
                tries += 1
                if tries > max_tries:
                    print(f"  ⚠️ Could not find valid foreground patch for {base} (patch {p_idx}). Skipping.")
                    break

                # Random corner (inclusive range)
                d0 = rng.randint(0, D - pd + 1)
                h0 = rng.randint(0, H - ph + 1)
                w0 = rng.randint(0, W - pw + 1)

                patch_ct   = ct[d0:d0+pd, h0:h0+ph, w0:w0+pw]
                patch_cbct = cbct[d0:d0+pd, h0:h0+ph, w0:w0+pw]
                patch_mask = mask[d0:d0+pd, h0:h0+ph, w0:w0+pw]

                # Center voxel in patch
                cd = pd // 2
                ch = ph // 2
                cw = pw // 2

                if patch_mask[cd, ch, cw] > 0:
                    # Valid patch (center in foreground)
                    break

            if patch_ct is None or tries > max_tries:
                continue  # move on to next patch/patient

            # Save patch as .npz (channels not added yet; they are [D,H,W])
            patch_out_dir = os.path.join(out_root, "patches", "train")
            patch_fname   = os.path.join(patch_out_dir, f"{base}_patch{p_idx:03d}.npz")

            np.savez(
                patch_fname,
                ct=patch_ct.astype(np.float32),
                cbct=patch_cbct.astype(np.float32),
                mask=patch_mask.astype(np.uint8),
            )

        print(f"  ✅ Done patches for {base}")


# ======================================
# Main
# ======================================
def main():
    parser = argparse.ArgumentParser(description="Convert MHA to 3D NIfTI with HU norm, split, and extract patches.")
    parser.add_argument("--root", required=True, help="Root folder with AB/HN/TH subfolders (raw .mha)")
    parser.add_argument("--out",  required=True, help="Output folder for .nii.gz volumes and patches")
    parser.add_argument("--train_frac", type=float, default=0.8, help="Train fraction (0-1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split and patch positions")
    parser.add_argument("--patch_D", type=int, default=64, help="Patch depth (D)")
    parser.add_argument("--patch_H", type=int, default=128, help="Patch height (H)")
    parser.add_argument("--patch_W", type=int, default=128, help="Patch width (W)")
    parser.add_argument("--patches_per_patient", type=int, default=10, help="Number of patches per train patient")

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 1) Export full 3D HU-normalized volumes and split into train/val
    export_all_volumes(args.root, args.out, train_frac=args.train_frac, seed=args.seed)

    # 2) Extract random 3D patches from TRAIN only
    patch_size = (args.patch_D, args.patch_H, args.patch_W)
    extract_random_patches_for_train(
        out_root=args.out,
        patch_size=patch_size,
        patches_per_patient=args.patches_per_patient,
        seed=args.seed,
    )

    print("\n🎉 All done: volumes + patches created.")


if __name__ == "__main__":
    main()

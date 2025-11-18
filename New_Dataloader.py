import os, glob, random
import numpy as np
import SimpleITK as sitk

# ==============================
# USER SETTINGS (edit once)
# ==============================

# Root folder with HN/TH/AB subfolders (each with patient folders)
ROOT_DIR = "synthRAD2025_Task2_Train/playground"  # <-- CHANGE THIS

# Output folder for final .nii.gz slices
OUT_DIR = "./Sliced_nii"

# Target slice size (H = W)
SLICE_SIZE = 256

# How many slices to skip at each end of the volume
SKIP_SLICES = 5

# Cohorts present in your dataset
COHORTS = ("HN", "TH", "AB")

# Random seed for reproducible patient-wise split
SPLIT_SEED = 42


# ==============================
# Preprocessing helpers
# ==============================

def norm_hu(arr, lo=-1000, hi=2000):
    """
    Clip Hounsfield units to [lo, hi] and scale to [-1,1].
    -1000 ~ air, up to ~2000 ~ dense bone.
    """
    arr = np.clip(arr, lo, hi)
    return (2.0 * (arr - lo) / (hi - lo) - 1.0).astype(np.float32)


def pad_or_crop_to(arr, h=256, w=256):
    """
    Center-crop or pad a 2D slice to size (h, w).
    """
    H, W = arr.shape

    # Center crop
    y0 = max(0, (H - h) // 2); y1 = y0 + min(H, h)
    x0 = max(0, (W - w) // 2); x1 = x0 + min(W, w)
    cropped = arr[y0:y1, x0:x1]

    # Pad if needed
    ph = max(0, h - cropped.shape[0])
    pw = max(0, w - cropped.shape[1])
    top = ph // 2; bottom = ph - top
    left = pw // 2; right = pw - left

    if ph > 0 or pw > 0:
        cropped = np.pad(cropped, ((top, bottom), (left, right)), mode="edge")

    return cropped.astype(np.float32)


def crop_with_mask(img, mask):
    """
    Crop img to the bounding box of mask (nonzero region).
    If mask is empty, return img unchanged.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        # Empty mask → no cropping
        return img

    ymin, ymax = ys.min(), ys.max()
    xmin, xmax = xs.min(), xs.max()

    return img[ymin:ymax+1, xmin:xmax+1]


# ==============================
# Patient discovery & splitting
# ==============================

def collect_patients(root, cohorts=COHORTS):
    """
    Find all patient folders under root/<cohort>/*.
    Returns a list of dicts with cohort, pid, path.
    """
    patients = []
    for cohort in cohorts:
        cohort_dir = os.path.join(root, cohort)
        if not os.path.isdir(cohort_dir):
            continue
        for p in sorted(os.listdir(cohort_dir)):
            p_dir = os.path.join(cohort_dir, p)
            if not os.path.isdir(p_dir):
                continue
            patients.append({
                "cohort": cohort,
                "pid": p,
                "path": p_dir,
            })
    return patients


def split_patients(patients, train_frac=0.65, val_frac=0.10, seed=42):
    """
    Patient-wise random split into train/val/test.
    """
    random.seed(seed)
    random.shuffle(patients)

    N = len(patients)
    n_train = int(round(train_frac * N))
    n_val   = int(round(val_frac * N))
    n_test  = N - n_train - n_val

    train_pat = patients[:n_train]
    val_pat   = patients[n_train:n_train + n_val]
    test_pat  = patients[n_train + n_val:]

    return train_pat, val_pat, test_pat


# ==============================
# Slicing & export (with mask)
# ==============================

def slice_and_export_patient(pinfo, out_root, split, size=256, skip=5):
    """
    For one patient:
      - load cbct.mha, ct.mha, mask.mha
      - normalize CT/CBCT to [-1,1]
      - loop over slices [skip, Z-skip)
      - crop each slice by mask bounding box
      - pad/crop to size×size
      - save as .nii.gz under out_root/split/a,b
    """
    cohort = pinfo["cohort"]
    pid    = pinfo["pid"]
    p_dir  = pinfo["path"]

    cbct_path = os.path.join(p_dir, "cbct.mha")
    ct_path   = os.path.join(p_dir, "ct.mha")
    mask_path = os.path.join(p_dir, "mask.mha")

    if not (os.path.exists(cbct_path) and os.path.exists(ct_path) and os.path.exists(mask_path)):
        print(f"⚠️  Missing cbct.mha / ct.mha / mask.mha for {cohort}/{pid}, skipping.")
        return

    # Load volumes [Z, H, W]
    cbct = sitk.GetArrayFromImage(sitk.ReadImage(cbct_path)).astype(np.float32)
    ct   = sitk.GetArrayFromImage(sitk.ReadImage(ct_path)).astype(np.float32)
    mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path)).astype(np.uint8)

    if cbct.shape != ct.shape or cbct.shape != mask.shape:
        print(f"⚠️  Shape mismatch for {cohort}/{pid}, skipping.")
        return

    # Normalize to [-1,1]
    cbct = norm_hu(cbct)
    ct   = norm_hu(ct)

    Z = cbct.shape[0]

    out_a = os.path.join(out_root, split, "a")  # CT
    out_b = os.path.join(out_root, split, "b")  # CBCT
    os.makedirs(out_a, exist_ok=True)
    os.makedirs(out_b, exist_ok=True)

    base_id = f"{cohort}_{pid}"

    # Skip slices at beginning and end
    start_z = max(skip, 0)
    end_z   = max(Z - skip, start_z + 1)  # ensure at least 1 slice if small
    for z in range(start_z, end_z):
        slice_cbct = cbct[z]
        slice_ct   = ct[z]
        slice_mask = mask[z]

        # Crop using mask
        slice_cbct = crop_with_mask(slice_cbct, slice_mask)
        slice_ct   = crop_with_mask(slice_ct,   slice_mask)

        # Then pad/crop to target size
        slice_cbct = pad_or_crop_to(slice_cbct, size, size)
        slice_ct   = pad_or_crop_to(slice_ct,   size, size)

        # Convert back to SimpleITK images
        cbct_img = sitk.GetImageFromArray(slice_cbct)
        ct_img   = sitk.GetImageFromArray(slice_ct)

        fname = f"{base_id}_z{z:03d}.nii.gz"
        sitk.WriteImage(cbct_img, os.path.join(out_b, fname))
        sitk.WriteImage(ct_img,   os.path.join(out_a, fname))


def export_all(root, out, size=256, cohorts=COHORTS, skip=SKIP_SLICES, seed=SPLIT_SEED):
    """
    Collect patients, split them 70/15/15 by patient, and
    export slices for train/val/test under out/.
    """
    patients = collect_patients(root, cohorts=cohorts)
    if len(patients) == 0:
        raise RuntimeError(f"No patients found under {root} (cohorts={cohorts})")

    print(f"📁 Found {len(patients)} patients total.")
    train_pat, val_pat, test_pat = split_patients(
        patients,
        train_frac=0.7,
        val_frac=0.15,
        seed=seed,
    )
    print(f" Train: {len(train_pat)} patients")
    print(f" Val:   {len(val_pat)} patients")
    print(f" Test:  {len(test_pat)} patients")

    splits = {
        "train": train_pat,
        "val":   val_pat,
        "test":  test_pat,
    }

    for split_name, plist in splits.items():
        print(f"\n🔄 Exporting {split_name} set ({len(plist)} patients)…")
        for pinfo in plist:
            slice_and_export_patient(
                pinfo,
                out_root=out,
                split=split_name,
                size=size,
                skip=skip,
            )

    print("\n Done!")
    print(f"   Slices saved as .nii.gz under:")
    print(f"   {out}/train/a,b")
    print(f"   {out}/val/a,b")
    print(f"   {out}/test/a,b")


# ==============================
# Run once
# ==============================

if __name__ == "__main__":
    export_all(
        root=ROOT_DIR,
        out=OUT_DIR,
        size=SLICE_SIZE,
        cohorts=COHORTS,
        skip=SKIP_SLICES,
        seed=SPLIT_SEED,
    )

import os, glob, random
import numpy as np
import torch
import SimpleITK as sitk
from torch.utils.data import Dataset

# HU clipping + normalization

def norm_hu(arr, lo=-1000, hi=1024):
    """
    Clip HU values to [lo, hi], then map to [-1, 1].
    """
    arr = np.clip(arr, lo, hi)
    return (2.0 * (arr - lo) / (hi - lo) - 1.0).astype(np.float32)


# Patient collecting

def collect_patients(root, cohorts=("HN", "TH", "AB")):
    patients = []
    for cohort in cohorts:
        cdir = os.path.join(root, cohort)
        if not os.path.isdir(cdir):
            continue
        for p in sorted(os.listdir(cdir)):
            pdir = os.path.join(cdir, p)
            if os.path.isdir(pdir):
                patients.append({"cohort": cohort, "pid": p, "path": pdir})
    return patients

# Patient splitting

def split_patients(patients, train_frac=0.8, val_frac=0.2, seed=42):
    rng = random.Random(seed)
    patients = patients.copy()
    rng.shuffle(patients)

    N = len(patients)
    n_train = int(round(train_frac * N))
    n_val   = int(round(val_frac * N))

    return patients[:n_train], patients[n_train:n_train+n_val]


# Final 3D Patch-Based Dataset

class ImageDatasetNii3D(Dataset):
    """
    Loads full CT/CBCT/mask volumes and extracts random 3D patches.

    Requirements from supervisor:
    - No slicing.
    - No padding.
    - No full-volume cropping.
    - Patch extraction only.
    - Center voxel must be foreground in mask.
    """

    def __init__(
        self,
        root,
        split="train",
        cohorts=("HN", "TH", "AB"),
        patch_size=(64, 128, 128),
        max_attempts=50,
        seed=42,
    ):
        super().__init__()

        self.patch_D, self.patch_H, self.patch_W = patch_size
        self.max_attempts = max_attempts

        # Discover + split patients
        all_patients = collect_patients(root, cohorts)
        train_pat, val_pat = split_patients(all_patients, seed=seed)
        self.patients = train_pat if split == "train" else val_pat

        if len(self.patients) == 0:
            raise RuntimeError(f"No {split} patients found under {root}")

    def __len__(self):
        return len(self.patients)

    # Volume loading

    def load_volume(self, pinfo):
        """
        Load cbct.mha, ct.mha, mask.mha for ONE patient.
        Returns 3 numpy arrays of shape [D, H, W].
        """
        pdir = pinfo["path"]
        cb_path = os.path.join(pdir, "cbct.mha")
        ct_path = os.path.join(pdir, "ct.mha")
        m_path  = os.path.join(pdir, "mask.mha")

        if not (os.path.exists(cb_path) and os.path.exists(ct_path) and os.path.exists(m_path)):
            raise FileNotFoundError(f"Missing files in {pdir}")

        cbct = sitk.GetArrayFromImage(sitk.ReadImage(cb_path)).astype(np.float32)
        ct   = sitk.GetArrayFromImage(sitk.ReadImage(ct_path)).astype(np.float32)
        mask = sitk.GetArrayFromImage(sitk.ReadImage(m_path)).astype(np.uint8)

        print("Loaded volume shape:", ct.shape, "Patient:", pinfo["pid"])

        if not (cbct.shape == ct.shape == mask.shape):
            raise ValueError(f"Shape mismatch in {pdir}")

        # Only HU clipping + normalization
        cbct = norm_hu(cbct)
        ct   = norm_hu(ct)

        return cbct, ct, mask

    # Random patch sampling

    def __getitem__(self, idx):
        pinfo = self.patients[idx]
        cbct, ct, mask = self.load_volume(pinfo)

        D, H, W = ct.shape

        for _ in range(self.max_attempts):
            # Random corner
            z0 = random.randint(0, D - self.patch_D)
            y0 = random.randint(0, H - self.patch_H)
            x0 = random.randint(0, W - self.patch_W)

            z1 = z0 + self.patch_D
            y1 = y0 + self.patch_H
            x1 = x0 + self.patch_W

            patch_mask = mask[z0:z1, y0:y1, x0:x1]

            # Center voxel check
            cz = self.patch_D // 2
            cy = self.patch_H // 2
            cx = self.patch_W // 2

            if patch_mask[cz, cy, cx] == 1:
                # Valid patch found
                patch_ct   = ct[z0:z1, y0:y1, x0:x1]
                patch_cbct = cbct[z0:z1, y0:y1, x0:x1]

                return {
                    "CT": torch.from_numpy(patch_ct).unsqueeze(0),
                    "CBCT": torch.from_numpy(patch_cbct).unsqueeze(0),
                    "mask": torch.from_numpy(patch_mask.astype(np.float32)).unsqueeze(0),
                }

        # Fallback: return last random patch even if center not fg
        patch_ct   = ct[z0:z1, y0:y1, x0:x1]
        patch_cbct = cbct[z0:z1, y0:y1, x0:x1]
        patch_mask = mask[z0:z1, y0:y1, x0:x1]

        return {
            "CT": torch.from_numpy(patch_ct).unsqueeze(0),
            "CBCT": torch.from_numpy(patch_cbct).unsqueeze(0),
            "mask": torch.from_numpy(patch_mask.astype(np.float32)).unsqueeze(0),
        }

# datasets_3d.py
#
# 3D volume dataset for SynthRAD Task 2
# - Loads full cbct.mha / ct.mha / mask.mha
# - Performs a 20/20/60 patient split into train/val/test.
# - Samples valid 3D patches where the patch center lies inside the mask.

import os
import random
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk


# ==============================
# Helper functions
# ==============================

def norm_ct(arr, lo=-1024, hi=3000): 
    return (2.0 * (arr - lo) / (hi - lo) - 1.0).astype(np.float32)

def collect_patients(root: str,
                     cohorts: Tuple[str, ...] = ("HN", "TH", "AB")) -> List[Dict]:
    """
    Find all patient folders under root/<cohort>/*.
    Returns entries with paths to cbct.mha, ct.mha, mask.mha.
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

            cbct_path = os.path.join(p_dir, "cbct.mha")
            ct_path   = os.path.join(p_dir, "ct.mha")
            mask_path = os.path.join(p_dir, "mask.mha")

            if not (os.path.exists(cbct_path) and os.path.exists(ct_path) and os.path.exists(mask_path)):
                continue

            patients.append({
                "cohort": cohort,
                "pid": p,
                "path": p_dir,
                "cbct_path": cbct_path,
                "ct_path": ct_path,
                "mask_path": mask_path,
            })

    return patients


def split_patients_train_val_test(
    patients: List[Dict],
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Deterministic patient-wise split into train/val/test.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    rng = random.Random(seed)
    patients_shuffled = patients.copy()
    rng.shuffle(patients_shuffled)

    N = len(patients_shuffled)
    n_train = int(round(train_frac * N))
    n_val   = int(round(val_frac * N))

    # Safety constraints
    n_train = max(1, min(n_train, N - 2))
    n_val   = max(1, min(n_val, N - n_train - 1))

    train_pat = patients_shuffled[:n_train]
    val_pat   = patients_shuffled[n_train:n_train + n_val]
    test_pat  = patients_shuffled[n_train + n_val:]

    return train_pat, val_pat, test_pat


# ==============================
# VolumePatchDataset3D
# ==============================

class VolumePatchDataset3D(Dataset):
    """
    Loads full SynthRAD Task2 volumes (CBCT/CT/mask),
    and returns random 3D patches with center inside the mask.

    __getitem__ returns:
        {
            "CBCT": FloatTensor [1, D, H, W],
            "pCT":  FloatTensor [1, D, H, W],
            "meta": {"cohort", "pid"}
        }
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        patch_size: Tuple[int, int, int] = (96, 128, 128),
        cohorts: Tuple[str, ...] = ("HN", "TH", "AB"),
        train_frac: float = 0.6,
        val_frac: float = 0.2,
        test_frac: float = 0.2,
        seed: int = 42,
        max_tries: int = 50,
        patches_per_patient: int = 1,
        normalize_hu: bool = True,
    ):
        super().__init__()
        assert split in ("train", "val", "test")

        self.root = root
        self.split = split
        self.patch_size = patch_size
        self.max_tries = max_tries
        self.patches_per_patient = max(1, patches_per_patient)
        self.normalize_hu = normalize_hu

        self.seed = seed  # <--- ADD THIS LINE

        # Caches for full volumes
        #self._cache_cbct = {}
        #self._cache_ct = {}
        #self._cache_mask = {}

        # RNG for patch sampling
        self._torch_rng = torch.Generator()
        self._torch_rng.manual_seed(seed)

        # -------------------------------
        # 1) Collect all patients
        # -------------------------------
        all_patients = collect_patients(root, cohorts=cohorts)
        if len(all_patients) == 0:
            raise RuntimeError(f"No patients found under {root}")

        # -------------------------------
        # 2) Train/val/test split
        # -------------------------------
        train_pat, val_pat, test_pat = split_patients_train_val_test(
            all_patients,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
            seed=seed
        )

        if split == "train":
            self.patients = train_pat
        elif split == "val":
            self.patients = val_pat
        else:
            self.patients = test_pat

        if len(self.patients) == 0:
            raise RuntimeError(f"Split '{split}' has 0 patients! Check split fractions.")

        self.n_patients = len(self.patients)
        self._length = self.n_patients * self.patches_per_patient

        print(f"[VolumePatchDataset3D] Split={split} | Patients={self.n_patients} | Length={self._length}")

    def __len__(self):
        return self._length

    # ------------------------------
    # Loading volumes with caching
    # ------------------------------

    def _load_volume(self, path: str) -> np.ndarray:
        """
        Load volume (raw float32) with caching.
        """
        #if path.endswith("cbct.mha"):
            #cache = self._cache_cbct
        #else:
            #cache = self._cache_ct
                    
        #if path not in cache:
            #img = sitk.ReadImage(path)
            #cache[path] = sitk.GetArrayFromImage(img).astype(np.float32)
        #return cache[path]
    
        img = sitk.ReadImage(path)
        return sitk.GetArrayFromImage(img).astype(np.float32)

    def _load_mask(self, path: str) -> np.ndarray:
        #if path not in self._cache_mask:
            #img = sitk.ReadImage(path)
            #self._cache_mask[path] = sitk.GetArrayFromImage(img).astype(np.uint8)
        #return self._cache_mask[path]
        
        img = sitk.ReadImage(path)
        return sitk.GetArrayFromImage(img).astype(np.uint8)

    # ------------------------------
    # Patch sampling helpers
    # ------------------------------

    def _sample_patch_corner(self, vol_shape, idx):
        D, H, W = vol_shape
        pD, pH, pW = self.patch_size

        if self.split == "train":
            g = self._torch_rng
        else:
            g = torch.Generator().manual_seed(self.seed + int(idx))

        z0 = int(torch.randint(0, D - pD + 1, (1,), generator=g))
        y0 = int(torch.randint(0, H - pH + 1, (1,), generator=g))
        x0 = int(torch.randint(0, W - pW + 1, (1,), generator=g))
        return z0, y0, x0


    def _sample_valid_patch_corner(self, mask: np.ndarray, idx: int):
        D, H, W = mask.shape
        pD, pH, pW = self.patch_size

        # IMPORTANT: make a generator ONCE per item, not once per try
        if self.split == "train":
            g = self._torch_rng
        else:
            g = torch.Generator().manual_seed(self.seed + int(idx))  # fixed per idx

        for _ in range(self.max_tries):
            # sample NEW corners each try (deterministic for val/test because g is fixed)
            z0 = int(torch.randint(0, D - pD + 1, (1,), generator=g))
            y0 = int(torch.randint(0, H - pH + 1, (1,), generator=g))
            x0 = int(torch.randint(0, W - pW + 1, (1,), generator=g))

            zc = z0 + pD // 2
            yc = y0 + pH // 2
            xc = x0 + pW // 2

            mask_patch = mask[z0:z0+pD, y0:y0+pH, x0:x0+pW] # to reduce background-heavy patches
            if (mask[zc, yc, xc] > 0) and (mask_patch.mean() > 0.2):
                return z0, y0, x0

        print("⚠️  Could not find valid patch inside mask, using central patch")
        return (
            max(0, (D - pD) // 2),
            max(0, (H - pH) // 2),
            max(0, (W - pW) // 2)
        )


    # ------------------------------
    # __getitem__
    # ------------------------------

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        patient_index = idx % self.n_patients
        pinfo = self.patients[patient_index]

        cbct = self._load_volume(pinfo["cbct_path"])
        ct   = self._load_volume(pinfo["ct_path"])
        mask = self._load_mask(pinfo["mask_path"])

        # Get the dimension for coordinates
        D, H, W = cbct.shape

        assert cbct.shape == ct.shape == mask.shape
        z0, y0, x0 = self._sample_valid_patch_corner(mask, idx)

        pD, pH, pW = self.patch_size

        cbct_patch = cbct[z0:z0+pD, y0:y0+pH, x0:x0+pW]
        ct_patch   = ct[z0:z0+pD, y0:y0+pH, x0:x0+pW]
        mask_patch = mask[z0:z0+pD, y0:y0+pH, x0:x0+pW]

        # ---------------------------------------------------------
        # 4. NEW: Generate Normalized Coordinate Maps
        #    Range [-1, 1] relative to the original full volume
        # ---------------------------------------------------------
        
        # Z coordinates (Depth)
        z_range = np.arange(z0, z0 + pD, dtype=np.float32)
        z_norm = 2.0 * z_range / (D - 1) - 1.0  # Normalize to [-1, 1]

        # Y coordinates (Height)
        y_range = np.arange(y0, y0 + pH, dtype=np.float32)
        y_norm = 2.0 * y_range / (H - 1) - 1.0

        # X coordinates (Width)
        x_range = np.arange(x0, x0 + pW, dtype=np.float32)
        x_norm = 2.0 * x_range / (W - 1) - 1.0

        # Create 3D Meshgrid
        # indexing='ij' ensures dimensions are (Z, Y, X) order
        Z, Y, X = np.meshgrid(z_norm, y_norm, x_norm, indexing='ij')

        # Stack into a single tensor [3, pD, pH, pW]
        coords_patch = np.stack([Z, Y, X], axis=0).astype(np.float32)

        # ---------------------------------------------------------

        if self.normalize_hu:
            ct_patch = norm_ct(ct_patch, lo=-1024, hi=3000)
            cbct_patch = norm_ct(cbct_patch, lo=-1024, hi=3000)

        return {
            "CBCT": torch.from_numpy(cbct_patch).unsqueeze(0),
            "pCT":  torch.from_numpy(ct_patch).unsqueeze(0),
            "mask": torch.from_numpy(mask_patch).float(),
            "coords": torch.from_numpy(coords_patch),          # [3, D, H, W] <--- NEW
            "meta": {"cohort": pinfo["cohort"], "pid": pinfo["pid"]}
        }
